# app/api/chat_routes.py
"""
Chatbot endpoints (Pydantic-typed for /chat/ask).
- Uses your sqlite3 helpers and history-aware OpenAI calls.
- Adds SSE streaming with POST /api/chat/ask/stream (EventSourceResponse).
- NEW: Auto-title a chat after the first assistant response.
"""
from utils import CitationIndex, render_retrieval_context, _title_with_expl
from retreival.retreive import retrieve_for_chat
# top of app/api/routes/chat.py
from retreival.milvus_store import retrieve_for_chat
from services.citations import CitationIndex,render_retrieval_context

from database.database_functions import (
    get_db_connection,
    ensure_chat,
    append_message,
    fetch_chat_messages,
    list_chats,
    delete_chat,
    trim_chat_history,
    has_documents_for_chat,      # ⬅️ NEW
    count_documents_for_chat,    # ⬅️ NEW (optional)
)
from ai.openai_integration import (
    generate_openai_answer_with_history,
    generate_openai_answer_with_history_stream,  # <--- streaming
    generate_openai_embedding,                    # ⬅️ we’ll use for question embedding
)
from services.retreival import retrieve_for_chat, render_retrieval_context
from typing import Any, Dict, List, Optional, AsyncGenerator
import logging
import json
import asyncio
import uuid
import time
from services.retreival import retrieve_top_chunks_for_chat
from fastapi import APIRouter, HTTPException, status, Body, Depends, Request
from sse_starlette.sse import EventSourceResponse
from services.citations import CitationIndex
from auth.authentication import get_current_user

# ✅ use your actual schema path
from models.chat import AskRequest, AskResponse, CreateDocumentResponse,RequestDocument
from endpoints.web_search_endpoints import enhance_context_with_web_search,_title_with_expl

# DB helpers
from database.database_functions import (
    get_db_connection,
    ensure_chat,
    append_message,
    fetch_chat_messages,
    list_chats,
    delete_chat,
    trim_chat_history,
)

# OpenAI integration (history-aware)
from ai.openai_integration import (
    generate_openai_answer_with_history,
    generate_openai_answer_with_history_stream,  # <--- streaming
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat", "documents"])
router = APIRouter(prefix="/api", tags=["chat"])

# ---------------------------
# Helpers
# ---------------------------

def _derive_title_from_seed(seed: str, max_words: int = 7) -> str:
    """Deterministic, fast title from the first user message (no LLM)."""
    clean = " ".join((seed or "").replace("\n", " ").split())
    clean = clean.lstrip(":-–—•*#[](){}<>\"' ")
    words = clean.split()
    t = " ".join(words[:max_words]) if words else "New chat"
    return t.rstrip(".!?,;:·•-–—") or "New chat"

MAX_QUESTION_CHARS = 10_000  # ~2.5k tokens rough; adjust if needed


def _require_db():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return conn


def _massage_message_row(chat_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sqlite row dict to plain message dict and parse JSON columns."""
    # base fields
    out = {
        "id": row.get("id"),
        "chat_id": chat_id,
        "role": row.get("role"),
        "content": row.get("content"),
        "model_used": row.get("model_used"),
        "tokens_used": row.get("tokens_used"),
        "metadata": row.get("metadata"),
        "sources_json": row.get("sources_json"),
        "citations_json": row.get("citations_json"),
        "created_at": row.get("created_at"),
    }

    # parsed convenience fields (non-breaking)
    import json
    def _parse_json(s):
        if not s:
            return None
        if isinstance(s, (dict, list)):
            return s
        try:
            return json.loads(s)
        except Exception:
            return None

    out["sources"] = _parse_json(out["sources_json"])        # array of full sources
    out["citations"] = _parse_json(out["citations_json"])    # array of {id,title,url}

    # also expose legacy metadata.citations parsed (if present)
    meta_obj = _parse_json(out["metadata"])
    out["citations_legacy"] = (meta_obj or {}).get("citations") if isinstance(meta_obj, dict) else None

    return out



def _ev(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shape a strict-JSON SSE event payload.
    We JSON-encode 'data' so frontends can safely JSON.parse(e.data).
    """
    return {"event": event_type, "data": json.dumps(payload, ensure_ascii=False)}


async def _auto_title_if_empty(conn, chat_id: str, seed_message: str, *, max_words: int = 7) -> Optional[str]:
    """
    If the chat has no title, generate one from the first user message.
    - Silent/soft failure: never blocks the main flow.
    """
    try:
        # Check current title
        cur = conn.cursor()
        cur.execute("SELECT title FROM chats WHERE id = ?", (chat_id,))
        row = cur.fetchone()
        cur.close()
        current_title = (row["title"] if row and "title" in row.keys() else (row[0] if row else None)) or ""
        if current_title.strip():
            return current_title  # already titled

        seed = (seed_message or "").strip()
        if not seed:
            # Fallback: first non-empty user message in DB
            rows = fetch_chat_messages(conn, chat_id)
            first_user = next((r for r in rows if (r.get("role") == "user" and (r.get("content") or "").strip())), None)
            seed = (first_user.get("content") or "").strip() if first_user else ""

        if not seed:
            return None  # nothing to summarize

        prompt = (
            "Create a concise conversation title (max "
            f"{max_words} words). No punctuation at the end. No quotes.\n\n"
            f"Message: {seed}\n\nTitle:"
        )

        llm = await generate_openai_answer_with_history(
            history_messages=[],
            user_question=prompt,
            system_prompt="You write very short, descriptive titles.",
            context=None,
        )
        raw_title = (llm.get("answer") or "").strip()

        # sanitize: one line, limit words
        title = " ".join(raw_title.replace("\n", " ").split())
        words = title.split()
        if len(words) > max_words:
            title = " ".join(words[:max_words])

        # persist
        ensure_chat(conn, chat_id=chat_id, title=title)
        return title
    except Exception:
        logger.warning("Auto-title failed (non-fatal).", exc_info=True)
        return None


# ---------------------------
# Endpoints
# ---------------------------

@router.post("/chat/ask", status_code=status.HTTP_200_OK, response_model=AskResponse)
async def ask_endpoint(payload: AskRequest) -> Dict[str, Any]:
    """
    Non-streaming endpoint with web citations:
      - augments system_prompt with [n] rules + SOURCES when web search is enabled
      - persists citations array to answer_message.metadata
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Field 'question' is required and cannot be empty")

    input_truncated = False
    original_len = len(question)
    if original_len > MAX_QUESTION_CHARS:
        slice_ = question[:MAX_QUESTION_CHARS]
        last_space = slice_.rfind(" ")
        if last_space > int(MAX_QUESTION_CHARS * 0.9):
            slice_ = slice_[:last_space]
        question = slice_
        input_truncated = True

    chat_id: Optional[str] = payload.chat_id
    user_id: Optional[str] = payload.user_id
    system_prompt: Optional[str] = payload.system_prompt
    context: str = (payload.context or "").strip()

    conn = _require_db()
    try:
        # ensure chat
        chat_id = ensure_chat(conn, chat_id=chat_id, title=None, user_id=user_id)

        # build prior history
        def _strip_legacy_system_prefix(text: str) -> str:
            if text.startswith("[system:"):
                sep = text.find("]\n")
                if sep != -1:
                    return text[sep + 2 :]
            return text

        prior_rows = fetch_chat_messages(conn, chat_id)
        history_messages: List[Dict[str, str]] = []
        for r in prior_rows:
            role = r.get("role")
            content = r.get("content") or ""
            if role == "user":
                content = _strip_legacy_system_prefix(content)
            if role in {"system", "user", "assistant"} and content.strip():
                history_messages.append({"role": role, "content": content})

        # store user message
        try:
            q_msg_id = append_message(
                conn,
                chat_id=chat_id,
                role="user",
                content=question,
                model_used=None,
                tokens_used=None,
                metadata=None,
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        # optionally enhance context + prepare SOURCES for citations
        effective_context = context
        web_sources: List[Dict[str, Any]] = []
        try:
            if getattr(payload, "web_search_enabled", False):
                ws_ctx = {
                    "webSearchEnabled": True,
                    "settings": getattr(payload, "web_search_settings", None) or {},
                }
                enhanced = await enhance_context_with_web_search(
                    question, existing_context=effective_context or "", context=ws_ctx
                )
                effective_context = enhanced.get("enhanced_context") or effective_context
                web_sources = enhanced.get("web_sources", []) or []

                if web_sources:
                    citation_map_lines = []
                    for s in web_sources:
                        citation_map_lines.append(f"[{s.get('id')}] {s.get('title','')}\nURL: {s.get('url','')}")
                    citation_map = "\n".join(citation_map_lines)

                    citation_instruction = (
                        "When using any facts from the sources below, add an inline citation marker like [1] "
                        "right after the relevant sentence. Do not invent citations or use URLs not listed here. "
                        "At the end of your answer, include a 'References' section that lists each numbered source "
                        "on its own line in the form: [n] Title — URL."
                    )

                    sys_block = (system_prompt + "\n\n" if system_prompt else "")
                    system_prompt = sys_block + citation_instruction + "\n\nSOURCES:\n" + citation_map
        except Exception as e:
            logger.warning("Web search context enhancement failed (continuing without it).", exc_info=True)

        # call LLM
        try:
            llm = await generate_openai_answer_with_history(
                history_messages=history_messages,
                user_question=question,
                system_prompt=system_prompt,
                context=effective_context,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LLM call failed")
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")

        answer = (llm.get("answer") or "").strip() or "Sorry — I couldn't generate a response."
        model_used = llm.get("model_used")
        tokens_used = llm.get("tokens_used")

        # persist assistant with metadata.citations
        assistant_metadata = None
        # persist assistant with metadata.citations + native columns
        assistant_metadata = None
        normalized_citations = []
        try:
            meta = {}
            if web_sources:
                # keep legacy path for existing front-ends
                meta["citations"] = web_sources

                # also prepare a normalized numbered list for DB column
                normalized_citations = [
                    {"id": i + 1, "title": s.get("title"), "url": s.get("url")}
                    for i, s in enumerate(web_sources)
                ]
            assistant_metadata = json.dumps(meta, ensure_ascii=False)
        except Exception:
            assistant_metadata = None

        try:
            a_msg_id = append_message(
                conn,
                chat_id=chat_id,
                role="assistant",
                content=answer,
                model_used=model_used,
                tokens_used=tokens_used if isinstance(tokens_used, int) else None,
                metadata=assistant_metadata,
                # NEW: persist into first-class columns too
                sources_json=web_sources or None,
                citations_json=normalized_citations or None,
            )
        except ValueError as ve:
            raise HTTPException(status_code=502, detail=f"Assistant message invalid: {ve}")

        # auto-title if empty
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM chats WHERE id = ?", (chat_id,))
            row = cursor.fetchone()
            cursor.close()
            current_title = (row["title"] if row and "title" in row.keys() else (row[0] if row else None)) or ""
            if not current_title.strip():
                title = _derive_title_from_seed(question, max_words=7)
                ensure_chat(conn, chat_id=chat_id, title=title)
        except Exception:
            logger.warning("Auto-title (non-stream) skipped due to an error", exc_info=True)

        # trim history
        try:
            trim_chat_history(conn, chat_id, keep_last=200)
        except Exception:
            logger.warning("History trim skipped due to an error", exc_info=True)

        # fetch just-stored messages for return
        rows = fetch_chat_messages(conn, chat_id) or []
        if not rows:
            raise HTTPException(status_code=500, detail="Failed to retrieve stored messages")
        q_row = next((r for r in rows if r["id"] == q_msg_id), rows[-2] if len(rows) >= 2 else rows[-1])
        a_row = next((r for r in rows if r["id"] == a_msg_id), rows[-1])

        return {
            "chat_id": chat_id,
            "question_message": _massage_message_row(chat_id, dict(q_row)),
            "answer_message": _massage_message_row(chat_id, dict(a_row)),
            "model_used": model_used,
            "tokens_used": tokens_used,
            "input_truncated": input_truncated,
            "original_question_length": original_len,
            "used_question_length": len(question),
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass



# ---- STREAMING (SSE) ---------------------------------------------------------

def _ev(event: str, data: Dict[str, Any]) -> dict:
    return {"event": event, "data": data}


@router.post("/chat/ask/stream")
async def ask_stream_endpoint(request: Request, payload: AskRequest):
    """
    SSE streaming variant of 'continue to chat'.

    Events (all JSON-encoded via _ev()):
      - event: start   data: {"chat_id","request_id","model_used"?: "..."}
      - event: saved   data: {"question_message_id": int} / {"answer_message_id": int}
      - event: meta    data: {...}
      - event: delta   data: {"content": "partial text"}
      - event: error   data: {"message": "...", "status_code"?: int}
      - event: done    data: {"chat_id","request_id","model_used","tokens_used","full_content_len",...}
    """
    # Validate & normalize question
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Field 'question' is required and cannot be empty")

    input_truncated = False
    original_len = len(question)
    if original_len > MAX_QUESTION_CHARS:
        slice_ = question[:MAX_QUESTION_CHARS]
        last_space = slice_.rfind(" ")
        if last_space > int(MAX_QUESTION_CHARS * 0.9):
            slice_ = slice_[:last_space]
        question = slice_
        input_truncated = True

    chat_id: Optional[str] = payload.chat_id
    user_id: Optional[str] = payload.user_id
    system_prompt: Optional[str] = payload.system_prompt
    context: str = (payload.context or "").strip()

    conn = _require_db()  # open DB once; closed in finally inside generator
    logger = logging.getLogger(__name__)

    async def event_generator() -> AsyncGenerator[dict, None]:
        request_id = f"req_{uuid4().hex[:12]}"
        model_used: Optional[str] = None
        tokens_used: Optional[int] = None
        a_msg_id: Optional[int] = None
        q_msg_id: Optional[int] = None
        full_answer_parts: List[str] = []

        # capture for DB persistence (both user and assistant messages)
        web_sources_captured: List[Dict[str, Any]] = []
        normalized_citations_captured: List[Dict[str, Any]] = []

        try:
            # 1) Ensure chat (reuse chat_id if provided)
            nonlocal chat_id
            chat_id = ensure_chat(conn, chat_id=chat_id, title=None, user_id=user_id)

            # 🔒 RAG gate for this chat
            rag_enabled = has_documents_for_chat(conn, chat_id)
            attached_docs = count_documents_for_chat(conn, chat_id) if rag_enabled else 0

            # Emit early meta so UI can show “RAG off/on”
            yield _ev("meta", {
                "stage": "rag_gate",
                "rag_enabled": rag_enabled,
                "documents_attached": attached_docs,
            })

            # 2) Build conversation history (chronological)
            def _strip_legacy_system_prefix(text: str) -> str:
                if text.startswith("[system:"):
                    sep = text.find("]\n")
                    if sep != -1:
                        return text[sep + 2 :]
                return text

            prior_rows = fetch_chat_messages(conn, chat_id)
            history_messages: List[Dict[str, str]] = []
            for r in prior_rows:
                role = r.get("role")
                content = r.get("content") or ""
                if role == "user":
                    content = _strip_legacy_system_prefix(content)
                if role in {"system", "user", "assistant"} and content.strip():
                    history_messages.append({"role": role, "content": content})

            # 3) Start with provided context; optionally enhance via web search
            effective_context = context
            try:
                if getattr(payload, "web_search_enabled", False):
                    yield _ev("meta", {"stage": "web_search", "progress": 10, "message": "Searching the web..."})

                    ws_ctx = {
                        "webSearchEnabled": True,
                        "settings": getattr(payload, "web_search_settings", None) or {},
                    }

                    enhanced = await enhance_context_with_web_search(
                        question, existing_context=effective_context or "", context=ws_ctx
                    )

                    # merge web context
                    effective_context = enhanced.get("enhanced_context") or effective_context

                    # collect sources for DB + UI
                    web_sources = enhanced.get("web_sources", []) or []
                    web_sources_captured = web_sources

                    # emit completion + small preview
                    yield _ev("meta", {
                        "stage": "web_search",
                        "progress": 100,
                        "message": enhanced.get("web_search_message", "Web search complete"),
                        "total_web_results": enhanced.get("total_web_results", len(web_sources)),
                        "web_sources": web_sources[:3],
                    })
            except Exception as e:
                # Non-fatal; continue without web context
                logger.warning("Web search context enhancement failed (continuing without it).", exc_info=True)
                yield _ev("meta", {"stage": "web_search", "error": str(e)})

            # 4) If RAG is enabled, compute question embedding
            question_embedding: Optional[List[float]] = None
            if rag_enabled:
                try:
                    question_embedding = await generate_openai_embedding(
                        question, cache_key=f"qemb_{chat_id}"
                    )
                    yield _ev("meta", {"stage": "embeddings", "message": "Question embedding computed"})
                except Exception as e:
                    # If embedding fails, emit and proceed sans-RAG
                    rag_enabled = False
                    yield _ev("meta", {
                        "stage": "embeddings",
                        "error": f"Embedding failed; continuing without RAG: {str(e)}",
                    })

            # 5) Retrieval (only if RAG still enabled and we have an embedding)
            retrieval: Dict[str, Any] = {"chunks": [], "stats": {}}
            if rag_enabled and question_embedding:
                try:
                    retrieval = await retrieve_for_chat(
                        conn, chat_id, question_embedding, top_k=6, min_score=0.20
                    )
                    yield _ev("meta", {
                        "stage": "retrieval",
                        "progress": 100,
                        "engine": retrieval.get("stats", {}).get("engine"),
                        "returned": retrieval.get("stats", {}).get("returned", 0),
                        "top_k": retrieval.get("stats", {}).get("top_k"),
                        "min_score": retrieval.get("stats", {}).get("min_score"),
                    })
                except Exception as e:
                    yield _ev("meta", {"stage": "retrieval", "error": f"Retrieval failed: {str(e)}"})
                    retrieval = {"chunks": [], "stats": {"engine": "none", "returned": 0}}

            # 6) Unified citations (web + RAG) with stable ids
            citations_ui: List[Dict[str, Any]] = []
            # first: web sources
            for i, s in enumerate(web_sources_captured[:10], start=1):
                citations_ui.append({
                    "id": i,
                    "type": "web",
                    "title": _title_with_expl(s, question),
                    "url": s.get("url"),
                })

            # build a map for (doc_id, chunk_id) -> id
            citation_id_by_chunk: Dict[Tuple[str, int], int] = {}
            next_id = len(citations_ui) + 1

            # then: RAG chunks
            for ch in retrieval.get("chunks", []):
                key = (str(ch.get("doc_id")), int(ch.get("chunk_id")))
                if key in citation_id_by_chunk:
                    continue
                cid = next_id
                next_id += 1
                citation_id_by_chunk[key] = cid
                meta = ch.get("source") or {}
                citations_ui.append({
                    "id": cid,
                    "type": "rag",
                    "doc_id": key[0],
                    "chunk_id": key[1],
                    "title": meta.get("title") or meta.get("doc_name") or "Attached document",
                    "url": meta.get("url"),
                    "page": meta.get("page"),
                })

            # emit unified citations to the UI
            if citations_ui:
                yield _ev("meta", {
                    "stage": "citations",
                    "progress": 100,
                    "count": len(citations_ui),
                    "citations": citations_ui[:10],  # keep payload light
                })
            normalized_citations_captured = citations_ui  # save to DB later

            # 7) Render retrieved context with correct [n] ids & merge into effective_context
            retrieval_context_block = render_retrieval_context(
                retrieval.get("chunks", []),
                citation_id_by_chunk=citation_id_by_chunk,
                max_chars_per_chunk=900,
                max_total_chars=4000,
            )
            if retrieval_context_block:
                effective_context = ((effective_context + "\n\n") if effective_context else "") + retrieval_context_block

            # 8) Persist user message (now we have sources/citations ready)
            try:
                q_msg_id = append_message(
                    conn,
                    chat_id=chat_id,
                    role="user",
                    content=question,
                    model_used=None,
                    tokens_used=None,
                    metadata=None,
                    sources_json=web_sources_captured or None,
                    citations_json=normalized_citations_captured or None,
                )
                yield _ev("saved", {"question_message_id": q_msg_id})
            except ValueError as ve:
                yield _ev("error", {"message": f"Invalid user message: {ve}"})
                return

            # 9) Announce stream start
            yield _ev("start", {"chat_id": chat_id, "request_id": request_id})

            # 10) Stream from LLM
            async for chunk in generate_openai_answer_with_history_stream(
                history_messages=history_messages,
                user_question=question,
                system_prompt=system_prompt,
                context=effective_context,
            ):
                ctype = chunk.get("type")

                if ctype == "start":
                    model_used = chunk.get("model_used")
                    if model_used:
                        yield _ev("start", {"chat_id": chat_id, "request_id": request_id, "model_used": model_used})

                elif ctype == "delta":
                    text = chunk.get("content") or ""
                    if text:
                        full_answer_parts.append(text)
                        yield _ev("delta", {"content": text})

                elif ctype == "meta":
                    meta_payload = {k: v for k, v in chunk.items() if k != "type"}
                    if meta_payload:
                        yield _ev("meta", meta_payload)

                elif ctype == "done":
                    tokens_used = chunk.get("tokens_used")

                elif ctype == "error":
                    yield _ev("error", {"message": chunk.get("message", "Unknown LLM error")})
                    return

            # 11) Persist assistant message (with retrieval & citations meta)
            answer = "".join(full_answer_parts).strip() or "Sorry — I couldn't generate a response."
            try:
                a_msg_id = append_message(
                    conn,
                    chat_id=chat_id,
                    role="assistant",
                    content=answer,
                    model_used=model_used,
                    tokens_used=tokens_used if isinstance(tokens_used, int) else None,
                    metadata=json.dumps({
                        "rag_enabled": rag_enabled,
                        "documents_attached": attached_docs,
                        "retrieval_stats": (retrieval.get("stats") if isinstance(retrieval, dict) else {}) or {},
                        "retrieved_chunk_ids": [
                            {"doc_id": ch.get("doc_id"), "chunk_id": ch.get("chunk_id"), "score": ch.get("score")}
                            for ch in (retrieval.get("chunks", []) if isinstance(retrieval, dict) else [])
                        ],
                        "citations_count": len(normalized_citations_captured or []),
                    }, ensure_ascii=False),
                    sources_json=web_sources_captured or None,
                    citations_json=normalized_citations_captured or None,
                )
                yield _ev("saved", {"answer_message_id": a_msg_id})
            except ValueError as ve:
                yield _ev("error", {"message": f"Assistant message invalid: {ve}"})
                return

            # 12) Auto-title on first turn if empty (deterministic, no LLM)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT title FROM chats WHERE id = ?", (chat_id,))
                row = cursor.fetchone()
                cursor.close()
                current_title = (row["title"] if row and "title" in row.keys() else (row[0] if row else None)) or ""
                if not current_title.strip():
                    title = _derive_title_from_seed(question, max_words=7)
                    ensure_chat(conn, chat_id=chat_id, title=title)
            except Exception:
                logger.warning("Auto-title (stream) skipped due to an error", exc_info=True)

            # 13) Trim history (best-effort)
            try:
                trim_chat_history(conn, chat_id, keep_last=200)
            except Exception:
                logger.warning("History trim skipped due to an error", exc_info=True)

            # 14) Done
            yield _ev("done", {
                "chat_id": chat_id,
                "request_id": request_id,
                "model_used": model_used,
                "tokens_used": tokens_used,
                "full_content_len": sum(len(p) for p in full_answer_parts),
                "input_truncated": input_truncated,
                "original_question_length": original_len,
                "used_question_length": len(question),
            })

        except HTTPException as he:
            yield _ev("error", {"message": he.detail, "status_code": he.status_code})
        except Exception as e:
            logger.exception("SSE stream failed")
            yield _ev("error", {"message": f"LLM/Server error: {e}"})
        finally:
            try:
                conn.close()
            except Exception:
                pass

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        ping=15000,  # keepalive every 15s
    )


# ------------------ Non-streaming utility endpoints (unchanged) ---------------

@router.get("/chat/{chat_id}/messages", status_code=status.HTTP_200_OK)
def get_chat_messages(chat_id: str) -> Dict[str, Any]:
    """Return all messages in a chat (chronological)."""
    conn = _require_db()
    try:
        rows = fetch_chat_messages(conn, chat_id)
        if rows is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {
            "chat_id": chat_id,
            "messages": [_massage_message_row(chat_id, dict(r)) for r in rows],
            "count": len(rows),
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.get("/chats", status_code=status.HTTP_200_OK)
def get_chats(limit: int = 50, offset: int = 0, user_id: Optional[str] = None) -> Dict[str, Any]:
    """List chats (most recent first) with simple pagination."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    conn = _require_db()
    try:
        items = list_chats(conn, user_id=user_id, limit=limit, offset=offset)
        # total count
        try:
            cursor = conn.cursor()
            if user_id:
                cursor.execute("SELECT COUNT(*) FROM chats WHERE user_id = ?", (user_id,))
            else:
                cursor.execute("SELECT COUNT(*) FROM chats")
            total = int(cursor.fetchone()[0])
            cursor.close()
        except Exception:
            total = len(items)
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.delete("/chat/{chat_id}", status_code=status.HTTP_200_OK)
def delete_chat_endpoint(chat_id: str) -> Dict[str, Any]:
    """Delete a chat and its messages."""
    conn = _require_db()
    try:
        deleted = delete_chat(conn, chat_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"deleted": True, "chat_id": chat_id}
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.post("/chat/{chat_id}/auto-title", status_code=status.HTTP_200_OK)
async def _auto_title_if_empty(conn, chat_id: str, seed_message: str, *, max_words: int = 7) -> Optional[str]:
    """
    If the chat has no title, generate one from the first user message.
    Tries LLM first; if that fails, falls back to a deterministic snippet of the seed.
    """
    try:
        # Check current title
        cur = conn.cursor()
        cur.execute("SELECT title FROM chats WHERE id = ?", (chat_id,))
        row = cur.fetchone()
        cur.close()
        current_title = (row["title"] if row and "title" in row.keys() else (row[0] if row else None)) or ""
        if current_title.strip():
            return current_title  # already titled

        # Find a seed
        seed = (seed_message or "").strip()
        if not seed:
            rows = fetch_chat_messages(conn, chat_id)
            first_user = next(
                (r for r in rows if (r.get("role") == "user" and (r.get("content") or "").strip())),
                None
            )
            seed = (first_user.get("content") or "").strip() if first_user else ""

        if not seed:
            return None  # nothing to base a title on

        # -------- try LLM title first --------
        llm_title: Optional[str] = None
        try:
            prompt = (
                "Create a concise conversation title (max "
                f"{max_words} words). No punctuation at the end. No quotes.\n\n"
                f"Message: {seed}\n\nTitle:"
            )
            llm = await generate_openai_answer_with_history(
                history_messages=[],
                user_question=prompt,
                system_prompt="You write very short, descriptive titles.",
                context=None,
            )
            raw = (llm.get("answer") or "").strip()
            if raw:
                t = " ".join(raw.replace("\n", " ").split())
                words = t.split()
                if len(words) > max_words:
                    t = " ".join(words[:max_words])
                llm_title = t
        except Exception:
            # swallow and use fallback
            llm_title = None

        # -------- deterministic fallback --------
        if not llm_title:
            # take first max_words words from seed
            # 1) normalize whitespace
            clean = " ".join(seed.replace("\n", " ").split())
            # 2) drop leading punctuation/noise
            clean = clean.lstrip(":-–—•*#[](){}<>\"' ")
            # 3) cut at max_words
            words = clean.split()
            t = " ".join(words[:max_words]) if words else "New chat"
            # 4) trim trailing punctuation
            t = t.rstrip(".!?,;:·•-–—")
            llm_title = t if t else "New chat"

        # persist
        ensure_chat(conn, chat_id=chat_id, title=llm_title)
        return llm_title
    except Exception:
        logger.warning("Auto-title failed (non-fatal).", exc_info=True)
        return None
@router.get("/chat/{chat_id}/export", status_code=status.HTTP_200_OK)
def export_chat(chat_id: str) -> Dict[str, Any]:
    """
    Export a chat (metadata + messages) as JSON.
    """
    conn = _require_db()
    try:
        # metadata
        cur = conn.cursor()
        cur.execute("SELECT id, title, user_id, created_at, updated_at FROM chats WHERE id = ?", (chat_id,))
        chat_row = cur.fetchone()
        if not chat_row:
            cur.close()
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_meta = {
            "id": chat_row["id"],
            "title": chat_row["title"],
            "user_id": chat_row["user_id"],
            "created_at": chat_row["created_at"],
            "updated_at": chat_row["updated_at"],
        }
        cur.close()

        # messages
        rows = fetch_chat_messages(conn, chat_id) or []
        messages = [_massage_message_row(chat_id, dict(r)) for r in rows]

        return {
            "chat": chat_meta,
            "messages": messages,
            "count": len(messages),
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.get("/my/chats", status_code=status.HTTP_200_OK)
def get_my_chats(
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List chats for the authenticated user (most recent first).
    Uses the 'user_id' from the auth token.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    user_id = current_user.get("userId") or current_user.get("id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Authenticated user has no userId in token")

    conn = _require_db()
    try:
        items = list_chats(conn, user_id=user_id, limit=limit, offset=offset)

        # total rows for this user
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chats WHERE user_id = ?", (user_id,))
            total = int(cursor.fetchone()[0])
            cursor.close()
        except Exception:
            total = len(items)

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "user_id": user_id,
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.post("/chat/{chat_id}/title", status_code=status.HTTP_200_OK)
def set_chat_title(chat_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Manually set a chat title. Won't accept empty/whitespace titles.
    {
      "title": "My custom title"
    }
    """
    title = (payload.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title cannot be empty")

    conn = _require_db()
    try:
        # verify chat exists
        cur = conn.cursor()
        cur.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            raise HTTPException(status_code=404, detail="Chat not found")

        # update via ensure_chat (idempotent)
        ensure_chat(conn, chat_id=chat_id, title=title)

        return {"chat_id": chat_id, "title": title}
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.get("/chat/{chat_id}/messages/paged", status_code=status.HTTP_200_OK)
def get_chat_messages_paged(
    chat_id: str,
    limit: int = 50,
    before_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Cursor-style pagination for messages.
    - Returns messages in chronological order (oldest -> newest) for UI render.
    - Use `before_id` as a cursor to fetch older messages than that message id.
    """
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

    conn = _require_db()
    try:
        cur = conn.cursor()
        # Verify chat exists
        cur.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,))
        if cur.fetchone() is None:
            cur.close()
            raise HTTPException(status_code=404, detail="Chat not found")

        # Grab newest first, then reverse to chronological
        params: List[Any] = [chat_id]
        where_more = ""
        if before_id is not None:
            where_more = " AND id < ?"
            params.append(before_id)

        cur.execute(
            f"""
            SELECT id, role, content, model_used, tokens_used, metadata,
                sources_json, citations_json, created_at
            FROM messages
            WHERE chat_id = ? {where_more}
            ORDER BY id DESC
            LIMIT ?
            """,
            (*params, limit),
        )

        rows = cur.fetchall()
        cur.close()

        rows = list(rows)[::-1]  # chronological

        messages = [_massage_message_row(chat_id, dict(r)) for r in rows]

        # Determine next cursor (older page)
        next_before_id = rows[0]["id"] if rows else None  # smallest id in this page
        has_more = False
        if next_before_id is not None:
            cur2 = conn.cursor()
            cur2.execute(
                "SELECT 1 FROM messages WHERE chat_id = ? AND id < ? LIMIT 1",
                (chat_id, next_before_id),
            )
            has_more = cur2.fetchone() is not None
            cur2.close()
            if not has_more:
                next_before_id = None

        return {
            "chat_id": chat_id,
            "messages": messages,
            "count": len(messages),
            "next_before_id": next_before_id,
            "has_more": has_more,
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass
# api/chats_router.py
# ------------------------------------------------- document -------------------------------------------------
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ai.openai_integration import generate_openai_embedding
from database.database_functions import (
    get_db_connection,
    save_document_with_vector,
    ensure_chat,            # ✅ ensure chat row exists or create it
)

from models.chat import RequestDocument, CreateDocumentResponse  # ✅ import your Pydantic models

# Reuse existing router for document endpoints

def _generate_chat_id() -> str:
    return f"chat_{uuid4().hex}"

@router.post("/documents", response_model=CreateDocumentResponse)
async def create_document(payload: RequestDocument) -> CreateDocumentResponse:
    """
    Create a document and attach it to a chat.
    - If payload.chat_id is missing, generate one.
    - Always ensure the chat row exists (create if missing).
    - Store content in SQLite (BLOB) and upsert vector into Milvus (DocumentsV2).
    - Roll back SQLite insert if Milvus upsert fails.
    """
    # 1) chat_id (generate if missing)
    chat_id = (payload.chat_id or _generate_chat_id()).strip()

    # 2) document_id (UUID4)
    document_id = uuid4()

    # 3) locals
    created_at_iso = datetime.now(timezone.utc).isoformat()
    title   = payload.document_title
    content = payload.content
    summary = payload.summary or ""
    metadata = payload.metadata or {}

    # 4) ensure the chat exists (safe + idempotent)
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database unavailable")
    try:
        # You may optionally pass a title/user_id; here we at least ensure the row
        ensure_chat(conn, chat_id=chat_id, title=title or "Chat")  # will create if not found

        # 5) generate embedding (title + content + summary)
        embed_text = f"{title}\n\n{content}\n\n{summary}".strip()
        embedding = await generate_openai_embedding(embed_text)
        if not isinstance(embedding, list) or not embedding:
            raise HTTPException(status_code=500, detail="Invalid embedding generated")

        # 6) persist (SQLite insert + Milvus upsert with rollback on failure)
        await save_document_with_vector(
            conn,
            document_id=document_id,
            chat_id=chat_id,
            title=title,
            content=content,      # stored as BLOB
            summary=summary,
            metadata=metadata,
            embedding=embedding,
            created_at=created_at_iso,
        )

        return CreateDocumentResponse(document_id=document_id, chat_id=chat_id)

    finally:
        try:
            conn.close()
        except Exception:
            pass
from typing import List, Dict, Any
from database.database_functions import get_db_connection

@router.get("/documents/by-chat/{chat_id}")
def list_documents_by_chat(chat_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    List recent documents for a given chat_id from SQLite only.
    Compatible with legacy and new schemas.
    """
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(documents);")
        cols = {r[1] for r in cur.fetchall()}

        select_cols = []
        if "document_id" in cols:
            select_cols.append("document_id AS doc_id")
        elif "id" in cols:
            select_cols.append("id AS doc_id")
        else:
            select_cols.append("NULL AS doc_id")

        if "title" in cols:
            select_cols.append("title")
        if "created_at" in cols:
            # force TEXT to avoid Python's TIMESTAMP converter blowing up on ISO8601 with 'T'
            select_cols.append("CAST(created_at AS TEXT) AS created_at")
        if "summary" in cols:
            select_cols.append("summary")
        if "metadata_json" in cols:
            select_cols.append("metadata_json")

        sql = f"SELECT {', '.join(select_cols)} FROM documents WHERE chat_id = ? ORDER BY rowid DESC LIMIT ?"
        cur.execute(sql, (chat_id, limit))
        rows = cur.fetchall()
        cur.close()

        col_aliases = [c.split(" AS ")[-1] for c in select_cols]
        docs: List[Dict[str, Any]] = [dict(zip(col_aliases, r)) for r in rows]

        return {"chat_id": chat_id, "count": len(docs), "documents": docs}
    finally:
        try:
            conn.close()
        except Exception:
            pass
from fastapi import Path

@router.get("/documents/{doc_id}/content")
def get_document_content_legacy_aware(
    doc_id: str = Path(..., description="UUID for new schema, or integer id for legacy")
) -> Dict[str, Any]:
    """
    Return the document content from SQLite only.
    - If 'content_blob' exists, decode bytes as UTF-8.
    - Else fall back to legacy 'content' TEXT column.
    Compatible with both old/new schemas.
    """
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    try:
        cur = conn.cursor()
        # discover columns
        cur.execute("PRAGMA table_info(documents);")
        cols = {r[1] for r in cur.fetchall()}

        # figure out key column and coerce doc_id for legacy int PK
        key_col = "document_id" if "document_id" in cols else ("id" if "id" in cols else None)
        if not key_col:
            raise HTTPException(status_code=500, detail="documents table has no primary key column")

        key_val = doc_id
        if key_col == "id":
            try:
                key_val = int(doc_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Legacy documents use integer id")

        # build select
        select_cols = []
        if "title" in cols: select_cols.append("title")
        if "summary" in cols: select_cols.append("summary")
        if "metadata_json" in cols: select_cols.append("metadata_json")
        # always try to fetch both content_blob and content; one may be NULL
        if "content_blob" in cols: select_cols.append("content_blob")
        if "content" in cols: select_cols.append("content")
        if "created_at" in cols: select_cols.append("CAST(created_at AS TEXT) AS created_at")

        sql = f"SELECT {', '.join(select_cols)} FROM documents WHERE {key_col} = ? LIMIT 1"
        cur.execute(sql, (key_val,))
        row = cur.fetchone()
        cur.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        # map columns back
        col_aliases = [c.split(" AS ")[-1] for c in select_cols]
        data = dict(zip(col_aliases, row))

        # decode content (prefer blob)
        text = None
        blob = data.get("content_blob")
        if isinstance(blob, (bytes, bytearray)):
            try:
                text = blob.decode("utf-8", errors="replace")
            except Exception:
                text = None
        if text is None:
            txt = data.get("content")
            text = txt if isinstance(txt, str) else ""

        return {
            "doc_id": doc_id,
            "title": data.get("title"),
            "summary": data.get("summary"),
            "created_at": data.get("created_at"),
            "metadata": data.get("metadata_json"),
            "content": text,
            "source": "content_blob" if isinstance(blob, (bytes, bytearray)) else "content",
        }
    finally:
        try: conn.close()
        except Exception: pass
# ---------------------------------------------------------------retriever-----------------------------
from retreival.index_milvus import upsert_chunk_embeddings

chunks = [(row["chunk_id"], row["text"]) for row in chunk_rows]  # List[Tuple[int, str]]

inserted = await upsert_chunk_embeddings(
    chat_id=chat_id,
    doc_id=new_doc_id,
    chunks=chunks,
    replace_existing=True,  # okay on re-uploads
)
