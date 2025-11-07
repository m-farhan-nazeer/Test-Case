"""
Retrieval service:
- add_document (with chunking, dedupe, embeddings)
- query_chunks (Milvus if available else SQLite cosine)
"""

import logging
from typing import List, Dict, Any, Optional
from math import sqrt

from ai.openai_integration import generate_openai_embedding
from database.database_functions import (
    get_db_connection, ensure_tables,
    upsert_document, insert_chunk, upsert_chunk_embedding,
    fetch_all_embeddings
)
from database.milvus_functions import ensure_milvus_connection, get_milvus_connection_status
from core.database import db_manager  # Milvus manager if available

logger = logging.getLogger(__name__)

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot/(na*nb)

def _maybe_milvus() -> bool:
    try:
        return db_manager.milvus_connected
    except Exception:
        return False

# ---------- Ingestion ----------

def add_document(
    *,
    title: str,
    content: str,
    summary: Optional[str],
    metadata: Optional[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    source_path: Optional[str] = None
) -> Dict[str, Any]:
    conn = get_db_connection()
    ensure_tables(conn)

    doc_id = upsert_document(
        conn,
        title=title, content=content, summary=summary,
        metadata=metadata, source_path=source_path
    )

    deduped = 0
    created = 0
    milvus_ok = _maybe_milvus()

    # Pre-create Milvus collection if not present
    if milvus_ok:
        try:
            db_manager.ensure_vector_collection()
        except Exception as e:
            logger.warning(f"Milvus ensure_vector_collection failed: {e}")
            milvus_ok = False

    for ch in chunks:
        res = insert_chunk(
            conn,
            document_id=doc_id,
            chunk_index=ch["chunk_index"],
            text=ch["text"],
            start_char=ch.get("start_char"),
            end_char=ch.get("end_char"),
        )
        if res.get("deduped"):
            deduped += 1
            # If deduped, we do NOT insert duplicate embedding again.
            continue

        created += 1

        # Embed chunk
        try:
            emb = generate_openai_embedding(res["text"])
        except Exception as e:
            logger.error(f"Embedding error on chunk {res['id']}: {e}")
            emb = None

        if emb:
            # SQLite store
            upsert_chunk_embedding(conn, chunk_id=res["id"], embedding=emb)

            # Milvus store (optional)
            if milvus_ok:
                try:
                    db_manager.insert_vectors([(res["id"], emb, {"document_id": doc_id, "title": title})])
                except Exception as e:
                    logger.warning(f"Milvus insert failed for chunk {res['id']}: {e}")
                    milvus_ok = False

    return {
        "document_id": doc_id,
        "chunks_created": created,
        "chunks_deduped": deduped,
        "milvus_used": milvus_ok
    }

# ---------- Query ----------

def query_chunks(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    q_emb = generate_openai_embedding(question)

    # If Milvus connected, try vector search
    if _maybe_milvus():
        try:
            results = db_manager.search_vectors(q_emb, top_k=top_k)
            # Map to unified format
            out = []
            for r in results:
                out.append({
                    "chunk_id": r["id"],
                    "score": float(r["score"]),
                    "text": r["payload"].get("text") or "",  # if you store text in Milvus payloads
                    "document_id": r["payload"].get("document_id"),
                    "title": r["payload"].get("title"),
                })
            if out:
                return out
        except Exception as e:
            logger.warning(f"Milvus search failed; falling back. {e}")

    # Fallback: cosine over SQLite
    conn = get_db_connection()
    vecs = fetch_all_embeddings(conn, limit=5000)
    scored = []
    for row in vecs:
        scored.append({
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "text": row["text"],
            "score": _cosine(q_emb, row["embedding"])
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
