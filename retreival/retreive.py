# app/retrieval/retrieve.py
from typing import Any, Dict, List, Optional, Tuple

from .milvus_store import milvus_search_for_chat, MILVUS_ENABLED
from .sqlite_meta import fetch_chunk_texts

async def retrieve_for_chat(
    conn,
    chat_id: str,
    question_embedding: List[float],
    top_k: int = 6,
    min_score: float = 0.20,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "chunks": [
          {"doc_id","chunk_id","text","page","score","source":{...}}
        ],
        "context_text": "joined compact text",   # convenience
        "stats": {"engine":"milvus"|"none","returned":N,"top_k":top_k,"min_score":min_score}
      }
    """
    stats = {"engine": "none", "returned": 0, "top_k": top_k, "min_score": min_score}
    chunks: List[Dict[str, Any]] = []

    hits = milvus_search_for_chat(question_embedding, chat_id, top_k=top_k, min_score=min_score)
    if hits:
        stats["engine"] = "milvus"
        # hydrate texts from SQLite
        pairs = [(h["doc_id"], int(h["chunk_id"])) for h in hits]
        hydrated = fetch_chunk_texts(conn, chat_id, pairs)
        # map for quick merge
        score_by_pair = {(h["doc_id"], int(h["chunk_id"])): float(h["score"]) for h in hits}

        for h in hydrated:
            key = (h["doc_id"], int(h["chunk_id"]))
            chunks.append({
                **h,
                "score": score_by_pair.get(key, 0.0),
            })

        # keep order by score desc
        chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    stats["returned"] = len(chunks)

    # Convenience field for your earlier “retrieval['context_text']” usage
    joined = "\n".join((c.get("text") or "").strip() for c in chunks if c.get("text"))

    return {"chunks": chunks, "context_text": joined, "stats": stats}
