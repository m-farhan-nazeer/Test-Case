# app/retrieval/milvus_store.py
import os
from typing import Any, Dict, List, Optional, Tuple
from database.database_functions import milvus_search_for_chat, upsert_embeddings,fetch_chunk_texts_by_ids
MILVUS_ENABLED = os.getenv("MILVUS_ENABLED", "1") == "1"

if MILVUS_ENABLED:
    try:
        from pymilvus import (
            connections, utility, FieldSchema, CollectionSchema, DataType, Collection
        )
    except Exception as e:
        MILVUS_ENABLED = False
        _IMPORT_ERR = e  # optional: log this in your app logger


def _embedding_dim() -> int:
    try:
        return int(os.getenv("EMBEDDING_DIM", "1536"))
    except Exception:
        return 1536


def _collection_name() -> str:
    # single shared collection; filter by chat_id in the query expr
    return os.getenv("MILVUS_COLLECTION", "rag_chunks")


def _connect() -> None:
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    token = os.getenv("MILVUS_TOKEN")  # optional
    connections.connect("default", uri=uri, token=token)


def _ensure_collection(dim: int) -> "Collection":
    name = _collection_name()
    if utility.has_collection(name):
        return Collection(name)

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chat_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="score_hint", dtype=DataType.FLOAT),  # optional
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="RAG chunks (by chat_id)")
    col = Collection(name, schema, consistency_level="Bounded")
    # IVF_FLAT is simple to start with; tweak later
    col.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
    )
    return col


def milvus_search_for_chat(
    embedding: List[float],
    chat_id: str,
    top_k: int = 6,
    min_score: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
      { "doc_id": str, "chunk_id": int, "score": float }
    """
    if not MILVUS_ENABLED:
        return []

    _connect()
    dim = _embedding_dim()
    if not embedding or len(embedding) != dim:
        # wrong dimensionality — return empty so caller can soft-fail
        return []

    col = _ensure_collection(dim)
    col.load()
    # Filter to a single chat’s attached chunks
    expr = f'chat_id == "{chat_id}"'
    # Use inner-product (cosine proxy when vectors are normalized)
    res = col.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 32}},
        limit=top_k,
        expr=expr,
        output_fields=["doc_id", "chunk_id", "score_hint"],
    )

    hits = []
    for hit in res[0]:
        score = float(hit.distance)
        if score < min_score:
            continue
        doc_id = str(hit.entity.get("doc_id"))
        chunk_id = int(hit.entity.get("chunk_id"))
        hits.append({"doc_id": doc_id, "chunk_id": chunk_id, "score": score})
    return hits

# Retrieves top-K from Milvus, then hydrates the full text from your SQL (no schema change), and returns a shape compatible with what you already wired (chunks + context_text + stats).

# app/rag/retrieval_milvus.py
from typing import Dict, Any, List, Optional
from database.database_functions import search_embeddings
from database.database_functions import fetch_chunk_texts_by_ids


async def ingest_document_embeddings_for_chat(chat_id: str, doc_id: int,
                                              chunk_ids: List[int], embeddings: List[List[float]]):
    """
    You already computed `embeddings` for each chunk in order.
    """
    rows = []
    for chunk_id, vec in zip(chunk_ids, embeddings):
        rows.append((doc_id, int(chunk_id), vec))
    inserted = upsert_embeddings(chat_id, rows)
    return inserted


# app/rag/retrieval_milvus.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import MilvusClient, DataType  # pip install pymilvus

# ---- Config (override via env) ----
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")  # for Zilliz Cloud use "<username>:<password>" or API key
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "nx_chunks")
MILVUS_PARTITION_BY_CHAT = os.getenv("MILVUS_PARTITION_BY_CHAT", "1") in ("1", "true", "True")

# Field names expected in the Milvus schema
F_DOC_ID = os.getenv("MILVUS_FIELD_DOC_ID", "doc_id")          # Int64
F_CHUNK_ID = os.getenv("MILVUS_FIELD_CHUNK_ID", "chunk_id")    # Int64
F_CHAT_ID = os.getenv("MILVUS_FIELD_CHAT_ID", "chat_id")       # VarChar
F_VECTOR = os.getenv("MILVUS_FIELD_VECTOR", "embedding")       # FloatVector
F_SCORE = "distance"  # returned by search()

# ---- Client singleton ----
_client: Optional[MilvusClient] = None

def _client_or_init() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN or None)
    return _client


def _build_expr(chat_id: Optional[str]) -> Optional[str]:
    """
    Restrict search to this chat's partition/expr when available.
    Adjust if you use collections instead of partitions for tenanting.
    """
    if MILVUS_PARTITION_BY_CHAT and chat_id:
        # If you actually use Milvus partitions named = chat_id, the SDK search can take 'partition_names=[chat_id]'
        # We still keep an expr fallback in case schema also stores chat_id as a field.
        return f'{F_CHAT_ID} == "{chat_id}"'
    return None


def _pairs_from_results(results: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for r in results:
        try:
            pairs.append((int(r[F_DOC_ID]), int(r[F_CHUNK_ID])))
        except Exception:
            continue
    # keep order but unique
    seen = set()
    ordered: List[Tuple[int, int]] = []
    for p in pairs:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def retrieve_for_chat(
    conn,
    chat_id: Optional[str],
    question_embedding: List[float],
    top_k: int = 6,
    min_score: float = 0.20,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "chunks": [ { "doc_id": int, "chunk_id": int, "score": float }, ... ],
        "stats": { "engine": "milvus", "returned": int, "top_k": int, "min_score": float },
        "context_text": "…optional stitched text…"
      }
    """
    if not question_embedding:
        return {
            "chunks": [],
            "stats": {"engine": "milvus", "returned": 0, "top_k": top_k, "min_score": min_score},
            "context_text": "",
        }

    client = _client_or_init()

    # Build search params — adjust metric per how you inserted (e.g., "COSINE", "IP", "L2")
    # If you stored normalized vectors for cosine, metric_type="IP" is common. Change if needed.
    metric_type = os.getenv("MILVUS_METRIC", "IP")
    ef = int(os.getenv("MILVUS_EF", "64"))
    nprobe = int(os.getenv("MILVUS_NPROBE", "16"))

    search_params = {
        "metric_type": metric_type,   # "IP" | "COSINE" (Milvus uses "IP" and "L2"; some wrappers support "COSINE")
        "params": {"ef": ef, "nprobe": nprobe},
        "anns_field": F_VECTOR,
        "output_fields": [F_DOC_ID, F_CHUNK_ID, F_CHAT_ID],
        "limit": max(top_k, 1),
        "expr": _build_expr(chat_id),
        # If you actually use Milvus partitions per chat_id, also pass:
        # "partition_names": [chat_id] if MILVUS_PARTITION_BY_CHAT and chat_id else None,
    }

    # Execute search
    results = client.search(
        collection_name=MILVUS_COLLECTION,
        data=[question_embedding],          # batch of 1
        search_params=search_params,
    )

    # Flatten results[0] -> list of dicts
    hits: List[Dict[str, Any]] = []
    if results and len(results) > 0:
        for hit in results[0]:
            # hit.fields contains non-vector fields; hit.distance is similarity/distance
            row = {
                F_DOC_ID: hit.get(F_DOC_ID) if isinstance(hit, dict) else getattr(hit, F_DOC_ID, None),
                F_CHUNK_ID: hit.get(F_CHUNK_ID) if isinstance(hit, dict) else getattr(hit, F_CHUNK_ID, None),
                F_SCORE: hit.get("distance") if isinstance(hit, dict) else getattr(hit, "distance", None),
            }

            # If SDK returns as object not dict
            if row[F_DOC_ID] is None and hasattr(hit, "fields"):
                row[F_DOC_ID] = hit.fields.get(F_DOC_ID)
                row[F_CHUNK_ID] = hit.fields.get(F_CHUNK_ID)
                if row[F_SCORE] is None and hasattr(hit, "distance"):
                    row[F_SCORE] = hit.distance

            # Score filter
            try:
                score_val = float(row[F_SCORE]) if row[F_SCORE] is not None else 0.0
            except Exception:
                score_val = 0.0

            if score_val >= float(min_score):
                hits.append(row)

    # Build chunk pairs and fetch plaintext from SQLite/Postgres
    pairs = _pairs_from_results(hits)
    texts: List[Dict[str, Any]] = []
    if pairs:
        try:
            texts = fetch_chunk_texts_by_ids(conn, pairs)
        except Exception:
            texts = []

    # Prepare context text (compact, ordered by (doc_id, chunk_id))
    text_map = {(int(t.get("doc_id")), int(t.get("chunk_id"))): t for t in texts if t.get("text")}
    stitched_parts: List[str] = []
    for d, c in pairs:
        trow = text_map.get((d, c))
        if trow and trow.get("text"):
            stitched_parts.append(trow["text"].strip())

    context_text = "\n\n".join(stitched_parts)

    # Normalize output chunk list with scores
    chunks_out = []
    for row in hits:
        try:
            chunks_out.append({
                "doc_id": int(row[F_DOC_ID]),
                "chunk_id": int(row[F_CHUNK_ID]),
                "score": float(row[F_SCORE]) if row[F_SCORE] is not None else 0.0,
            })
        except Exception:
            continue

    return {
        "chunks": chunks_out,
        "stats": {
            "engine": "milvus",
            "returned": len(chunks_out),
            "top_k": top_k,
            "min_score": float(min_score),
            "metric": metric_type,
        },
        "context_text": context_text,
    }
