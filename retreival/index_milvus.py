# app/rag/index_milvus.py
from __future__ import annotations
import os
from typing import Any, Dict, Iterable, List, Tuple, Optional

from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from ai.openai_integration import generate_openai_embedding  # you already use this in ask flow

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "nx_chunks")
MILVUS_PARTITION_BY_CHAT = os.getenv("MILVUS_PARTITION_BY_CHAT", "1") in ("1", "true", "True")

F_DOC_ID   = os.getenv("MILVUS_FIELD_DOC_ID", "doc_id")          # Int64
F_CHUNK_ID = os.getenv("MILVUS_FIELD_CHUNK_ID", "chunk_id")      # Int64
F_CHAT_ID  = os.getenv("MILVUS_FIELD_CHAT_ID", "chat_id")        # VarChar
F_VECTOR   = os.getenv("MILVUS_FIELD_VECTOR", "embedding")       # FloatVector

VECTOR_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))  # match your OpenAI embedding model

_client: Optional[MilvusClient] = None
def _client_or_init() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN or None)
    return _client


def ensure_collection() -> None:
    """
    Idempotent: create collection + index if not exists.
    Schema: (doc_id INT64, chunk_id INT64, chat_id VARCHAR, embedding FLOAT_VECTOR)
    """
    client = _client_or_init()
    if client.has_collection(MILVUS_COLLECTION):
        return

    fields = [
        FieldSchema(name=F_DOC_ID,   dtype=DataType.INT64, is_primary=False, auto_id=False),
        FieldSchema(name=F_CHUNK_ID, dtype=DataType.INT64, is_primary=False, auto_id=False),
        FieldSchema(name=F_CHAT_ID,  dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name=F_VECTOR,   dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="doc chunks (doc_id,chunk_id,chat_id,embedding)")
    client.create_collection(collection_name=MILVUS_COLLECTION, schema=schema)

    # Build index (tune to taste). If using cosine, many use IP with normalized vectors.
    metric_type = os.getenv("MILVUS_METRIC", "IP")
    client.create_index(
        collection_name=MILVUS_COLLECTION,
        field_name=F_VECTOR,
        index_params={"index_type": "IVF_FLAT", "metric_type": metric_type, "params": {"nlist": 1024}},
    )
    client.load_collection(MILVUS_COLLECTION)


def _partition_for_chat(chat_id: Optional[str]) -> Optional[str]:
    if MILVUS_PARTITION_BY_CHAT and chat_id:
        return chat_id
    return None


def ensure_partition(chat_id: Optional[str]) -> None:
    client = _client_or_init()
    pname = _partition_for_chat(chat_id)
    if not pname:
        return
    if not client.has_partition(collection_name=MILVUS_COLLECTION, partition_name=pname):
        client.create_partition(collection_name=MILVUS_COLLECTION, partition_name=pname)


def delete_vectors_for_doc(doc_id: int, chat_id: Optional[str]) -> int:
    """
    Delete all vectors for a document (e.g., on re-upload).
    Returns number of deleted rows (best-effort; Milvus returns int).
    """
    client = _client_or_init()
    expr = f'{F_DOC_ID} == {int(doc_id)}'
    # Optionally also filter by chat_id if you store multi-tenant in one collection
    if chat_id and MILVUS_PARTITION_BY_CHAT:
        # partition-aware delete is faster if you pass partition_names
        return client.delete(
            collection_name=MILVUS_COLLECTION,
            expr=expr,
            partition_names=[chat_id],
        )
    return client.delete(collection_name=MILVUS_COLLECTION, expr=expr)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch helper using your existing OpenAI embedding util.
    Falls back to per-item if you didn’t implement batching there.
    """
    embs: List[List[float]] = []
    for t in texts:
        e = await generate_openai_embedding(t)
        embs.append(e)
    return embs


async def upsert_chunk_embeddings(
    *,
    chat_id: Optional[str],
    doc_id: int,
    chunks: List[Tuple[int, str]],   # [(chunk_id, text)]
    replace_existing: bool = True,
) -> int:
    """
    Ensures collection/partition, optionally deletes existing vectors for this doc,
    computes embeddings for provided chunk texts, and inserts into Milvus.
    Returns number of inserted vectors.
    """
    ensure_collection()
    ensure_partition(chat_id)

    if replace_existing:
        try:
            delete_vectors_for_doc(doc_id, chat_id)
        except Exception:
            pass  # non-fatal

    if not chunks:
        return 0

    # Prepare embeddings
    chunk_ids = [cid for cid, _ in chunks]
    texts     = [txt for _,  txt in chunks]
    vectors   = await embed_texts(texts)

    # Build rows for Milvus
    rows = []
    for cid, vec in zip(chunk_ids, vectors):
        rows.append({
            F_DOC_ID:   int(doc_id),
            F_CHUNK_ID: int(cid),
            F_CHAT_ID:  chat_id or "",
            F_VECTOR:   vec,
        })

    client = _client_or_init()
    pname = _partition_for_chat(chat_id)
    client.insert(collection_name=MILVUS_COLLECTION, data=rows, partition_name=pname)
    return len(rows)
