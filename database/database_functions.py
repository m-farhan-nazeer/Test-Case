"""
Database functions for SQLite operations with RAG chunk support.
- Tables: documents, chunks, chunk_embeddings
- Dedup via UNIQUE(chunk_hash)
"""

import os
import sqlite3
import json
import logging
import hashlib
from typing import Optional, Any, List, Dict, Tuple
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import HTTPException

DB_PATH = os.environ.get("SQLITE_DB_PATH", "app.db")
logger = logging.getLogger(__name__)

# ---------- Core connection ----------

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------- Schema ----------

def _exec(conn: sqlite3.Connection, sql: str, params: Tuple = ()):
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    return cur

def ensure_tables(conn: sqlite3.Connection) -> None:
    _exec(conn, """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT,
        summary TEXT,
        metadata TEXT,
        source_path TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    _exec(conn, """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        chunk_hash TEXT NOT NULL UNIQUE,
        start_char INTEGER,
        end_char INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    );
    """)

    _exec(conn, """
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
        chunk_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL,
        dim INTEGER NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
    );
    """)

# ---------- Helpers ----------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _normalize(text: str) -> str:
    return " ".join(text.split()).strip()

def chunk_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()

def _to_blob(vec: List[float]) -> bytes:
    # store as JSON in blob for simplicity/compat
    return json.dumps(vec).encode("utf-8")

def _from_blob(blob: bytes) -> List[float]:
    return json.loads(blob.decode("utf-8"))

# ---------- Documents ----------

def upsert_document(
    conn: sqlite3.Connection,
    *,
    title: str,
    content: Optional[str],
    summary: Optional[str],
    metadata: Optional[Dict[str, Any]],
    source_path: Optional[str] = None,
    document_id: Optional[str] = None
) -> str:
    ensure_tables(conn)
    doc_id = document_id or str(uuid4())
    created_at = updated_at = now_iso()
    _exec(conn, """
        INSERT OR REPLACE INTO documents(id, title, content, summary, metadata, source_path, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM documents WHERE id=?), ?), ?)
    """, (
        doc_id, title, content, summary,
        json.dumps(metadata or {}), source_path,
        doc_id, created_at, updated_at
    ))
    return doc_id

def get_document(conn: sqlite3.Connection, document_id: str) -> Optional[Dict[str, Any]]:
    ensure_tables(conn)
    row = _exec(conn, "SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("metadata"):
        d["metadata"] = json.loads(d["metadata"])
    return d

def list_documents(conn: sqlite3.Connection, limit: int = 200) -> List[Dict[str, Any]]:
    ensure_tables(conn)
    rows = _exec(conn, "SELECT * FROM documents ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        out.append(d)
    return out

def delete_document(conn: sqlite3.Connection, document_id: str) -> None:
    ensure_tables(conn)
    _exec(conn, "DELETE FROM documents WHERE id=?", (document_id,))

# ---------- Chunks ----------

def insert_chunk(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    chunk_index: int,
    text: str,
    start_char: Optional[int],
    end_char: Optional[int]
) -> Dict[str, Any]:
    ensure_tables(conn)
    h = chunk_hash(text)
    exists = _exec(conn, "SELECT id FROM chunks WHERE chunk_hash=?", (h,)).fetchone()
    if exists:
        # de-duped — return existing
        return {"id": exists["id"], "document_id": document_id, "chunk_index": chunk_index, "text": text,
                "start_char": start_char, "end_char": end_char, "deduped": True}

    chunk_id = str(uuid4())
    _exec(conn, """
        INSERT INTO chunks(id, document_id, chunk_index, text, chunk_hash, start_char, end_char, created_at)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
    """, (chunk_id, document_id, chunk_index, text, h, start_char, end_char, now_iso()))
    return {"id": chunk_id, "document_id": document_id, "chunk_index": chunk_index, "text": text,
            "start_char": start_char, "end_char": end_char, "deduped": False}

def list_chunks_for_document(conn: sqlite3.Connection, document_id: str) -> List[Dict[str, Any]]:
    ensure_tables(conn)
    rows = _exec(conn, "SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_index ASC", (document_id,)).fetchall()
    return [dict(r) for r in rows]

def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> Optional[Dict[str, Any]]:
    ensure_tables(conn)
    r = _exec(conn, "SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
    return dict(r) if r else None

# ---------- Embeddings ----------

def upsert_chunk_embedding(conn: sqlite3.Connection, *, chunk_id: str, embedding: List[float]) -> None:
    ensure_tables(conn)
    dim = len(embedding)
    _exec(conn, """
        INSERT OR REPLACE INTO chunk_embeddings(chunk_id, embedding, dim)
        VALUES (?, ?, ?)
    """, (chunk_id, _to_blob(embedding), dim))

def fetch_embeddings_for_document(conn: sqlite3.Connection, document_id: str) -> List[Dict[str, Any]]:
    ensure_tables(conn)
    rows = _exec(conn, """
        SELECT c.id as chunk_id, c.text, e.embedding, e.dim
        FROM chunks c
        JOIN chunk_embeddings e ON e.chunk_id = c.id
        WHERE c.document_id = ?
        ORDER BY c.chunk_index ASC
    """, (document_id,)).fetchall()
    out = []
    for r in rows:
        out.append({
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "embedding": _from_blob(r["embedding"]),
            "dim": r["dim"]
        })
    return out

def fetch_all_embeddings(conn: sqlite3.Connection, limit: int = 5000) -> List[Dict[str, Any]]:
    ensure_tables(conn)
    rows = _exec(conn, """
        SELECT c.id as chunk_id, c.text, c.document_id, e.embedding, e.dim
        FROM chunks c
        JOIN chunk_embeddings e ON e.chunk_id = c.id
        ORDER BY c.created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [{
        "chunk_id": r["chunk_id"],
        "document_id": r["document_id"],
        "text": r["text"],
        "embedding": _from_blob(r["embedding"]),
        "dim": r["dim"]
    } for r in rows]
