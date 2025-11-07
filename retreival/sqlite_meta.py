# app/retrieval/sqlite_meta.py
from typing import Any, Dict, List, Tuple

def fetch_chunk_texts(
    conn,
    chat_id: str,
    pairs: List[Tuple[str, int]],
) -> List[Dict[str, Any]]:
    """
    Given (doc_id, chunk_id) pairs, returns hydrated chunk rows:
      {doc_id, chunk_id, text, page, source:{title, url, doc_name}}
    Expected SQLite tables:
      documents(id TEXT PRIMARY KEY, chat_id TEXT, title TEXT, url TEXT, name TEXT, ...)
      doc_chunks(chat_id TEXT, doc_id TEXT, chunk_id INTEGER, text TEXT, page INTEGER, ...)
    """
    if not pairs:
        return []

    qmarks = ",".join(["(?,?)"] * len(pairs))
    params = []
    for doc_id, chunk_id in pairs:
        params.extend([doc_id, chunk_id])

    sql = f"""
    SELECT c.doc_id, c.chunk_id, c.text, c.page,
           d.title AS doc_title, d.url AS doc_url, d.name AS doc_name
    FROM doc_chunks c
    LEFT JOIN documents d ON d.id = c.doc_id
    WHERE c.chat_id = ?
      AND (c.doc_id, c.chunk_id) IN ({qmarks})
    ORDER BY c.doc_id, c.chunk_id
    """
    cur = conn.cursor()
    cur.execute(sql, [chat_id] + params)
    rows = cur.fetchall()
    cur.close()

    out = []
    for r in rows:
        # works with row factory dict or tuple
        get = (lambda k, default=None: r[k] if hasattr(r, "keys") and k in r.keys() else default)
        doc_id = get("doc_id") or r[0]
        chunk_id = get("chunk_id") or r[1]
        text = get("text") or r[2] or ""
        page = get("page") or r[3]
        doc_title = get("doc_title") or r[4]
        doc_url = get("doc_url") or r[5]
        doc_name = get("doc_name") or r[6]

        out.append({
            "doc_id": str(doc_id),
            "chunk_id": int(chunk_id),
            "text": text,
            "page": page,
            "source": {"title": doc_title, "url": doc_url, "doc_name": doc_name},
        })
    return out
