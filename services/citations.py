# app/rag/citations.py
from typing import List, Dict, Any, Tuple

class CitationIndex:
    def __init__(self):
        self._items: List[Dict[str,Any]] = []
        self._next_id = 1

    def add_web_sources(self, web_sources: List[Dict[str,Any]]):
        for s in web_sources or []:
            self._items.append({
                "id": self._next_id,
                "type": "web",
                "title": s.get("title"),
                "url": s.get("url"),
            })
            self._next_id += 1

    def add_rag_chunks(self, chunks: List[Dict[str,Any]]):
        for ch in chunks or []:
            self._items.append({
                "id": self._next_id,
                "type": "rag",
                "doc_id": ch.get("doc_id"),
                "chunk_id": ch.get("chunk_id"),
                "title": ch.get("source_title"),
                "url": ch.get("source_url"),
            })
            self._next_id += 1

    def finalize(self):
        pass

    def serialize_for_ui(self) -> List[Dict[str,Any]]:
        return self._items

    def render_citation_block(self) -> str:
        if not self._items:
            return ""
        lines = ["### Sources"]
        for it in self._items:
            tag = "[web]" if it["type"] == "web" else f"[doc {it.get('doc_id')}#{it.get('chunk_id')}]"
            title = it.get("title") or "Source"
            url = it.get("url") or ""
            if url:
                lines.append(f"- {tag} {title} — {url}")
            else:
                lines.append(f"- {tag} {title}")
        return "\n".join(lines)

def render_retrieval_context(chunks: List[Dict[str,Any]],
                             citation_id_by_chunk: Dict[Tuple[str,int], int],
                             max_chars_per_chunk: int = 900,
                             max_total_chars: int = 4000) -> str:
    out = []
    total = 0
    for ch in chunks or []:
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        text = text[:max_chars_per_chunk]
        key = (str(ch.get("doc_id")), int(ch.get("chunk_id")))
        cid = citation_id_by_chunk.get(key)
        line = f"{text}"
        if cid:
            line += f"  [{cid}]"
        if total + len(line) > max_total_chars:
            break
        out.append(line)
        total += len(line)
    if not out:
        return ""
    return "### Retrieved Knowledge (Cited)\n" + "\n\n".join(out)
