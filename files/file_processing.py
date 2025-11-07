"""
Chunker for uploads: returns normalized chunks with indices & positions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _split_into_sentences(text: str) -> List[str]:
    import re
    # very light sentence split; good enough for most docs
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def smart_chunk(
    content: str,
    target_tokens: int = 350,
    overlap_tokens: int = 60
) -> List[Dict[str, Any]]:
    """
    Greedy sentence packing with overlap. Token ≈ word here for simplicity.
    Returns: [{text, chunk_index, start_char, end_char}]
    """
    words = content.split()
    if len(words) <= target_tokens:
        return [{"text": content.strip(), "chunk_index": 0,
                 "start_char": 0, "end_char": len(content)}]

    sents = _split_into_sentences(content)
    chunks: List[Dict[str, Any]] = []
    cur: List[str] = []
    start_char = 0
    chunk_index = 0

    def emit(text: str, idx: int, start_c: int):
        end_c = start_c + len(text)
        chunks.append({
            "text": text,
            "chunk_index": idx,
            "start_char": start_c,
            "end_char": end_c
        })
        return end_c

    buf_words = 0
    acc_text = ""
    for s in sents:
        sw = len(s.split())
        if buf_words + sw <= target_tokens:
            acc_text = (acc_text + " " + s) if acc_text else s
            buf_words += sw
        else:
            # emit current
            end_last = emit(acc_text, chunk_index, start_char)
            # compute overlap seed
            overlap = " ".join(acc_text.split()[-overlap_tokens:]) if overlap_tokens > 0 else ""
            start_char = end_last - len(overlap) if overlap else end_last
            acc_text = (overlap + " " + s).strip() if overlap else s
            buf_words = len(acc_text.split())
            chunk_index += 1

    if acc_text:
        emit(acc_text, chunk_index, start_char)

    return chunks

def prepare_single_document_payload(
    *,
    title: str,
    content: str,
    summary: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    chunks = smart_chunk(content)
    return {
        "title": title.strip() or "Untitled",
        "content": content,
        "summary": (summary or "").strip() or None,
        "metadata": metadata or {},
        "chunks": chunks
    }
