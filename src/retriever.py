"""RAG 파이프라인: 질의 시점에 수집한 근거 문서 → 청킹 → 하이브리드 검색(BM25 + 임베딩) → RRF 융합 → 다양성 리랭킹.

정형 지표(kind="metric")는 짧고 모두 중요하므로 검색 없이 항상 근거에 넣고,
비정형 텍스트(kind="text": 논문 초록·HN 글/댓글)만 질문과 관련된 청크를 골라 넣습니다.
"""
from __future__ import annotations

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .config import get_embeddings

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RRF_K = 60
MAX_CHUNKS_PER_DOC = 2


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[가-힣]+", text.lower())


def chunk_docs(docs: list[dict]) -> list[dict]:
    chunks = []
    for d in docs:
        text = d["text"]
        head = text.split("\n", 1)[0]  # 제목 줄은 모든 청크에 붙여 문맥 유지
        start = 0
        i = 0
        while start < len(text):
            piece = text[start : start + CHUNK_SIZE]
            if i > 0:
                piece = head + "\n…" + piece
            chunks.append({**d, "chunk_id": f"{d['doc_id']}#{i}", "text": piece})
            i += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _rrf(rankings: list[list[int]], n: int) -> np.ndarray:
    scores = np.zeros(n)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (RRF_K + rank + 1)
    return scores


def hybrid_search(queries: list[str], docs: list[dict], top_k: int = 8) -> tuple[list[dict], dict]:
    """queries: 원 질문 + 확장 쿼리(플래너가 뽑은 대상 기술명 등). 반환: (청크 리스트, 디버그 정보)."""
    chunks = chunk_docs(docs)
    if not chunks:
        return [], {"chunks": 0}
    n = len(chunks)
    bm25 = BM25Okapi([_tokens(c["text"]) for c in chunks])
    rankings, info = [], {"chunks": n, "dense": True}
    for q in queries:
        rankings.append(list(np.argsort(-bm25.get_scores(_tokens(q)))))

    logging.getLogger("langchain_aws").setLevel(logging.CRITICAL)
    try:  # 임베딩 검색 (실패 시 BM25 만으로 진행)
        emb = get_embeddings()
        mat = np.array(emb.embed_documents([c["text"] for c in chunks]))
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        for q in queries:
            qv = np.array(emb.embed_query(q))
            qv /= np.linalg.norm(qv) + 1e-9
            rankings.append(list(np.argsort(-(mat @ qv))))
    except Exception as e:  # noqa: BLE001
        info["dense"] = False
        info["dense_error"] = type(e).__name__

    scores = _rrf(rankings, n)
    picked, per_doc = [], {}
    for idx in np.argsort(-scores):  # 다양성 리랭킹: 문서당 최대 N 청크
        c = chunks[idx]
        if per_doc.get(c["doc_id"], 0) >= MAX_CHUNKS_PER_DOC:
            continue
        per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
        picked.append({**c, "score": round(float(scores[idx]), 4)})
        if len(picked) >= top_k:
            break
    return picked, info
