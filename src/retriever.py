"""RAG 파이프라인: 질의 시점에 수집한 근거 문서 → 청킹 → 하이브리드 검색(BM25 + 임베딩) → RRF 융합 → 다양성 리랭킹.

정형 지표(kind="metric")는 짧고 모두 중요하므로 검색 없이 항상 근거에 넣고,
비정형 텍스트(kind="text": 논문 초록·HN 글/댓글)만 질문과 관련된 청크를 골라 넣습니다.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rank_bm25 import BM25Okapi

from .config import get_embeddings

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RRF_K = 60
MAX_CHUNKS_PER_DOC = 2
EMBED_WORKERS = 8  # [3차] Titan 임베딩은 텍스트 1건당 API 1회 → 순차 호출(청크 30개 ≈ 10초+) 대신 병렬 호출

# [3차] 같은 텍스트의 임베딩은 프로세스 안에서 재사용 (같은 근거·같은 확장 검색어가 반복되는 경우)
_EMB_CACHE: dict[str, list[float]] = {}


def _embed_all(emb, texts: list[str]) -> list[list[float]]:
    missing = [t for t in dict.fromkeys(texts) if t not in _EMB_CACHE]
    if missing:
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as ex:
            for t, v in zip(missing, ex.map(emb.embed_query, missing)):
                _EMB_CACHE[t] = v
    return [_EMB_CACHE[t] for t in texts]


def _diverse(order, chunks: list[dict], top_k: int, scores=None) -> list[dict]:
    """다양성 리랭킹: 문서당 최대 MAX_CHUNKS_PER_DOC 청크."""
    picked, per_doc = [], {}
    for idx in order:
        c = chunks[idx]
        if per_doc.get(c["doc_id"], 0) >= MAX_CHUNKS_PER_DOC:
            continue
        per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
        picked.append({**c, "score": round(float(scores[idx]), 4)} if scores is not None else c)
        if len(picked) >= top_k:
            break
    return picked


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
    t0 = time.perf_counter()
    chunks = chunk_docs(docs)
    if not chunks:
        return [], {"chunks": 0}
    n = len(chunks)
    if n <= top_k:  # [3차] 고를 것이 없으면 검색·임베딩을 건너뜀
        picked = _diverse(range(n), chunks, top_k)
        return picked, {"chunks": n, "dense": False, "skipped": "chunks<=top_k",
                        "elapsed_s": round(time.perf_counter() - t0, 2)}
    bm25 = BM25Okapi([_tokens(c["text"]) for c in chunks])
    rankings, info = [], {"chunks": n, "dense": True}
    for q in queries:
        rankings.append(list(np.argsort(-bm25.get_scores(_tokens(q)))))

    logging.getLogger("langchain_aws").setLevel(logging.CRITICAL)
    try:  # 임베딩 검색 (실패 시 BM25 만으로 진행) · [3차] 청크·질의 임베딩을 한 번에 병렬 호출
        vecs = np.array(_embed_all(get_embeddings(), [c["text"] for c in chunks] + list(queries)))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        mat, qmat = vecs[:n], vecs[n:]
        for qv in qmat:
            rankings.append(list(np.argsort(-(mat @ qv))))
    except Exception as e:  # noqa: BLE001
        info["dense"] = False
        info["dense_error"] = type(e).__name__

    scores = _rrf(rankings, n)
    picked = _diverse(np.argsort(-scores), chunks, top_k, scores)
    info["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return picked, info
