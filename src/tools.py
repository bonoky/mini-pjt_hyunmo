"""도메인 도구: 키 없이 쓰는 공개 API 4종.

각 도구는 (LLM 에게 보여줄 요약 텍스트, 근거 문서 리스트) 를 함께 반환합니다
(response_format="content_and_artifact"). 근거 문서는 ToolMessage.artifact 로
그래프에 전달되어 RAG 단계와 API 응답의 contexts 로 쓰입니다.

근거 문서 형식: {"doc_id", "source", "kind", "title", "url", "fetched_at", "text"}
  kind = "metric"  : 짧은 정형 지표 (스타·버전 등) → 항상 근거에 포함
  kind = "text"    : 긴 비정형 텍스트 (초록·댓글) → 하이브리드 검색으로 골라서 포함
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import statistics
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx
from langchain_core.tools import tool

from .config import CACHE_DIR, USE_CACHE, today

UA = {"User-Agent": "TechScout-mini-project/0.1"}
_client = httpx.Client(headers=UA, timeout=20, follow_redirects=True)


class NotFound(Exception):
    pass


def _get(url: str, params: dict | None = None, kind: str = "json"):
    """GET + 디스크 캐시. 404 는 NotFound 로 올립니다."""
    key_src = url + "?" + json.dumps(params or {}, sort_keys=True)
    path = CACHE_DIR / (hashlib.sha1(key_src.encode()).hexdigest()[:16] + ".json")
    if USE_CACHE and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["body"], cached["fetched_at"]

    headers = {}
    if "api.github.com" in url and os.getenv("GITHUB_TOKEN"):
        # 토큰은 서버 환경변수에서만 읽음 (사용자 입력 토큰은 절대 사용하지 않음)
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    for attempt in range(3):  # 재시도 미들웨어 (일시 오류·rate limit)
        resp = _client.get(url, params=params, headers=headers)
        if resp.status_code == 404:
            raise NotFound(url)
        if resp.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(2 * (attempt + 1))
            continue
        resp.raise_for_status()
        break
    body = resp.json() if kind == "json" else resp.text
    fetched_at = today()
    if USE_CACHE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"url": url, "params": params, "fetched_at": fetched_at, "body": body}, ensure_ascii=False),
            encoding="utf-8",
        )
    return body, fetched_at


def _strip_html(s: str | None) -> str:
    s = re.sub(r"<p>", "\n", s or "")
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def _date(s: str | None) -> str:
    return (s or "")[:10]


# ---------------------------------------------------------------- arXiv
ATOM = {"a": "http://www.w3.org/2005/Atom"}


@tool(response_format="content_and_artifact")
def search_papers(query: str, max_results: int = 5, sort: str = "recent"):
    """arXiv 에서 기술 주제의 논문을 검색합니다. 연구 동향·최신 기법을 물을 때 사용하세요.

    Args:
        query: 영어 검색어 (예: "agentic RAG", "speculative decoding"). 한국어 금지.
        max_results: 가져올 논문 수 (1~10)
        sort: "recent" (최신순) 또는 "relevance" (관련도순)
    """
    max_results = max(1, min(int(max_results), 10))
    terms = " AND ".join(f'all:"{t}"' if " " in t else f"all:{t}" for t in [query.strip()])
    params = {
        "search_query": terms,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate" if sort == "recent" else "relevance",
        "sortOrder": "descending",
    }
    try:
        body, fetched_at = _get("https://export.arxiv.org/api/query", params, kind="text")
    except Exception as e:  # noqa: BLE001
        return f"[search_papers] 조회 실패: {e}", []
    root = ET.fromstring(body)
    docs, lines = [], []
    for entry in root.findall("a:entry", ATOM):
        url = entry.findtext("a:id", default="", namespaces=ATOM).strip()
        title = " ".join(entry.findtext("a:title", default="", namespaces=ATOM).split())
        summary = " ".join(entry.findtext("a:summary", default="", namespaces=ATOM).split())
        published = _date(entry.findtext("a:published", default="", namespaces=ATOM))
        doc = {
            "doc_id": "arxiv:" + url.rsplit("/", 1)[-1],
            "source": "arxiv",
            "kind": "text",
            "title": title,
            "url": url,
            "fetched_at": fetched_at,
            "text": f"[논문] {title} (제출 {published})\n{summary}",
        }
        docs.append(doc)
        lines.append(f"- {title} | 제출 {published} | {url}")
    if not docs:
        return f"[search_papers] '{query}' 검색 결과 없음 (조회일 {fetched_at})", []
    return f"[search_papers] '{query}' 논문 {len(docs)}건 (조회일 {fetched_at})\n" + "\n".join(lines), docs


# ---------------------------------------------------------------- GitHub
def _resolve_repo(repo: str) -> str:
    repo = repo.strip().removeprefix("https://github.com/").strip("/")
    if "/" in repo:
        return repo
    body, _ = _get("https://api.github.com/search/repositories", {"q": f"{repo} in:name", "sort": "stars", "per_page": 1})
    items = body.get("items") or []
    if not items:
        raise NotFound(repo)
    return items[0]["full_name"]


@tool(response_format="content_and_artifact")
def get_repo_stats(repo: str):
    """GitHub 저장소의 유지보수·채택도 지표를 조회합니다 (스타, 포크, 오픈 이슈, 마지막 push, 최신 릴리스).

    Args:
        repo: "owner/name" 형식 권장 (예: "langchain-ai/langgraph"). 이름만 주면 스타 순으로 가장 유명한 저장소를 찾습니다.
    """
    try:
        full = _resolve_repo(repo)
        r, fetched_at = _get(f"https://api.github.com/repos/{full}")
    except NotFound:
        return f"[get_repo_stats] '{repo}' 저장소를 찾을 수 없음 (조회일 {today()})", []
    except Exception as e:  # noqa: BLE001
        return f"[get_repo_stats] 조회 실패: {e}", []
    try:
        rel, _ = _get(f"https://api.github.com/repos/{full}/releases/latest")
        release = f"{rel.get('tag_name')} ({_date(rel.get('published_at'))})"
    except NotFound:
        release = "GitHub 릴리스 없음"
    except Exception:  # noqa: BLE001
        release = "조회 실패"
    url = r.get("html_url", f"https://github.com/{full}")
    text = (
        f"[GitHub] {full}\n"
        f"- 설명: {r.get('description') or '-'}\n"
        f"- 스타: {r.get('stargazers_count'):,} · 포크: {r.get('forks_count'):,} · 오픈 이슈+PR: {r.get('open_issues_count'):,}\n"
        f"- 생성일: {_date(r.get('created_at'))} · 마지막 push: {_date(r.get('pushed_at'))}\n"
        f"- 최신 릴리스: {release}\n"
        f"- 라이선스: {(r.get('license') or {}).get('spdx_id') or '-'} · 보관(archived): {r.get('archived')}\n"
        f"- 조회일: {fetched_at} · 출처: {url}"
    )
    doc = {
        "doc_id": f"github:{full}",
        "source": "github",
        "kind": "metric",
        "title": f"GitHub {full}",
        "url": url,
        "fetched_at": fetched_at,
        "text": text,
    }
    return text, [doc]


# ---------------------------------------------------------------- PyPI
@tool(response_format="content_and_artifact")
def get_package_info(package: str):
    """PyPI 패키지의 최신 버전과 최근 릴리스 이력·주기를 조회합니다. 릴리스 안정성·지원 Python 버전 확인에 사용하세요.

    Args:
        package: PyPI 패키지 이름 (예: "langgraph", "vllm")
    """
    name = package.strip().lower()
    try:
        body, fetched_at = _get(f"https://pypi.org/pypi/{name}/json")
    except NotFound:
        return f"[get_package_info] PyPI 에 '{package}' 패키지 없음 (조회일 {today()})", []
    except Exception as e:  # noqa: BLE001
        return f"[get_package_info] 조회 실패: {e}", []
    info = body["info"]
    releases = []
    for ver, files in body.get("releases", {}).items():
        times = [f["upload_time_iso_8601"] for f in files if not f.get("yanked")]
        if times:
            releases.append((min(times)[:10], ver))
    releases.sort(reverse=True)
    recent = releases[:8]
    gaps = []
    for (d1, _), (d2, _) in zip(recent, recent[1:]):
        gaps.append((datetime.fromisoformat(d1) - datetime.fromisoformat(d2)).days)
    cadence = f"최근 {len(gaps)}회 간격 중앙값 {statistics.median(gaps):.0f}일" if gaps else "판단 불가"
    url = f"https://pypi.org/project/{info['name']}/"
    text = (
        f"[PyPI] {info['name']}\n"
        f"- 요약: {info.get('summary') or '-'}\n"
        f"- 최신 버전: {info['version']} · 필요 Python: {info.get('requires_python') or '-'}\n"
        f"- 최근 릴리스: " + ", ".join(f"{v}({d})" for d, v in recent) + "\n"
        f"- 릴리스 주기: {cadence} · 전체 릴리스 수: {len(releases)}\n"
        f"- 조회일: {fetched_at} · 출처: {url}"
    )
    doc = {
        "doc_id": f"pypi:{info['name']}",
        "source": "pypi",
        "kind": "metric",
        "title": f"PyPI {info['name']}",
        "url": url,
        "fetched_at": fetched_at,
        "text": text,
    }
    return text, [doc]


# ---------------------------------------------------------------- Hacker News
@tool(response_format="content_and_artifact")
def search_community(query: str, days: int = 365):
    """Hacker News 에서 기술에 대한 현업 개발자 반응을 검색합니다 (글 제목·포인트·댓글 수·상위 댓글).

    Args:
        query: 영어 검색어 (예: "MCP", "LangGraph")
        days: 최근 며칠 이내 글만 볼지 (기본 365)
    """
    since = int((datetime.now(timezone.utc) - timedelta(days=int(days))).timestamp())
    params = {"query": query, "tags": "story", "numericFilters": f"created_at_i>{since},points>5", "hitsPerPage": 8}
    try:
        body, fetched_at = _get("https://hn.algolia.com/api/v1/search", params)
    except Exception as e:  # noqa: BLE001
        return f"[search_community] 조회 실패: {e}", []
    hits = sorted(body.get("hits", []), key=lambda h: h.get("points") or 0, reverse=True)[:6]
    docs, lines = [], []
    for i, h in enumerate(hits):
        hn_url = f"https://news.ycombinator.com/item?id={h['objectID']}"
        title = h.get("title") or ""
        head = f"[HN] {title} (작성 {_date(h.get('created_at'))}, {h.get('points')}점, 댓글 {h.get('num_comments')}개)"
        comments = []
        if i < 3:  # 상위 3개 글은 대표 댓글까지 수집
            try:
                item, _ = _get(f"https://hn.algolia.com/api/v1/items/{h['objectID']}")
                for c in (item.get("children") or [])[:5]:
                    t = _strip_html(c.get("text"))
                    if t:
                        comments.append("· " + t[:500])
            except Exception:  # noqa: BLE001
                pass
        story = _strip_html(h.get("story_text"))[:800]
        docs.append({
            "doc_id": f"hn:{h['objectID']}",
            "source": "hackernews",
            "kind": "text",
            "title": title,
            "url": hn_url,
            "fetched_at": fetched_at,
            "text": "\n".join(x for x in [head, story, *comments] if x),
        })
        lines.append(f"- {title} | {h.get('points')}점 · 댓글 {h.get('num_comments')} | {_date(h.get('created_at'))} | {hn_url}")
    if not docs:
        return f"[search_community] '{query}' 최근 {days}일 글 없음 (조회일 {fetched_at})", []
    return f"[search_community] '{query}' 글 {len(docs)}건 (조회일 {fetched_at})\n" + "\n".join(lines), docs


TOOLS = [search_papers, get_repo_stats, get_package_info, search_community]
