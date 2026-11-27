"""가드레일: 입력 비밀정보 마스킹 · 프롬프트 인젝션 탐지 · 외부 데이터 정화 · 출력 검증."""
from __future__ import annotations

import re

SECRET_PATTERNS = [
    ("GitHub 토큰", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{8,}\b|\bgithub_pat_[A-Za-z0-9_]{10,}\b")),
    ("API 키", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")),
    ("AWS 키", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack 토큰", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("이메일", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("전화번호", re.compile(r"\b01[016789]-?\d{3,4}-?\d{4}\b")),
    ("주민등록번호", re.compile(r"\b\d{6}-[1-4]\d{6}\b")),
]

# 사용자 입력·외부 데이터 양쪽에서 쓰는 명령형 지시 패턴
_COMMON_INJECTION = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"(이전|위의?|앞의?)\s*(모든\s*)?(지시|지침|명령|규칙)(사항)?(을|를)?\s*(다\s*|모두\s*|전부\s*)?무시",
]

INJECTION_PATTERNS = _COMMON_INJECTION + [
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+",
    r"you\s+are\s+now\s+",
    r"(reveal|print|show)\s+(me\s+)?(your|the)\s+system\s+prompt",
    r"system\s*prompt",
    r"jailbreak",
    r"시스템\s*프롬프트",
    r"너는\s*이제부터",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# [2차] 외부 데이터(논문 초록·HN 글·README)용: 명령형 지시만 탐지.
# 1차에서는 'system prompt' 같은 단순 단어까지 잡아, 프롬프트 연구 논문 초록마다 경고가 붙는 오탐이 있었음.
EXTERNAL_INJECTION_PATTERNS = _COMMON_INJECTION + [
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions|prompts?|rules)",
    r"(reveal|print|show|output)\s+(me\s+)?(your|the)\s+system\s+prompt",
    r"(you\s+must|you\s+should|assistant\s*:)\s+(now\s+)?(say|answer|respond|tell)",
    r"시스템\s*프롬프트(를|을)?\s*(출력|공개|보여|알려)",
]
_INJ_EXT = re.compile("|".join(EXTERNAL_INJECTION_PATTERNS), re.IGNORECASE)


def find_injection(text: str) -> list[str]:
    return [m.group(0) for m in _INJ.finditer(text or "")]


def strip_injection(text: str) -> str:
    """사용자가 인용한 텍스트 속 지시문 제거 (입력용 패턴)."""
    return _INJ.sub("[⚠ 지시문 제거됨]", text)


def scrub_external(text: str) -> tuple[str, list[str]]:
    """외부 데이터에서 명령형 지시문을 찾아 제거. 반환: (정화된 텍스트, 탐지된 문구들)."""
    hits = [m.group(0) for m in _INJ_EXT.finditer(text or "")]
    if hits:
        text = _INJ_EXT.sub("[⚠ 외부 지시문 제거됨]", text)
    return text, hits


def mask_secrets(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pat in SECRET_PATTERNS:
        text, n = pat.subn(lambda m: m.group(0)[:4] + "****[마스킹]", text)
        if n:
            found.append(label)
    return text, found


def sanitize_external(docs: list[dict]) -> tuple[list[dict], list[str]]:
    """외부 문서 안의 지시문은 데이터일 뿐 → 문구를 제거하고 플래그를 남깁니다 (간접 인젝션 방어)."""
    flags, clean = [], []
    for d in docs:
        text, hits = scrub_external(d["text"])
        if hits:
            flags.append(f"{d['doc_id']}: {hits[0][:60]}")
            d = {**d, "text": text}
        clean.append(d)
    return clean, flags


URL_RE = re.compile(r"https?://[^\s)\]>|]+")


def verify_output(answer: str, evidence_urls: set[str]) -> tuple[str, list[str]]:
    """출력 검증: (1) 비밀정보 재마스킹 (2) 근거에 없는 URL 제거 (날조 링크 차단)."""
    answer, _ = mask_secrets(answer)
    issues = []
    norm = {u.rstrip("/.") for u in evidence_urls}
    # 마크다운 링크 [제목](url) 은 줄 단위로 처리: 근거에 없으면 링크를 떼고 표시
    def _md(m):
        title, url = m.group(1), m.group(2)
        if url.rstrip("/.,") in norm:
            return m.group(0)
        issues.append(f"근거에 없는 링크 제거: {url}")
        return f"{title} (⚠ 근거에서 확인되지 않은 링크라 제거함)"
    answer = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", _md, answer)
    for url in set(URL_RE.findall(answer)):
        if url.rstrip("/.,") not in norm:
            issues.append(f"근거에 없는 링크 제거: {url}")
            answer = answer.replace(url, "(출처 확인 불가 링크 제거됨)")
    return answer, issues
