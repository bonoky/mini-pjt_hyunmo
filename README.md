# 미니 PJT: TechScout — 실무형 최신 기술 리서치 어시스턴트

## 무엇을 푸나
"이 기술 지금 써도 되나?"를 논문·GitHub·PyPI·Hacker News 공개 데이터 근거로 판단해 주는 리서치 에이전트.
따로 준비한 데이터 없이, 질의 시점에 키 없는 공개 API 에서 근거를 모아 **출처·조회일이 붙은 리포트**를 만듭니다. 단, 조회 결과는 만료 없는 디스크 캐시에 저장되어 재사용되므로(기본 활성), 표시되는 조회일은 최초 수집일입니다. LLM·임베딩은 Amazon Bedrock 을 쓰므로 AWS 자격 증명이 필요합니다.

## 활용한 패턴 (Day 1~7)
- **Day 1 LCEL + Pydantic 구조화 출력**: 입력 분류(`GuardDecision`)·계획(`Plan`)·리포트(`Report`)를 모두 구조화 출력으로 받고, 작성기는 `prompt | llm.with_structured_output(Report)` 체인
- **Day 2 RAG**: 수집한 초록·커뮤니티 글을 청킹 → BM25 + Titan 임베딩 하이브리드 검색 → RRF 융합 → 문서당 2청크 다양성 리랭킹. 플래너의 영어 검색어로 쿼리 확장
- **Day 3 ReAct**: `bind_tools` 로 4개 도구를 자율 선택·병렬 호출 (최대 2라운드, `MAX_TOOL_ROUNDS=2`)
- **Day 4 도구 다중**: arXiv · GitHub · PyPI · Hacker News 를 한 질의에서 결합
- **Day 5 가드레일·미들웨어**: 비밀정보 마스킹, 직접/간접 프롬프트 인젝션 방어, 근거에 없는 링크 제거, API 재시도
- **Day 7 Plan-Execute**: 플래너가 대상·의도·단계를 먼저 정하고 리서처가 실행
- **Day 7 Observability**: LangSmith 자동 트레이스 + API 응답의 `trace` 필드에 단계별 입출력 기록
- **Day 7 평가**: LLM-as-Judge 인-아웃 세트 통과율 + RAGAS 4개 지표 (`evaluation/run_eval.py`)

## 아키텍처
```
POST /query {"question"}
   │
guard_input ──(비밀정보 / 직접 인젝션 / 범위 밖 / 모호)──► respond_blocked ──► 응답
   │ research
planner (Plan: intent · targets · search_terms · steps)
   │
researcher ⇄ tools        search_papers     → arXiv
   │  (ReAct, ≤2라운드)     get_repo_stats    → GitHub REST
   │                       get_package_info  → PyPI JSON
   │                       search_community  → Hacker News Algolia
   │                       └ 외부 데이터 속 지시문 제거 (간접 인젝션 방어)
retrieve_docs   정형 지표(GitHub·PyPI)는 전부 + 비정형 텍스트는 하이브리드 검색 top-5
   │
writer          LCEL → Report(Pydantic) → 마크다운 → guard_output(근거 밖 링크 제거·재마스킹)
   │
{"answer", "contexts", "trace"}
```
- 정형 지표는 짧고 모두 중요해서 검색 없이 넣고, 긴 텍스트만 RAG 로 골라 **환각 여지와 토큰을 함께 줄였습니다**.
- 조회 결과는 `data/cache/` 에 만료 없이 저장해 재사용합니다 (`TECHSCOUT_CACHE=0` 이면 항상 새로 조회). 1차·2차 평가가 같은 근거였는지는 두 라운드의 캐시 키를 대조해야 확인할 수 있으며, 아직 확인하지 않았습니다.

## 실행 방법
```bash
# 1) 가상환경 · 패키지
python -m venv .venv && source .venv/Scripts/activate   # macOS/Linux: .venv/bin/activate
pip install -r requirements.txt

# 2) 키 설정: .env.example 을 .env 로 복사 후 AWS·LangSmith 값 입력
cp .env.example .env

# 3) 실행
./run.sh ask "LangGraph 지금 실무에 도입해도 괜찮아?"   # CLI 한 번 질의
./run.sh server                                         # API 서버 (http://localhost:8000/docs)
./run.sh eval 1                                         # 1차 자체 평가 → evaluation/round1_report.md
```
Docker (클린 환경 재현):
```bash
docker build -t techscout .
docker run --rm -p 8000:8000 --env-file .env techscout   # 키는 실행 시에만 전달
```
API 예시:
```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{"question": "Qdrant vs Chroma 운영 환경 비교"}'
```

## RAGAS 평가 결과 (Round 2 기준 · 괄호는 Round 1)
- context_recall: 0.517 (0.479)
- context_precision: 0.364 (0.322)
- faithfulness: 0.579 (0.579)
- answer_relevancy: 0.512 (0.543)

> 리서치 서비스라 고정 정답이 없어 RAGAS `reference` 는 `expected_traits` 로 대체했습니다.

## 인-아웃 세트 통과율 (자체 평가)
- 1차 (Day 9 종료): 13 / 15 통과 (87%) · 평균 42.0초
- 2차 (Day 10 개선 후): 15 / 15 통과 (100%) · 평균 34.2초
- 개선폭: +2 건 (주요 개선 사항: 구조화 출력 검증·재시도로 #3 에러 해결, 입력 분류 기준 보완으로 #11 과잉 되묻기 해결, 외부 데이터 인젝션 오탐 제거, 인용 번호 불일치 7건→0건, 응답 시간 -19%)
- 상세: `evaluation/round1_report.md`, `evaluation/round2_report.md`

## 트라이앤에러 회고
### 시도했지만 실패한 접근 · 왜 실패했는가
| 시도 (1차) | 결과 | 실패 원인 |
|---|---|---|
| 작성기가 Pydantic `Report` 를 한 번에 생성, 실패 처리 없음 | #3 답변이 에러 메시지만 출력 | LLM 이 리스트 필드(`caveats`)를 문자열로 반환 → 검증 실패, 재시도·폴백이 없었음 |
| 입력 분류기가 대상이 불명확하면 `ambiguous` 로 되묻기 | #11 "프레임워크 5개 비교" 에 비교 대신 되묻기 | 분류 기준이 넓어 의도가 분명한 요청까지 모호로 판정 |
| 사용자 입력과 외부 데이터에 같은 인젝션 패턴 적용 | 프롬프트 연구 논문 초록마다 ⚠ 경고 (#6 오탐) | `system prompt` 같은 단순 단어까지 지시문으로 탐지 |
| 작성기(LLM)가 출처 목록을 직접 작성 | 본문 `[n]` 과 출처 번호 불일치 7건 | 본문은 근거 번호, 출처 목록은 1부터 새로 번호를 매김 · URL 날조 여지도 있음 |
| 리서처 ReAct 최대 4라운드 | 평균 42.0초 · 최대 117.2초 (목표 30초) | 같은 도구를 비슷한 검색어로 반복 호출, 도입·비교 질문에도 논문 검색 |
| 텍스트 청크 top-8 | context_precision 0.322 | 질문과 무관한 HN 글이 근거에 섞이고, 답변이 근거 밖 일반론을 덧붙임 |
| 평가 도구 첫 실행 | 11/15 · RAGAS 전부 NaN | 에이전트가 아니라 평가 도구 오류: Judge 가 오늘 날짜를 몰라 조회일을 '미래 날짜=날조'로 오판, ragas 가 Bedrock 종료 신호(`stopReason`)를 인식 못함 → 수정 후 같은 답변 재채점(`--rejudge`)으로 13/15 |

### 최종 채택한 접근 · 왜 그것으로 갔는가 (코드의 `[2차]` 주석)
- **구조화 출력 방어** — 문자열을 줄 단위 리스트로 바꾸는 validator(`_as_list`) + 1회 재시도 + 최종 실패 시 근거 목록 폴백(`_fallback`). 에러 화면 대신 최소한 근거라도 돌려주기 위해. → #3 해결
- **분류 기준 축소** — `ambiguous` 는 의도 자체를 알 수 없을 때만, 대상이 없으면 플래너가 대표 후보를 선정. 되묻기보다 조사 후 가정을 밝히는 편이 리서치 서비스에 맞아서. → #11 해결
- **외부 데이터 전용 인젝션 패턴** — 명령형 지시만 탐지(`EXTERNAL_INJECTION_PATTERNS`). 실제 인젝션(#14)은 잡고 오탐(#6)만 제거.
- **출처 목록을 시스템이 생성** — 본문의 `[n]` 을 파싱해 근거에서 출처를 만듦(`_sources`). 번호 불일치와 링크 날조를 구조적으로 차단. → 불일치 7건 → 0건
- **라운드 상한 4→2 + 첫 라운드 병렬 호출** — 계획 단계에서 필요한 도구를 확정해 한 번에 부르고, 부족할 때만 1회 보충. → 평균 42.0초 → 34.2초
- **top-k 8→5 + 근거 밖 일반론 금지 규칙** — 근거 정밀도를 우선. → context_precision 0.322 → 0.364

### 남은 한계 · 향후 개선 방향
- **응답 시간 목표(30초) 미달**: 평균 34.2초, 다중 대상 비교(#11 97.7초)·논문 동향(#3 88.1초)이 길다. 작성기의 긴 구조화 출력 비중이 큼 → 스트리밍 응답·작성기 모델 경량화. 캐시 적중이 섞인 수치라 콜드 상태는 더 느릴 수 있음
- **RAGAS 목표 미달**: faithfulness 0.579 정체, answer_relevancy 0.543 → 0.512 하락, context_precision 0.364 → 관련도 리랭커(cross-encoder) 도입, 케이스별 RAGAS 분석 후 작성기 프롬프트 개선
- **평가 신뢰도**: Judge·RAGAS 가 에이전트와 같은 모델, RAGAS reference 를 `expected_traits` 로 대체, 2차 15/15 는 1차 실패를 보고 고친 같은 15건(held-out 없음) → 정답 요지 확보, 더 어려운 edge·guardrail 케이스와 held-out 세트 추가
- **데이터 신선도**: 디스크 캐시에 만료가 없어 조회일이 최초 수집일로 고정 → 캐시 TTL 도입
- GitHub API 비인증 한도(시간당 60회): 평가를 반복하면 캐시 또는 `GITHUB_TOKEN` 필요
- HN 검색은 영어 키워드 의존 → 한국어 커뮤니티 반응은 다루지 못함
- 자세한 목표 대비 갭과 로드맵은 `SERVICE.md` 5·6절 참고

## 핵심 코드 위치
- `src/agent.py:341` — 메인 에이전트 그래프 (`build_graph`) · 노드: `guard_input:116` · `planner:176` · `researcher:202` · `retrieve:242` · `writer:311` · 출처 생성 `_sources:284` · 진입점 `run_agent:364`
- `src/tools.py` — 도메인 도구 4종 `search_papers:83` · `get_repo_stats:140` · `get_package_info:184` · `search_community:232` + 캐시·재시도 `_get:36`
- `src/retriever.py:51` — 하이브리드 RAG 파이프라인 (`hybrid_search`)
- `src/guardrails.py` — 마스킹 `mask_secrets:50` · 간접 인젝션 정화 `sanitize_external:63` · 출력 검증 `verify_output:78`
- `src/server.py:43` — `POST /query` API
- `evaluation/run_eval.py:102` — LLM-as-Judge(`judge:54`) + RAGAS(`run_ragas:75`) 평가, 라운드 리포트 생성(`write_report:150`)
