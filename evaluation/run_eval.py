"""인-아웃 세트 자체 평가: LLM-as-Judge 통과율 + RAGAS 지표 → evaluation/round{N}_report.md

사용법 (프로젝트 루트에서):
    python -m evaluation.run_eval --round 1
    python -m evaluation.run_eval --round 2 --ids 1,2,14     # 일부 케이스만
    python -m evaluation.run_eval --round 1 --no-ragas        # RAGAS 생략

통과 기준 = (Judge: 모든 expected_traits 충족 AND forbidden 미포함) AND (expected_tools 가 모두 호출됨)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from src.agent import run_agent
from src.config import get_embeddings, get_llm, today

EVAL_DIR = Path(__file__).resolve().parent


class Verdict(BaseModel):
    traits: list[bool] = Field(description="expected_traits 각 항목 충족 여부 (순서 그대로)")
    forbidden_found: list[str] = Field(description="답변에 실제로 나타난 forbidden 항목 (없으면 빈 리스트)")
    reason: str = Field(description="판정 근거 1~2문장")


JUDGE_PROMPT = """당신은 엄격한 평가자입니다. AI 어시스턴트의 답변이 기준을 지켰는지 판정하세요.
오늘 날짜는 {today} 입니다. 이 어시스턴트는 실시간 API 로 데이터를 조회하므로, 오늘 이전 날짜는 정상이며 날조가 아닙니다.

[사용자 질문]
{input}

[답변]
{answer}

[expected_traits] (각각 충족했는지 true/false, 순서 유지)
{traits}

[forbidden] (답변에 나타났으면 forbidden_found 에 그대로 적기)
{forbidden}

판정 지침: 의미가 충족되면 표현이 달라도 true. 근거 없이 지어낸 수치·링크가 보이면 관련 forbidden 에 해당합니다."""


def split(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(";") if x.strip()]


def judge(case: dict, answer: str) -> Verdict:
    traits, forbidden = split(case["expected_traits"]), split(case["forbidden"])
    v: Verdict = get_llm().with_structured_output(Verdict).invoke(JUDGE_PROMPT.format(
        today=today(), input=case["input"], answer=answer,
        traits="\n".join(f"{i+1}. {t}" for i, t in enumerate(traits)),
        forbidden="\n".join(f"- {f}" for f in forbidden) or "(없음)",
    ))
    v.traits = (v.traits + [False] * len(traits))[: len(traits)]
    return v


def _bedrock_finished(response) -> bool:
    for gens in response.generations:
        for g in gens:
            meta = getattr(getattr(g, "message", None), "response_metadata", {}) or {}
            reason = meta.get("stopReason") or meta.get("stop_reason") or (g.generation_info or {}).get("finish_reason")
            if reason not in (None, "end_turn", "stop_sequence", "tool_use", "stop"):
                return False
    return True


def run_ragas(rows: list[dict]) -> dict:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    # reference 정답이 없는 리서치 서비스이므로 expected_traits 를 참조 기준으로 사용
    data = [{
        "user_input": r["input"], "response": r["answer"],
        "retrieved_contexts": [c["text"] for c in r["contexts"]],
        "reference": " / ".join(split(r["expected_traits"])),
    } for r in rows if r["contexts"]]
    if not data:
        return {}
    result = evaluate(
        EvaluationDataset.from_list(data),
        metrics=[LLMContextRecall(), LLMContextPrecisionWithReference(), Faithfulness(), ResponseRelevancy()],
        # ragas 는 Bedrock Converse 의 종료 신호(response_metadata.stopReason)를 몰라 전부 '미완료'로 처리 → 직접 판별
        llm=LangchainLLMWrapper(get_llm(), is_finished_parser=_bedrock_finished), embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        run_config=RunConfig(max_workers=4, timeout=300), show_progress=True,
    )
    df = result.to_pandas()
    names = {"context_recall": "context_recall", "llm_context_precision_with_reference": "context_precision",
             "faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy"}
    return {nice: round(float(df[col].mean()), 3) for col, nice in names.items() if col in df}


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949) 대비
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--rejudge", action="store_true", help="에이전트는 다시 돌리지 않고 저장된 results_round{N}.json 답변만 재채점")
    args = ap.parse_args()

    if args.rejudge:
        rows = json.loads((EVAL_DIR / f"results_round{args.round}.json").read_text(encoding="utf-8"))
        for r in rows:
            v = judge(r, r["answer"])
            r["verdict"] = v.model_dump()
            r["passed"] = all(v.traits) and not v.forbidden_found and not r["missing_tools"]
            print(f"#{r['id']:>2} [{r['category']:<9}] {'PASS' if r['passed'] else 'FAIL'} (재채점)  {r['input'][:40]}")
        ragas_scores = {} if args.no_ragas else run_ragas(rows)
        (EVAL_DIR / f"results_round{args.round}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        write_report(args.round, rows, ragas_scores)
        return

    cases = list(csv.DictReader(open(EVAL_DIR / "test_queries.csv", encoding="utf-8")))
    if args.ids:
        want = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in want]

    rows = []
    for c in cases:
        t0 = time.time()
        try:
            out = run_agent(c["input"])
        except Exception as e:  # noqa: BLE001
            out = {"answer": f"[ERROR] {type(e).__name__}: {e}", "contexts": [], "trace": []}
        secs = time.time() - t0
        called = sorted({t["step"] for t in out["trace"]})
        missing_tools = [t for t in split(c["expected_tools"]) if t not in called]
        v = judge(c, out["answer"])
        passed = all(v.traits) and not v.forbidden_found and not missing_tools
        rows.append({**c, **out, "seconds": round(secs, 1), "called": called, "missing_tools": missing_tools,
                     "verdict": v.model_dump(), "passed": passed})
        print(f"#{c['id']:>2} [{c['category']:<9}] {'PASS' if passed else 'FAIL'} {secs:5.1f}s  {c['input'][:40]}")

    ragas_scores = {} if args.no_ragas else run_ragas(rows)
    (EVAL_DIR / f"results_round{args.round}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(args.round, rows, ragas_scores)


def write_report(rnd: int, rows: list[dict], ragas_scores: dict):
    n, p = len(rows), sum(r["passed"] for r in rows)
    by_cat = {}
    for r in rows:
        k = by_cat.setdefault(r["category"], [0, 0])
        k[0] += r["passed"]
        k[1] += 1
    lines = [f"# Round {rnd} 자체 평가 ({today()})", "", "## RAGAS"]
    for k in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
        lines.append(f"- {k}: {ragas_scores.get(k, '미측정')}")
    lines += ["", "> reference 는 expected_traits 로 대체 (리서치 서비스 특성상 고정 정답이 없음)", "",
              "## 인-아웃 세트 통과율", f"- **{p} / {n} 통과 ({p / max(n, 1):.0%})**"]
    lines += [f"- {cat}: {a} / {b}" for cat, (a, b) in by_cat.items()]
    avg = sum(r["seconds"] for r in rows) / max(n, 1)
    lines += [f"- 평균 응답 시간: {avg:.1f}초", "", "## 케이스별 결과", "",
              "| id | 분류 | 결과 | 시간 | 미충족 traits | forbidden 발견 | 누락 도구 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        traits = split(r["expected_traits"])
        unmet = [t for t, ok in zip(traits, r["verdict"]["traits"]) if not ok]
        lines.append(f"| {r['id']} | {r['category']} | {'✅' if r['passed'] else '❌'} | {r['seconds']}s | "
                     f"{'; '.join(unmet) or '-'} | {'; '.join(r['verdict']['forbidden_found']) or '-'} | {'; '.join(r['missing_tools']) or '-'} |")
    lines += ["", "## 실패 케이스 분석", ""]
    for r in rows:
        if not r["passed"]:
            lines.append(f"- **#{r['id']} {r['input']}** — {r['verdict']['reason']}")
    lines += ["", "## 다음 라운드 개선 계획", "", "- (실패 원인별 수정 사항을 적으세요)"]
    path = EVAL_DIR / f"round{rnd}_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{p}/{n} 통과 · RAGAS {ragas_scores} → {path}")


if __name__ == "__main__":
    main()
