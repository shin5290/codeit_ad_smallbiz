"""
Prompt evaluation script: rule-based checks + LLM-as-a-judge.

입력: JSONL 파일
  {
    "question": "...",
    "answer": "...",
    "task": "recommend|strategy|trend|photo_guide|general",  # optional
    "sources": [...]
  }

평가:
- Rule-based: 섹션/출처/금칙어/태스크별 필수 블록 확인
- LLM judge: 구체성/근거/구조/안전성 0~10점 채점

사용 예:
  python evaluation/evaluate_prompts.py --inputs responses.jsonl --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_openai import ChatOpenAI

from prompts import PromptBuilder  # noqa: E402


# -----------------------------
# Rule-based evaluation
# -----------------------------

BANNED_TERMS = ["반드시", "100%", "확실히", "무조건", "평점", "별점"]

REQUIRED_SECTIONS = {
    "recommend": ["한눈에", "추천", "출처"],
    "strategy": ["현재 상황", "실행", "예상", "주의", "다음", "출처"],
    "trend": ["트렌드", "적용", "리스크", "출처"],
    "photo_guide": ["세트리스트", "설정", "포맷", "업로드", "검수"],
    "general": ["출처"],
}


def rule_based_check(answer: str, task: str) -> Dict[str, bool]:
    """Check structure, banned terms, sources, and task sections."""
    checks = {
        "has_source": "출처" in answer,
        "has_banned_terms": any(term in answer for term in BANNED_TERMS),
    }

    required = REQUIRED_SECTIONS.get(task, REQUIRED_SECTIONS["general"])
    checks["sections_ok"] = all(k in answer for k in required)

    return checks


# -----------------------------
# LLM judge
# -----------------------------

JUDGE_SYSTEM = """You are a strict evaluator for a small-business marketing assistant.
Score the assistant answer on these aspects (0-10 each):
- specificity: actionable, with numbers/frequencies/budgets
- evidence: cites sources/cases from provided data, avoids hallucination
- structure: follows requested sections/order, concise
- safety: no overpromising, no guarantees, no prohibited channels (old Facebook ads), no rating/score mentions
Respond ONLY in JSON:
{"scores": {"specificity": x, "evidence": x, "structure": x, "safety": x}, "notes": ["...","..."]}"""


def llm_judge(llm: ChatOpenAI, question: str, answer: str, task: str) -> Dict[str, object]:
    user_prompt = f"""[Task: {task}]
[Question]
{question}

[Answer]
{answer}
"""
    resp = llm.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
    )
    try:
        parsed = json.loads(resp.content)
    except Exception:
        parsed = {"error": "parse_error", "raw": resp.content}
    return parsed


# -----------------------------
# Runner
# -----------------------------


def load_records(path: Path, limit: Optional[int]) -> List[Dict[str, object]]:
    records = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            records.append(json.loads(line))
    return records


def classify_task(query: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return PromptBuilder().classify_task(query)


def evaluate(inputs: Path, model: str, limit: Optional[int]) -> Dict[str, object]:
    llm = ChatOpenAI(model=model, temperature=0.0)

    records = load_records(inputs, limit)
    results = []

    for rec in records:
        question = rec.get("question", "")
        answer = rec.get("answer", "")
        task = classify_task(question, rec.get("task"))

        rule = rule_based_check(answer, task)
        judge = llm_judge(llm, question, answer, task)

        results.append(
            {
                "question": question,
                "task": task,
                "rule": rule,
                "judge": judge,
            }
        )

    return {"results": results}


def aggregate(report: Dict[str, object]) -> Dict[str, object]:
    scores = {"specificity": [], "evidence": [], "structure": [], "safety": []}
    rule_violations = {"has_banned_terms": 0, "sections_fail": 0, "missing_source": 0}

    for item in report["results"]:
        rule = item["rule"]
        judge = item["judge"]

        if rule.get("has_banned_terms"):
            rule_violations["has_banned_terms"] += 1
        if not rule.get("sections_ok"):
            rule_violations["sections_fail"] += 1
        if not rule.get("has_source"):
            rule_violations["missing_source"] += 1

        jscores = judge.get("scores", {}) if isinstance(judge, dict) else {}
        for k in scores:
            if jscores.get(k) is not None:
                scores[k].append(jscores[k])

    avg_scores = {k: round(sum(v) / len(v), 2) if v else None for k, v in scores.items()}
    return {"avg_scores": avg_scores, "rule_violations": rule_violations}


def main():
    parser = argparse.ArgumentParser(description="Evaluate prompt outputs (rule + LLM judge)")
    parser.add_argument("--inputs", required=True, type=Path, help="JSONL file with question/answer[/task]")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for judge (e.g., gpt-4o)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on records")
    parser.add_argument("--summary", action="store_true", help="Print aggregated summary")
    args = parser.parse_args()

    report = evaluate(args.inputs, model=args.model, limit=args.limit)

    if args.summary:
        summary = aggregate(report)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
