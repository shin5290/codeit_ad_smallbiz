"""
Generate a responses.jsonl file by querying SmallBizRAG for predefined questions.

Usage:
  python evaluation/build_responses.py --model gpt-4o --output evaluation/responses.jsonl

Notes:
- Requires OPENAI_API_KEY
- Respects LLM_MODEL env override if set; otherwise uses CLI --model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from rag.chain import SmallBizRAG  # noqa: E402
from rag.prompts import UserContext  # noqa: E402


DEFAULT_QUESTIONS = [
    # strategy: budget allocation
    {
        "question": "신메뉴 출시했는데 예산 30만 원으로 인스타그램/네이버 어디에 얼마씩 집행하면 좋을까?",
        "task": "strategy",
    },
    # strategy: retention without coupons
    {
        "question": "쿠폰 없이도 재방문 늘릴 만한 리텐션 캠페인 아이디어 알려줘",
        "task": "strategy",
    },
    # trend
    {
        "question": "요즘 유행하는 카페 마케팅 트렌드가 뭐야?",
        "task": "trend",
    },
    # photo guide
    {
        "question": "네이버에 올릴 매장 사진을 어떻게 찍어야 효과적일까?",
        "task": "photo_guide",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Build responses.jsonl by running SmallBizRAG on preset questions.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/responses.jsonl"), help="Output JSONL path")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o"), help="LLM model override")
    parser.add_argument("--k", type=int, default=7, help="retrieval k")
    args = parser.parse_args()

    rag = SmallBizRAG(llm_model=args.model, use_reranker=False)

    user_ctx = UserContext(
        industry="cafe",
        location="강남",
        budget=300000,
        goal="신규 고객 유치",
        platform="instagram/naver",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        for item in DEFAULT_QUESTIONS:
            res = rag.query(
                question=item["question"],
                k=args.k,
                task=item.get("task"),
                user_context=user_ctx,
            )
            f.write(
                json.dumps(
                    {
                        "question": res["question"],
                        "answer": res["answer"],
                        "task": res.get("task"),
                        "intent": res.get("intent"),
                        "sources": res.get("sources"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"✔ wrote: {item['question']}")

    print(f"\nDone. Saved to {args.output}")


if __name__ == "__main__":
    main()
