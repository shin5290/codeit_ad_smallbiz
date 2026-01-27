#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ and value:
            os.environ[key] = value


def main() -> int:
    repo_root = _project_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _load_env_file(repo_root / ".env")

    print("env:")
    for key in ("OPENAI_API_KEY", "LLM_MODEL", "TAVILY_API_KEY"):
        value = os.getenv(key)
        if value:
            print(f"  {key}: set (len={len(value)})")
        else:
            print(f"  {key}: missing")

    try:
        from src.generation.chat_bot.rag import SmallBizKnowledgeBase
    except Exception as exc:
        print(f"import error: {exc.__class__.__name__}: {exc}")
        return 1

    try:
        kb = SmallBizKnowledgeBase(use_reranker=False)
    except Exception as exc:
        print(f"init error: {exc.__class__.__name__}: {exc}")
        return 1

    query = "cafe marketing strategy"
    try:
        results = kb.search(query=query, limit=3)
    except Exception as exc:
        print(f"search error: {exc.__class__.__name__}: {exc}")
        return 1

    print(f"query: {query}")
    print(f"results: {len(results)}")
    for idx, item in enumerate(results, start=1):
        source = item.get("source") or "unknown"
        score = item.get("score")
        if isinstance(score, (int, float)):
            score_part = f"{score:.4f}"
        else:
            score_part = "n/a"
        content = (item.get("content") or "").strip().replace("\n", " ")
        snippet = content[:120]
        print(f"{idx}. source={source} score={score_part} snippet={snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
