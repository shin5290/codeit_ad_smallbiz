"""
임베딩 평가 v3 - v5 문서 기본 + prefix/모델 설정 분리

변경점:
- 기본 입력: documents_v5.jsonl
- 합성 쿼리 파일: synthetic_queries_v5.json
- e5에 passage/query prefix 적용, bge는 prefix 없음

사용법:
  DOCS_PATH=data/processed/documents_v5.jsonl python evaluation/04_evaluate_embeddings.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------
# 설정
# --------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"
DOCS_PATH = os.getenv("DOCS_PATH", str(DATA_DIR / "processed" / "documents_v5.jsonl"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_DOCS_FOR_EVAL = int(os.getenv("NUM_DOCS_FOR_EVAL", "100"))
QUERIES_PER_DOC = int(os.getenv("QUERIES_PER_DOC", "3"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QUERY_GEN_MODEL = os.getenv("QUERY_GEN_MODEL", "gpt-4o-mini")
TOP_K_VALUES = [1, 3, 5, 10]

# --------------------------------------------
# 모델 설정 (필요시 추가/수정)
# --------------------------------------------
EMBEDDING_MODELS = {
    "bge-m3": {
        "type": "sentence_transformer",
        "model_name": "BAAI/bge-m3",
        "dimension": 1024,
        "doc_prefix": "",
        "query_prefix": "",
    },
    "multilingual-e5-large": {
        "type": "sentence_transformer",
        "model_name": "intfloat/multilingual-e5-large",
        "dimension": 1024,
        "doc_prefix": "passage: ",
        "query_prefix": "query: ",
    },
    "text-embedding-3-small": {
        "type": "openai",
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "doc_prefix": "",
        "query_prefix": "",
    },
}


# --------------------------------------------
# 로더/LLM 쿼리 생성
# --------------------------------------------
def load_documents(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


class SyntheticQueryGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_queries(self, doc: Dict[str, Any], num_queries: int = 3) -> List[str]:
        text = doc.get("text", "") or doc.get("content", {}).get("text", "")
        meta = doc.get("metadata", {})
        title = meta.get("title", "")
        location = meta.get("location", "")
        industry = meta.get("industry", "")

        prompt = f"""다음은 {industry} 업종 매장 정보입니다.
제목: {title}
지역: {location}
내용: {text[:600]}
사용자가 이 매장을 찾기 위해 입력할 질문 {num_queries}개를 생성하세요.
- 한국어, 짧고 구어체
- 지역+특징이 드러나게
- 서로 다른 관점 (위치/메뉴/후기/평점 등)

형식:
1. ...
2. ...
3. ..."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            result = resp.choices[0].message.content
            queries: List[str] = []
            for line in result.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    q = line.split(".", 1)[-1].strip().strip("[]\"'")
                    if q:
                        queries.append(q)
            return queries[:num_queries]
        except Exception as e:  # noqa: BLE001
            print(f"쿼리 생성 실패: {e}")
            return []


# --------------------------------------------
# 임베딩 래퍼
# --------------------------------------------
class EmbeddingModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = config["type"]
        self.model_name = config["model_name"]
        self.dimension = config["dimension"]
        self.doc_prefix = config.get("doc_prefix", "")
        self.query_prefix = config.get("query_prefix", "")
        self._model = None
        self._client = None

    def _load_model(self) -> None:
        if self.model_type == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            from sentence_transformers import SentenceTransformer

            print(f"  로딩 중: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if self._model is None and self._client is None:
            self._load_model()

        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        if (not is_query) and self.doc_prefix:
            texts = [self.doc_prefix + t for t in texts]

        if self.model_type == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        batch = 100
        for i in range(0, len(texts), batch):
            resp = self._client.embeddings.create(
                model=self.model_name, input=texts[i : i + batch]
            )
            embeddings.extend([item.embedding for item in resp.data])
            time.sleep(0.05)
        return np.array(embeddings, dtype=np.float32)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


# --------------------------------------------
# 메트릭
# --------------------------------------------
def compute_metrics(
    queries: List[Dict[str, Any]],
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k_values: List[int],
) -> Dict[str, float]:
    import faiss

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(doc_embeddings)
    faiss.normalize_L2(query_embeddings)
    index.add(doc_embeddings)

    max_k = max(k_values)
    _, indices = index.search(query_embeddings, max_k)

    recalls = {k: [] for k in k_values}
    mrrs: List[float] = []

    for i, q in enumerate(queries):
        target = q["doc_id"]
        retrieved = indices[i].tolist()
        for k in k_values:
            recalls[k].append(1 if target in retrieved[:k] else 0)
        try:
            rank = retrieved.index(target) + 1
            mrrs.append(1.0 / rank)
        except ValueError:
            mrrs.append(0.0)

    results: Dict[str, float] = {}
    for k in k_values:
        results[f"Recall@{k}"] = float(np.mean(recalls[k]))
    results["MRR"] = float(np.mean(mrrs))
    return results


# --------------------------------------------
# 실행
# --------------------------------------------
def run_evaluation() -> pd.DataFrame:
    print("=" * 70)
    print("임베딩 모델 평가 v3 (v5 문서)")
    print("=" * 70)

    # 1. 문서 로드
    docs = load_documents(DOCS_PATH)
    print(f"전체 문서: {len(docs)}개")
    eval_docs = random.sample(docs, NUM_DOCS_FOR_EVAL) if len(docs) > NUM_DOCS_FOR_EVAL else docs
    print(f"평가용 문서: {len(eval_docs)}개")
    doc_id_to_idx = {doc.get("doc_id", i): i for i, doc in enumerate(docs)}

    # 2. 합성 쿼리 생성/로드
    queries_file = OUTPUT_DIR / "synthetic_queries_v5.json"
    if queries_file.exists():
        print(f"기존 쿼리 로드: {queries_file}")
        with queries_file.open("r", encoding="utf-8") as f:
            queries = json.load(f)
    else:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 환경변수를 설정하세요")
        generator = SyntheticQueryGenerator(OPENAI_API_KEY, QUERY_GEN_MODEL)
        queries = []
        for doc in tqdm(eval_docs, desc="쿼리 생성"):
            doc_id = doc.get("doc_id")
            if doc_id is None:
                continue
            gen = generator.generate_queries(doc, QUERIES_PER_DOC)
            for q in gen:
                queries.append(
                    {
                        "query": q,
                        "doc_id": doc_id_to_idx.get(doc_id, -1),
                        "doc_title": doc.get("metadata", {}).get("title", ""),
                    }
                )
            time.sleep(0.2)
        with queries_file.open("w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        print(f"쿼리 저장: {queries_file}")

    queries = [q for q in queries if q["doc_id"] >= 0]
    print(f"유효 쿼리: {len(queries)}개")

    # 3. 텍스트 준비
    doc_texts = [doc.get("text", "") or doc.get("content", {}).get("text", "") for doc in docs]
    query_texts = [q["query"] for q in queries]

    # 4. 모델별 평가
    all_results: List[Dict[str, float]] = []
    for model_name, cfg in EMBEDDING_MODELS.items():
        print(f"\n{'='*50}\n모델: {model_name}\n{'='*50}")
        model = EmbeddingModel(cfg)
        print("  문서 임베딩 중...")
        doc_emb = model.embed(doc_texts, is_query=False)
        print("  쿼리 임베딩 중...")
        query_emb = model.embed(query_texts, is_query=True)
        print("  메트릭 계산 중...")
        metrics = compute_metrics(queries, doc_emb, query_emb, TOP_K_VALUES)
        metrics["model"] = model_name
        all_results.append(metrics)
        for k, v in metrics.items():
            if k != "model":
                print(f"    {k}: {v:.4f}")
        del model, doc_emb, query_emb
        import gc

        gc.collect()

    df = pd.DataFrame(all_results)
    cols = ["model"] + [f"Recall@{k}" for k in TOP_K_VALUES] + ["MRR"]
    df = df[cols]
    print("\n최종 결과:\n")
    print(df.to_string(index=False))

    out_csv = OUTPUT_DIR / "embedding_evaluation_results_v3.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n결과 저장: {out_csv}")
    return df


if __name__ == "__main__":
    run_evaluation()
