"""
Chroma 벡터스토어 생성 (LangChain 기반)

기능:
- documents.jsonl을 Chroma 벡터스토어로 변환
- multilingual-e5-large 임베딩 사용
- 메타데이터 필터링 지원 (업종, 지역 등)
- 영구 저장 (persist)

출력:
- vectorstore/chroma_db/ 디렉토리에 Chroma DB 저장

사용법:
  python 04_build_vectorstore.py

필요 패키지:
  pip install langchain langchain-community chromadb sentence-transformers
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# --------------------------------------------
# 설정
# --------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "processed", "documents.jsonl")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore", "chroma_db")

# 임베딩 모델
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


# --------------------------------------------
# 로더
# --------------------------------------------
def load_documents(path: str) -> List[Dict[str, Any]]:
    """문서 로드"""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


# --------------------------------------------
# E5 임베딩 래퍼 (LangChain 호환)
# --------------------------------------------
class E5Embeddings:
    """
    E5 모델용 LangChain 호환 임베딩 클래스
    - 문서: "passage: " prefix
    - 쿼리: "query: " prefix
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        print(f"임베딩 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩 (passage prefix)"""
        prefixed = ["passage: " + t for t in texts]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩 (query prefix)"""
        prefixed = "query: " + text
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


# --------------------------------------------
# 메인
# --------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Chroma 벡터스토어 생성 (LangChain)")
    print("=" * 70)
    print(f"임베딩 모델: {EMBEDDING_MODEL}")
    print(f"저장 경로: {VECTORSTORE_DIR}")
    print()

    # 1. 문서 로드
    print("[1/4] 문서 로드")
    raw_docs = load_documents(DOCS_PATH)
    print(f"문서 수: {len(raw_docs)}개")

    # 2. LangChain Document 변환
    print("\n[2/4] LangChain Document 변환")
    from langchain_core.documents import Document

    documents = []
    for i, doc in enumerate(raw_docs):
        text = doc.get("text", "") or doc.get("content", {}).get("text", "")
        meta = doc.get("metadata", {})

        # 메타데이터 정리 (Chroma 필터링용)
        metadata = {
            "doc_id": str(doc.get("doc_id", i)),
            "title": meta.get("title", ""),
            "location": meta.get("location", ""),
            "industry": meta.get("industry", ""),
            "rating": float(meta.get("rating", 0)) if meta.get("rating") else 0.0,
            "source": "naver_place",
        }

        documents.append(Document(page_content=text, metadata=metadata))

    print(f"Document 객체 생성: {len(documents)}개")

    # 3. Chroma 벡터스토어 생성
    print("\n[3/4] Chroma 벡터스토어 생성")
    from langchain_community.vectorstores import Chroma

    # 임베딩 모델 초기화
    embeddings = E5Embeddings(EMBEDDING_MODEL)

    # 기존 DB 삭제 후 새로 생성
    import shutil
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
        print("  기존 DB 삭제")

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    # Chroma DB 생성
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name="smallbiz_places",
    )

    print(f"  벡터스토어 생성 완료: {vectorstore._collection.count()}개 문서")

    # 4. 검증 테스트
    print("\n[4/4] 검색 테스트")

    # 테스트 1: 기본 검색
    test_query = "강남역 근처 맛있는 카페"
    print(f"\n  쿼리: '{test_query}'")
    results = vectorstore.similarity_search(test_query, k=3)
    print("  Top-3 결과:")
    for i, doc in enumerate(results, 1):
        print(f"    {i}. [{doc.metadata['title']}]")
        print(f"       위치: {doc.metadata['location']}, 업종: {doc.metadata['industry']}")

    # 테스트 2: 메타데이터 필터링
    print(f"\n  쿼리: '{test_query}' + 필터: industry='cafe'")
    results_filtered = vectorstore.similarity_search(
        test_query,
        k=3,
        filter={"industry": "cafe"}
    )
    print("  필터링 결과:")
    for i, doc in enumerate(results_filtered, 1):
        print(f"    {i}. [{doc.metadata['title']}] - {doc.metadata['industry']}")

    # 테스트 3: 점수 포함 검색
    print(f"\n  쿼리 + 점수:")
    results_with_score = vectorstore.similarity_search_with_score(test_query, k=3)
    for i, (doc, score) in enumerate(results_with_score, 1):
        print(f"    {i}. [{doc.metadata['title']}] (score: {score:.4f})")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)
    print(f"\n저장 경로: {VECTORSTORE_DIR}")
    print("\n사용 예시:")
    print("""
    from langchain_community.vectorstores import Chroma

    # 벡터스토어 로드
    vectorstore = Chroma(
        persist_directory="vectorstore/chroma_db",
        embedding_function=E5Embeddings(),
        collection_name="smallbiz_places"
    )

    # 검색
    results = vectorstore.similarity_search("강남 카페", k=5)

    # 메타데이터 필터링
    results = vectorstore.similarity_search(
        "맛있는 음식점",
        k=5,
        filter={"location": "강남"}
    )

    # Retriever로 변환 (RAG 체인용)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    """)


if __name__ == "__main__":
    main()
