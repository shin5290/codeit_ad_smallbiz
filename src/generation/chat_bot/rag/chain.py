"""
LangChain 기반 RAG 체인 (프롬프트 엔지니어링 통합)

기능:
- 데이터 파이프라인에서 생성한 Chroma 벡터스토어 로드 (data/vectorstore/chroma_db)
- 프롬프트 엔지니어링 (prompts.py) 통합
- 태스크 자동 분류 (recommend, ad_copy, strategy, etc.)
- 메타데이터 필터링 지원
- Reranker 옵션 (선택)
- 대화 히스토리 지원

사용법:
  python -m rag.chain

필요 패키지:
  pip install langchain langchain-openai langchain-community chromadb sentence-transformers
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

# 프롬프트 모듈 임포트
from .prompts import PromptBuilder, UserContext, IntentRouter

# --------------------------------------------
# 설정
# --------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore" / "chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


# --------------------------------------------
# E5 임베딩 (큐잉 + 마이크로배치 지원)
# --------------------------------------------

# CPU 스레드 최적화 (벤치마크 결과: threads=2 최적)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
torch.set_num_threads(2)


@dataclass
class EmbedRequest:
    """임베딩 요청 객체"""
    text: str
    is_query: bool  # True: 쿼리, False: 문서
    future: Future = field(default_factory=Future)


class E5Embeddings:
    """
    E5 모델용 LangChain 호환 임베딩 클래스

    Features:
    - 큐잉: 동시 encode 방지 (Lock 기반)
    - 마이크로배치: 짧은 시간 동안 요청 모아서 배치 처리
    - CPU 최적화: threads=2, VRAM 0GB

    벤치마크 결과 (p95 기준):
    - threads=2, batch_size=1: ~200ms (short query)
    - threads=2, batch_size=4: ~87ms per sentence
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = "cpu",
        batch_wait_ms: int = 50,
        max_batch_size: int = 8,
        enable_micro_batch: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        # CPU 사용으로 VRAM 절약 (이미지 생성 모델에 GPU 전체 양보)
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

        # 마이크로배치 설정
        self.batch_wait_ms = batch_wait_ms  # 배치 대기 시간 (ms)
        self.max_batch_size = max_batch_size  # 최대 배치 크기
        self.enable_micro_batch = enable_micro_batch

        # 큐잉 (동시 encode 방지)
        self._lock = threading.Lock()
        self._queue: List[EmbedRequest] = []
        self._queue_lock = threading.Lock()

        # 마이크로배치 워커 스레드
        if enable_micro_batch:
            self._batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self._batch_thread.start()
            self._shutdown = False

    def _sanitize_text(self, text: str) -> str:
        """텍스트 전처리 (None, 빈 문자열, 인코딩 문제 해결)"""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return "empty query"
        text = text.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')
        return text

    def _encode_batch(self, texts: List[str], is_query: bool) -> List[List[float]]:
        """실제 인코딩 수행 (Lock으로 동시 실행 방지)"""
        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]

        with self._lock:
            embeddings = self.model.encode(prefixed, normalize_embeddings=True)

        return embeddings.tolist()

    def _batch_worker(self):
        """마이크로배치 워커 스레드"""
        while not getattr(self, '_shutdown', False):
            time.sleep(self.batch_wait_ms / 1000.0)

            # 큐에서 요청 가져오기
            with self._queue_lock:
                if not self._queue:
                    continue

                # 최대 batch_size 만큼 가져오기
                batch = self._queue[:self.max_batch_size]
                self._queue = self._queue[self.max_batch_size:]

            if not batch:
                continue

            # 쿼리/문서 분리
            query_requests = [r for r in batch if r.is_query]
            doc_requests = [r for r in batch if not r.is_query]

            # 쿼리 배치 처리
            if query_requests:
                texts = [self._sanitize_text(r.text) for r in query_requests]
                try:
                    embeddings = self._encode_batch(texts, is_query=True)
                    for req, emb in zip(query_requests, embeddings):
                        req.future.set_result(emb)
                except Exception as e:
                    for req in query_requests:
                        req.future.set_exception(e)

            # 문서 배치 처리
            if doc_requests:
                texts = [self._sanitize_text(r.text) for r in doc_requests]
                try:
                    embeddings = self._encode_batch(texts, is_query=False)
                    for req, emb in zip(doc_requests, embeddings):
                        req.future.set_result(emb)
                except Exception as e:
                    for req in doc_requests:
                        req.future.set_exception(e)

    def _submit_request(self, text: str, is_query: bool) -> Future:
        """마이크로배치 큐에 요청 제출"""
        request = EmbedRequest(text=text, is_query=is_query)

        with self._queue_lock:
            self._queue.append(request)

            # max_batch_size 도달하면 즉시 처리 트리거
            if len(self._queue) >= self.max_batch_size:
                pass  # 워커가 처리

        return request.future

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩 (오프라인 배치용 - 마이크로배치 미사용)"""
        # 오프라인 임베딩은 마이크로배치 없이 직접 처리 (효율성)
        sanitized = [self._sanitize_text(t) for t in texts]
        return self._encode_batch(sanitized, is_query=False)

    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩 (온라인용 - 마이크로배치 사용)"""
        sanitized = self._sanitize_text(text)

        if self.enable_micro_batch:
            # 마이크로배치 큐에 제출하고 결과 대기
            future = self._submit_request(sanitized, is_query=True)
            return future.result(timeout=10.0)  # 최대 10초 대기
        else:
            # 마이크로배치 비활성화 시 직접 처리
            return self._encode_batch([sanitized], is_query=True)[0]

    def shutdown(self):
        """워커 스레드 종료"""
        self._shutdown = True
        if hasattr(self, '_batch_thread'):
            self._batch_thread.join(timeout=1.0)


# --------------------------------------------
# RAG 체인 클래스
# --------------------------------------------
class SmallBizRAG:
    """
    소상공인 마케팅 상담 RAG 체인

    Features:
    - 프롬프트 엔지니어링 통합
    - 태스크 자동 분류
    - 대화 히스토리 관리
    - 메타데이터 필터링
    """

    def __init__(
        self,
        vectorstore_dir: Path | str = VECTORSTORE_DIR,
        llm_model: str = LLM_MODEL,
        use_reranker: bool = False,
    ):
        import chromadb
        from langchain_community.vectorstores import Chroma
        from langchain_openai import ChatOpenAI

        # 벡터스토어 로드 (chromadb client 직접 생성)
        print("벡터스토어 로드 중...")
        self.embeddings = E5Embeddings()

        # chromadb PersistentClient로 기존 DB 연결
        chroma_client = chromadb.PersistentClient(path=str(vectorstore_dir))

        self.vectorstore = Chroma(
            client=chroma_client,
            embedding_function=self.embeddings,
            collection_name="smallbiz_places",
        )
        print(f"  로드 완료: {self.vectorstore._collection.count()}개 문서")

        # LLM 초기화
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.7,
            api_key=OPENAI_API_KEY,
        )

        # 프롬프트 빌더
        self.prompt_builder = PromptBuilder()

        # 의도 분류기
        self.intent_router = IntentRouter()

        # Reranker (선택)
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            print("Reranker 로드 중...")
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

        # 대화 히스토리
        self.chat_history: List[Dict[str, str]] = []

    def _sanitize_text(self, text: str) -> str:
        """텍스트 전처리 (서로게이트 문자 제거)"""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        # 서로게이트 문자 제거
        text = text.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')
        return text

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        문서 검색 (+ 선택적 Reranking)

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter: 메타데이터 필터 (예: {"industry": "cafe"})

        Returns:
            검색된 문서 리스트
        """
        # 벡터 검색
        search_k = k * 3 if self.use_reranker else k
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=search_k,
            filter=filter,
        )

        # Reranking
        if self.use_reranker and self.reranker and len(results) > k:
            pairs = [(query, doc.page_content) for doc, _ in results]
            scores = self.reranker.predict(pairs)

            reranked = sorted(
                zip(scores, results),
                key=lambda x: x[0],
                reverse=True
            )
            results = [r for _, r in reranked[:k]]

        # 결과 포맷팅 (서로게이트 문자 제거)
        retrieved = []
        for doc, score in results[:k]:
            retrieved.append({
                "content": self._sanitize_text(doc.page_content),
                "metadata": doc.metadata,
                "score": float(score) if not self.use_reranker else None,
            })

        return retrieved

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        task: Optional[str] = None,
        user_context: Optional[UserContext] = None,
    ) -> str:
        """
        RAG 응답 생성 (프롬프트 엔지니어링 적용)

        Args:
            query: 사용자 질문
            retrieved_docs: 검색된 문서 리스트
            task: 태스크 유형 (None이면 자동 분류)
            user_context: 사용자 컨텍스트

        Returns:
            생성된 응답
        """
        # 태스크 자동 분류
        if task is None:
            task = self.prompt_builder.classify_task(query)

        # 프롬프트 생성
        messages = self.prompt_builder.build_prompt(
            query=query,
            retrieved_docs=retrieved_docs,
            task=task,
            user_context=user_context,
            chat_history=self.chat_history,
        )

        # LLM 호출
        response = self.llm.invoke(messages)

        # 히스토리 업데이트
        self._update_history(query, response.content)

        return response.content

    async def generate_stream(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        task: Optional[str] = None,
        user_context: Optional[UserContext] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        RAG 응답 스트리밍 생성 (프롬프트 엔지니어링 적용)
        """
        if task is None:
            task = self.prompt_builder.classify_task(query)

        if chat_history is None:
            chat_history = self.chat_history

        messages = self.prompt_builder.build_prompt(
            query=query,
            retrieved_docs=retrieved_docs,
            task=task,
            user_context=user_context,
            chat_history=chat_history,
        )

        stream = getattr(self.llm, "astream", None)
        if not callable(stream):
            response = self.llm.invoke(messages)
            content = getattr(response, "content", "") or ""
            if content:
                yield content
            return

        async for chunk in self.llm.astream(messages):
            if isinstance(chunk, str):
                content = chunk
            else:
                content = getattr(chunk, "content", None)
            if content:
                yield content

    def query(
        self,
        question: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
        user_context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        """
        RAG 파이프라인 실행 (검색 → 생성)

        Args:
            question: 사용자 질문
            k: 검색할 문서 수
            filter: 메타데이터 필터
            task: 태스크 유형 (None이면 자동 분류)
            user_context: 사용자 컨텍스트

        Returns:
            {
                "question": 질문,
                "answer": 답변,
                "task": 분류된 태스크,
                "intent": 의도 (rag/agent/mcp),
                "sources": 참조 매장 리스트,
            }
        """
        # 의도 분류
        intent = self.intent_router.classify(question)

        # 태스크 분류
        if task is None:
            task = self.prompt_builder.classify_task(question)

        # 검색
        retrieved = self.retrieve(question, k=k, filter=filter)

        # 지역 필터가 적용됐는데 데이터가 없을 때는 일반화된 답변 대신 지역 요청을 재확인
        if not retrieved and filter and filter.get("location"):
            location = filter.get("location")
            answer = (
                f"{location} 지역 사례 데이터가 아직 없어서 참고할 수 없어요. "
                "근처 동네나 다른 상권명으로도 찾아볼까요? "
                "업종/예산/목표를 다시 알려주시면 지역에 맞춰 다시 정리해 드릴게요."
            )
            self._update_history(question, answer)
            return {
                "question": question,
                "answer": answer,
                "task": task,
                "intent": intent,
                "sources": [],
            }

        # 생성
        answer = self.generate(
            question,
            retrieved,
            task=task,
            user_context=user_context,
        )

        return {
            "question": question,
            "answer": answer,
            "task": task,
            "intent": intent,
            "sources": [
                {
                    "title": doc["metadata"].get("title"),
                    "location": doc["metadata"].get("location"),
                    "industry": doc["metadata"].get("industry"),
                    "rating": doc["metadata"].get("rating"),
                }
                for doc in retrieved
            ],
        }

    def _update_history(self, query: str, answer: str):
        """대화 히스토리 업데이트"""
        # 서로게이트 문자 제거
        safe_query = self._sanitize_text(query)
        safe_answer = self._sanitize_text(answer)
        self.chat_history.append({"role": "user", "content": safe_query})
        self.chat_history.append({"role": "assistant", "content": safe_answer})

        # 최대 10턴 유지
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []


# --------------------------------------------
# 테스트
# --------------------------------------------
def main():
    print("=" * 70)
    print("LangChain RAG 체인 테스트 (프롬프트 엔지니어링 통합)")
    print("=" * 70)

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY 환경변수를 설정하세요")
        return

    # RAG 초기화
    rag = SmallBizRAG(use_reranker=False)

    # 사용자 컨텍스트
    user_ctx = UserContext(
        industry="cafe",
        location="강남",
        budget=100000,
        goal="신규 고객 유치",
    )

    # 테스트 1: 예산 배분/캠페인 전략
    print("\n" + "-" * 50)
    print("테스트 1: 예산 배분/캠페인 전략")
    print("-" * 50)

    result = rag.query(
        question="신메뉴 출시했는데 예산 30만 원으로 인스타그램/네이버 어디에 얼마씩 집행하면 좋을까?",
        k=3,
        user_context=user_ctx,
    )

    print(f"\n질문: {result['question']}")
    print(f"태스크: {result['task']}")
    print(f"\n답변:\n{result['answer']}")

    # 테스트 2: 리뷰/쿠폰 없이 리텐션 캠페인 설계
    print("\n" + "-" * 50)
    print("테스트 2: 리텐션 캠페인 설계")
    print("-" * 50)

    result = rag.query(
        question="쿠폰 없이도 재방문 늘릴 만한 리텐션 캠페인 아이디어 알려줘",
        k=3,
        user_context=user_ctx,
    )

    print(f"\n질문: {result['question']}")
    print(f"태스크: {result['task']}")
    print(f"\n답변:\n{result['answer']}")

    # 테스트 3: 트렌드 질문 (의도 분류 확인)
    print("\n" + "-" * 50)
    print("테스트 3: 트렌드 질문 (의도 분류)")
    print("-" * 50)

    result = rag.query(
        question="요즘 유행하는 카페 마케팅 트렌드가 뭐야?",
        k=3,
    )

    print(f"\n질문: {result['question']}")
    print(f"태스크: {result['task']}")
    print(f"의도: {result['intent']}  ← Agent 사용 권장")
    print(f"\n답변:\n{result['answer'][:500]}...")

    # 테스트 4: 매장 사진 촬영 가이드 (네이버 업로드)
    print("\n" + "-" * 50)
    print("테스트 4: 매장 사진 촬영 가이드 (네이버 업로드)")
    print("-" * 50)

    result = rag.query(
        question="네이버에 올릴 매장 사진을 어떻게 찍어야 효과적일까?",
        k=3,
        user_context=user_ctx,
    )

    print(f"\n질문: {result['question']}")
    print(f"\n답변:\n{result['answer']}")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
