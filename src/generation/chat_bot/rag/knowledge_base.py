"""
백엔드 연동용 Knowledge Base 구현

백엔드의 ConsultingKnowledgeBase 인터페이스를 구현하여
SmallBizRAG와 연결합니다.

사용법:
    from rag.knowledge_base import SmallBizKnowledgeBase, get_knowledge_base

    # 싱글톤 인스턴스 사용
    kb = get_knowledge_base()
    results = kb.search("카페 마케팅 전략", category="faq", limit=3)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------
# ConsultingKnowledgeBase 추상 인터페이스
# (백엔드와 동일한 인터페이스)
# --------------------------------------------

class ConsultingKnowledgeBase(ABC):
    """
    상담 지식베이스 인터페이스 (추상 클래스)
    백엔드의 ConsultingKnowledgeBase와 동일한 인터페이스
    """

    @abstractmethod
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict]:
        """
        정적 문서 검색

        Args:
            query: 검색 쿼리 (사용자 메시지 또는 키워드)
            category: 문서 카테고리 필터 (faq, generation_guide, modification_guide, industry_tips)
            limit: 반환할 문서 수 (기본값: 3)

        Returns:
            검색 결과 리스트:
            [{"content": str, "source": str, "score": float, "category": str, "metadata": dict}, ...]
        """
        pass

    def get_categories(self) -> List[str]:
        """사용 가능한 카테고리 목록 반환"""
        return ["faq", "generation_guide", "modification_guide", "industry_tips"]


# --------------------------------------------
# SmallBizKnowledgeBase 구현
# --------------------------------------------

class SmallBizKnowledgeBase(ConsultingKnowledgeBase):
    """
    소상공인 마케팅 지식베이스 구현

    SmallBizRAG의 벡터 검색을 백엔드 인터페이스에 맞게 래핑합니다.
    """

    def __init__(
        self,
        vectorstore_dir: Optional[Path] = None,
        use_reranker: bool = False,
    ):
        """
        Args:
            vectorstore_dir: 벡터스토어 디렉토리 경로 (None이면 기본 경로 사용)
            use_reranker: Reranker 사용 여부
        """
        from .chain import SmallBizRAG

        # RAG 인스턴스 생성
        if vectorstore_dir:
            self.rag = SmallBizRAG(
                vectorstore_dir=vectorstore_dir,
                use_reranker=use_reranker,
            )
        else:
            self.rag = SmallBizRAG(use_reranker=use_reranker)

        # 카테고리 → 메타데이터 필터 매핑
        self._category_filters = {
            "faq": None,  # 전체 검색
            "generation_guide": None,
            "modification_guide": None,
            "industry_tips": None,
        }

        print("SmallBizKnowledgeBase initialized")

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict]:
        """
        문서 검색 (백엔드 인터페이스 호환)

        Args:
            query: 검색 쿼리
            category: 카테고리 필터 (현재는 사용하지 않음, 향후 확장 가능)
            limit: 반환할 문서 수

        Returns:
            검색 결과 리스트 (백엔드 형식)
        """
        # 메타데이터 필터 설정 (필요시)
        filter_kwargs = self._category_filters.get(category) if category else None

        # RAG 검색 실행
        retrieved = self.rag.retrieve(
            query=query,
            k=limit,
            filter=filter_kwargs,
        )

        # 백엔드 형식으로 변환
        results = []
        for doc in retrieved:
            metadata = doc.get("metadata", {})

            # 카테고리 추론 (메타데이터 기반)
            inferred_category = self._infer_category(metadata, query)

            results.append({
                "content": doc.get("content", ""),
                "source": metadata.get("title", "unknown"),
                "score": doc.get("score", 0.0) or 0.85,  # score가 None이면 기본값
                "category": category or inferred_category,
                "metadata": {
                    "industry": metadata.get("industry"),
                    "location": metadata.get("location"),
                    "title": metadata.get("title"),
                    "rating": metadata.get("rating"),
                    "keywords": self._extract_keywords(doc.get("content", "")),
                },
            })

        return results

    def _infer_category(self, metadata: Dict, query: str) -> str:
        """메타데이터와 쿼리를 기반으로 카테고리 추론"""
        query_lower = query.lower()

        # 생성 관련 키워드
        if any(kw in query_lower for kw in ["만들", "생성", "제작", "이미지", "광고"]):
            return "generation_guide"

        # 수정 관련 키워드
        if any(kw in query_lower for kw in ["수정", "변경", "바꿔", "다시"]):
            return "modification_guide"

        # 업종별 팁
        industry = metadata.get("industry")
        if industry:
            return "industry_tips"

        # 기본값
        return "faq"

    def _extract_keywords(self, content: str, max_keywords: int = 5) -> List[str]:
        """콘텐츠에서 키워드 추출 (간단한 구현)"""
        # 주요 마케팅 키워드 목록
        marketing_keywords = [
            "카페", "커피", "음식점", "맛집", "베이커리", "디저트",
            "인스타그램", "네이버", "유튜브", "SNS", "마케팅",
            "광고", "프로모션", "할인", "이벤트", "쿠폰",
            "신규", "재방문", "고객", "매출", "예산",
        ]

        found = []
        content_lower = content.lower()
        for kw in marketing_keywords:
            if kw in content_lower and kw not in found:
                found.append(kw)
                if len(found) >= max_keywords:
                    break

        return found

    def get_rag_instance(self):
        """RAG 인스턴스 직접 접근 (고급 사용)"""
        return self.rag


# --------------------------------------------
# 확장: Agent 통합 Knowledge Base
# --------------------------------------------

class SmallBizConsultingKnowledgeBase(SmallBizKnowledgeBase):
    """
    Agent 기능이 통합된 확장 Knowledge Base

    트렌드 질문 등 웹 검색이 필요한 경우 Agent를 활용합니다.
    """

    def __init__(
        self,
        vectorstore_dir: Optional[Path] = None,
        use_reranker: bool = False,
        use_agent: bool = True,
    ):
        super().__init__(vectorstore_dir, use_reranker)

        self.use_agent = use_agent
        self._agent = None

        if use_agent:
            try:
                from ..agent.agent import TrendAgent
                self._agent = TrendAgent(verbose=False)
                print("TrendAgent integrated")
            except ImportError:
                print("Warning: TrendAgent not available")
                self.use_agent = False

    def search_with_agent(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Agent를 활용한 확장 검색

        트렌드/실시간 정보가 필요한 경우 웹 검색도 수행합니다.

        Returns:
            {
                "rag_results": [...],  # 벡터DB 검색 결과
                "web_results": [...],  # 웹 검색 결과 (트렌드 질문 시)
                "combined_answer": str,  # 통합 답변 (Agent 사용 시)
            }
        """
        # 기본 RAG 검색
        rag_results = self.search(query, category, limit)

        result = {
            "rag_results": rag_results,
            "web_results": [],
            "combined_answer": None,
        }

        # 트렌드 질문인지 확인
        if self._agent and self._is_trend_query(query):
            try:
                agent_result = self._agent.run(query)
                result["combined_answer"] = agent_result.get("answer")
                result["web_results"] = [
                    step for step in agent_result.get("steps", [])
                    if step.get("tool") == "web_search"
                ]
            except Exception as e:
                print(f"Agent search failed: {e}")

        return result

    def _is_trend_query(self, query: str) -> bool:
        """트렌드 관련 질문인지 확인"""
        trend_keywords = ["요즘", "최근", "트렌드", "유행", "밈", "2024", "2025", "최신"]
        query_lower = query.lower()
        return any(kw in query_lower for kw in trend_keywords)


# --------------------------------------------
# 싱글톤 인스턴스
# --------------------------------------------

_knowledge_base_instance: Optional[SmallBizKnowledgeBase] = None


def get_knowledge_base(
    use_agent: bool = False,
    use_reranker: bool = False,
) -> SmallBizKnowledgeBase:
    """
    Knowledge Base 싱글톤 인스턴스 반환

    Args:
        use_agent: Agent 기능 통합 여부 (트렌드 검색 등)
        use_reranker: Reranker 사용 여부

    Returns:
        SmallBizKnowledgeBase 또는 SmallBizConsultingKnowledgeBase 인스턴스
    """
    global _knowledge_base_instance

    if _knowledge_base_instance is None:
        if use_agent:
            _knowledge_base_instance = SmallBizConsultingKnowledgeBase(
                use_reranker=use_reranker,
                use_agent=True,
            )
        else:
            _knowledge_base_instance = SmallBizKnowledgeBase(
                use_reranker=use_reranker,
            )
        print("KnowledgeBase singleton instance created")

    return _knowledge_base_instance


def reset_knowledge_base():
    """싱글톤 인스턴스 리셋 (테스트용)"""
    global _knowledge_base_instance
    _knowledge_base_instance = None


# --------------------------------------------
# 테스트
# --------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SmallBizKnowledgeBase 테스트")
    print("=" * 70)

    # 기본 Knowledge Base 테스트
    kb = get_knowledge_base()

    print("\n[테스트 1] 카페 마케팅 검색")
    results = kb.search("카페 인스타그램 마케팅 전략", limit=3)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['source']} (score: {r['score']:.2f})")
        print(f"   카테고리: {r['category']}")
        print(f"   내용: {r['content'][:100]}...")

    print("\n[테스트 2] 음식점 프로모션 검색")
    results = kb.search("음식점 할인 프로모션 아이디어", category="industry_tips", limit=3)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['source']} (score: {r['score']:.2f})")
        print(f"   업종: {r['metadata'].get('industry')}")

    print("\n" + "=" * 70)
    print("테스트 완료!")
