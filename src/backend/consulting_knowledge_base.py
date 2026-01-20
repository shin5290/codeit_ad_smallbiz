"""
상담 지식베이스 모듈

이 모듈은 RAG 챗봇의 상담 응답 생성에 사용되는 지식베이스 인터페이스를 정의합니다.
- ConsultingKnowledgeBase: 추상 인터페이스 (팀원이 구현)
- MockKnowledgeBase: 테스트/개발용 Mock 구현
"""

from typing import Optional, List, Dict
from abc import ABC, abstractmethod

from src.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================
# ConsultingKnowledgeBase: 추상 인터페이스
# =====================================================

class ConsultingKnowledgeBase(ABC):
    """
    상담 지식베이스 인터페이스 (추상 클래스)
    팀원이 이 인터페이스를 구현하여 실제 VectorDB와 연동합니다.
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


# =====================================================
# MockKnowledgeBase: 테스트/개발용 Mock 구현
# =====================================================

class MockKnowledgeBase(ConsultingKnowledgeBase):
    """Mock 지식베이스 (테스트/개발용)"""

    def __init__(self):
        self._mock_data = self._build_mock_data()
        logger.info("MockKnowledgeBase initialized with mock data")

    def _build_mock_data(self) -> List[Dict]:
        """Mock 문서 데이터 생성"""
        return [
            # FAQ
            {
                "content": "카페 광고는 따뜻한 분위기와 커피 향을 연상시키는 이미지가 효과적입니다. "
                           "아늑한 조명, 김이 나는 커피잔, 편안한 인테리어를 강조해보세요.",
                "source": "cafe_faq.md",
                "score": 0.92,
                "category": "faq",
                "metadata": {"industry": "cafe", "keywords": ["카페", "커피", "분위기"]}
            },
            {
                "content": "음식점 광고는 신선한 재료와 맛있어 보이는 음식 사진이 중요합니다. "
                           "밝은 조명에서 촬영하고, 색감을 살려주세요.",
                "source": "restaurant_faq.md",
                "score": 0.88,
                "category": "faq",
                "metadata": {"industry": "restaurant", "keywords": ["음식점", "맛집", "음식"]}
            },
            # 생성 가이드
            {
                "content": "이미지 광고 생성 시 스타일 선택이 중요합니다:\n"
                           "- ultra_realistic: 실제 사진처럼 사실적인 이미지\n"
                           "- semi_realistic: 약간의 예술적 터치가 가미된 이미지\n"
                           "- anime: 애니메이션/일러스트 스타일",
                "source": "style_guide.md",
                "score": 0.90,
                "category": "generation_guide",
                "metadata": {"topic": "style", "keywords": ["스타일", "이미지", "생성"]}
            },
            {
                "content": "광고 비율 선택 가이드:\n"
                           "- 1:1 (정사각형): 인스타그램 피드, 프로필 이미지\n"
                           "- 16:9 (가로형): 유튜브 썸네일, 웹 배너\n"
                           "- 9:16 (세로형): 인스타그램/틱톡 스토리, 릴스",
                "source": "aspect_ratio_guide.md",
                "score": 0.87,
                "category": "generation_guide",
                "metadata": {"topic": "aspect_ratio", "keywords": ["비율", "사이즈", "크기"]}
            },
            # 수정 가이드
            {
                "content": "이미지 스타일 변경: '애니메이션 스타일로', '사실적으로' 등의 "
                           "요청으로 스타일을 변경할 수 있습니다.",
                "source": "modification_style.md",
                "score": 0.89,
                "category": "modification_guide",
                "metadata": {"topic": "style_change", "keywords": ["스타일", "변경", "수정"]}
            },
            # 업종별 팁
            {
                "content": "카페 업종 광고 팁:\n"
                           "- 시그니처 메뉴 강조\n"
                           "- 아늑한 공간 연출\n"
                           "- 계절 한정 메뉴 홍보\n"
                           "- 테이크아웃/배달 서비스 안내",
                "source": "cafe_tips.md",
                "score": 0.91,
                "category": "industry_tips",
                "metadata": {"industry": "cafe", "keywords": ["카페", "커피숍", "팁"]}
            },
        ]

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict]:
        """Mock 검색 구현 (키워드 매칭 기반)"""
        logger.info(f"MockKnowledgeBase.search: query='{query}', category={category}, limit={limit}")

        results = []
        query_lower = query.lower()

        for doc in self._mock_data:
            if category and doc["category"] != category:
                continue

            score = 0.0
            content_lower = doc["content"].lower()
            keywords = doc.get("metadata", {}).get("keywords", [])

            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    score += 0.3
                if any(word in kw.lower() for kw in keywords):
                    score += 0.4

            score = min(score + doc["score"] * 0.3, 1.0)

            if score > 0.2:
                results.append({**doc, "score": round(score, 2)})

        results.sort(key=lambda x: x["score"], reverse=True)

        if not results:
            logger.warning(f"MockKnowledgeBase.search: no results, returning top docs")
            results = sorted(self._mock_data, key=lambda x: x["score"], reverse=True)

        final_results = results[:limit]
        logger.info(f"MockKnowledgeBase.search: returning {len(final_results)} results")

        return final_results


# =====================================================
# 싱글톤 인스턴스
# =====================================================

_knowledge_base_instance: Optional[ConsultingKnowledgeBase] = None


def get_knowledge_base() -> ConsultingKnowledgeBase:
    """지식베이스 인스턴스 반환 (싱글톤)"""
    global _knowledge_base_instance
    if _knowledge_base_instance is None:
        _knowledge_base_instance = MockKnowledgeBase()
        logger.info("KnowledgeBase singleton instance created (MockKnowledgeBase)")
    return _knowledge_base_instance
