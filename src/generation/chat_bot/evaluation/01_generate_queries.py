"""
평가 데이터셋 v5.9 생성기 (다중 정답, 상담 챗봇 의도 반영)

목적:
- 상담 챗봇의 실제 의도 반영 (지역 기반, 규모 기반 등)
- 다중 정답 허용 (relevance 점수: 0-2)
- 쉬운 난이도 (명시적 질문)

의도 분포:
1. 지역 기반 참고 사례 (30%)
2. 규모 기반 참고 사례 (25%)
3. 채널별 전략 사례 (20%)
4. 문제 해결 사례 (15%)
5. 업종 전반 트렌드 (5%)
6. 복합 조건 사례 (5%)

실행:
  python evaluation/generate_queries_v59.py

출력:
  evaluation/results/synthetic_queries_v59.json
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# --------------------------------------------
# 설정
# --------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "documents_v5.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "synthetic_queries_v59.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 의도별 생성 개수 (총 100개)
INTENT_COUNTS = {
    "location_based": 30,      # 30%
    "scale_based": 25,         # 25%
    "channel_strategy": 20,    # 20%
    "problem_solving": 15,     # 15%
    "industry_trend": 5,       # 5%
    "complex_condition": 5,    # 5%
}

TOTAL = sum(INTENT_COUNTS.values())

# --------------------------------------------
# 질문 템플릿
# --------------------------------------------

TEMPLATES = {
    "location_based": [
        "{location}에서 {industry} 운영 중인데, 같은 지역 {industry_kr} 성공 사례 참고하고 싶어요.",
        "{location} {industry_kr}인데 주변 비슷한 가게들 프로모션 어떻게 하는지 궁금해요.",
        "{location}에 있는 {industry_kr} 마케팅 잘하는 곳들 전략 알려주세요.",
        "{location}에서 {industry} 하고 있는데 같은 동네 성공 사례 보고 싶어요.",
    ],
    "scale_based": [
        "리뷰 {review_range} 정도 있는 {industry_kr}인데, 비슷한 규모 마케팅 전략 알려주세요.",
        "{industry_kr} 운영 중이고 리뷰가 {review_range}쯤 되는데, 비슷한 곳들 어떻게 하는지 궁금해요.",
        "평점 {rating_range}대 {industry_kr}입니다. 비슷한 상황 가게들 참고하고 싶어요.",
        "리뷰 {review_range} 수준 {industry_kr}인데 같은 규모 성공 사례 보여주세요.",
    ],
    "channel_strategy": [
        "{industry_kr} 운영하는데 인스타그램 사진 게시는 어떻게 해야 할지 다른 곳 사례 보고 싶어요.",
        "{industry_kr}인데 네이버 플레이스 사진 잘 올리는 가게들 참고하고 싶습니다.",
        "인스타 마케팅 잘하는 {industry_kr} 사례 알려주세요.",
        "{industry_kr} 사진 마케팅 잘하는 곳들은 어떻게 하나요?",
    ],
    "problem_solving": [
        "평점이 {rating_range}대인 {location} {industry_kr}인데, 평점 올린 비슷한 사례 있을까요?",
        "{location}에서 {industry_kr} 하는데 리뷰가 적어서 고민이에요. 비슷한 상황 극복한 사례 알려주세요.",
        "리뷰 {review_range} {industry_kr}인데 더 늘리는 방법 성공 사례로 보여주세요.",
        "평점 {rating_range}대 {industry_kr}입니다. 개선 사례 참고하고 싶어요.",
    ],
    "industry_trend": [
        "{industry_kr}들은 요즘 어떤 마케팅을 많이 하나요?",
        "{industry_kr} 운영하는데 요즘 트렌드가 뭔지 궁금해요.",
        "{industry_kr} 마케팅 잘하는 곳들 사례 여러 개 보여주세요.",
        "{industry_kr} 성공 사례들 전반적으로 알고 싶어요.",
    ],
    "complex_condition": [
        "{location}에서 {industry_kr} 하는데 리뷰 {review_range} 이상인 곳들 어떻게 운영하는지 궁금해요.",
        "{location} {industry_kr} 중에 평점 {rating_range}대이면서 리뷰 많은 곳들 전략 알려주세요.",
        "리뷰 {review_range} 이상 {location} {industry_kr} 마케팅 사례 보고 싶습니다.",
        "{location}에 있는 평점 높은 {industry_kr}들 성공 비결 궁금해요.",
    ],
}

# 업종 표현
INDUSTRY_DISPLAY = {
    "cafe": ["카페", "커피숍"],
    "restaurant": ["식당", "음식점", "맛집"],
}


# --------------------------------------------
# 문서 로드
# --------------------------------------------
def load_documents():
    """documents_v5.jsonl 로드"""
    docs = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            doc = json.loads(line.strip())
            meta = doc.get("metadata", {})
            docs.append({
                "doc_id": doc.get("doc_id"),
                "doc_index": idx,
                "title": meta.get("title", ""),
                "location": meta.get("location", ""),
                "industry": meta.get("industry", ""),
                "review_count": meta.get("review_count", 0),
                "rating": meta.get("rating", 0.0),
            })
    return docs


# --------------------------------------------
# 쿼리 생성 함수들
# --------------------------------------------

def generate_location_based(docs, loc_ind_groups):
    """의도 1: 지역 기반 참고 사례"""
    # 충분한 문서가 있는 (location, industry) 조합 선택
    valid_groups = [(loc, ind, group) for (loc, ind), group in loc_ind_groups.items() if len(group) >= 5]
    if not valid_groups:
        return None

    loc, ind, group = random.choice(valid_groups)
    industry_kr = random.choice(INDUSTRY_DISPLAY.get(ind, ["가게"]))
    template = random.choice(TEMPLATES["location_based"])

    query = template.format(
        location=loc,
        industry=ind,
        industry_kr=industry_kr,
    )

    # 정답: 같은 location + industry (모두 relevance=2)
    relevant_docs = [
        {"doc_id": d["doc_id"], "relevance": 2}
        for d in group
    ]

    return {
        "query": query,
        "intent": "location_based",
        "relevant_docs": relevant_docs,
        "filters": {
            "location": loc,
            "industry": ind,
        },
    }


def generate_scale_based(docs):
    """의도 2: 규모 기반 참고 사례"""
    # 기준 문서 하나 선택
    base_doc = random.choice([d for d in docs if d["review_count"] > 100])
    review_count = base_doc["review_count"]
    rating = base_doc["rating"]
    industry = base_doc["industry"]
    industry_kr = random.choice(INDUSTRY_DISPLAY.get(industry, ["가게"]))

    # 리뷰 범위 또는 평점 범위로 쿼리 생성
    if random.random() < 0.7:  # 70% 리뷰 기반
        # 리뷰 범위 표현
        if review_count < 500:
            review_range = "100-500개"
            min_r, max_r = 100, 500
        elif review_count < 1000:
            review_range = "500-1000개"
            min_r, max_r = 500, 1000
        elif review_count < 3000:
            review_range = "1000-3000개"
            min_r, max_r = 1000, 3000
        else:
            review_range = "3000개 이상"
            min_r, max_r = 3000, 100000

        template = random.choice([t for t in TEMPLATES["scale_based"] if "review_range" in t])
        query = template.format(industry_kr=industry_kr, review_range=review_range)

        # 정답: 같은 업종 + 비슷한 리뷰 범위
        same_industry_similar = [
            {"doc_id": d["doc_id"], "relevance": 2}
            for d in docs
            if d["industry"] == industry and min_r <= d["review_count"] <= max_r
        ]

        # 다른 업종이지만 비슷한 규모 (relevance=1)
        other_industry_similar = [
            {"doc_id": d["doc_id"], "relevance": 1}
            for d in docs
            if d["industry"] != industry and min_r <= d["review_count"] <= max_r
        ][:10]  # 최대 10개

        relevant_docs = same_industry_similar + other_industry_similar

        return {
            "query": query,
            "intent": "scale_based",
            "relevant_docs": relevant_docs,
            "filters": {
                "industry": industry,
                "review_count_range": [min_r, max_r],
            },
        }

    else:  # 30% 평점 기반
        # 평점 범위
        if rating < 3.5:
            rating_range = "3점"
            min_r, max_r = 0, 3.5
        elif rating < 4.0:
            rating_range = "3-4점"
            min_r, max_r = 3.0, 4.0
        elif rating < 4.5:
            rating_range = "4점"
            min_r, max_r = 4.0, 4.5
        else:
            rating_range = "4.5점 이상"
            min_r, max_r = 4.5, 5.0

        template = random.choice([t for t in TEMPLATES["scale_based"] if "rating_range" in t])
        query = template.format(industry_kr=industry_kr, rating_range=rating_range)

        # 정답: 같은 업종 + 비슷한 평점
        relevant_docs = [
            {"doc_id": d["doc_id"], "relevance": 2}
            for d in docs
            if d["industry"] == industry and min_r <= d["rating"] <= max_r
        ]

        return {
            "query": query,
            "intent": "scale_based",
            "relevant_docs": relevant_docs,
            "filters": {
                "industry": industry,
                "rating_range": [min_r, max_r],
            },
        }


def generate_channel_strategy(docs):
    """의도 3: 채널별 전략 사례"""
    industry = random.choice(["cafe", "restaurant"])
    industry_kr = random.choice(INDUSTRY_DISPLAY[industry])
    template = random.choice(TEMPLATES["channel_strategy"])

    query = template.format(industry_kr=industry_kr)

    # 정답: 같은 업종 모두 (사진/채널 전략은 업종별로 유사)
    relevant_docs = [
        {"doc_id": d["doc_id"], "relevance": 2}
        for d in docs
        if d["industry"] == industry
    ]

    return {
        "query": query,
        "intent": "channel_strategy",
        "relevant_docs": relevant_docs,
        "filters": {
            "industry": industry,
        },
    }


def generate_problem_solving(docs, loc_ind_groups):
    """의도 4: 문제 해결 사례"""
    # 평점 낮거나 리뷰 적은 케이스
    if random.random() < 0.6:  # 60% 평점 문제
        valid_groups = [(loc, ind, group) for (loc, ind), group in loc_ind_groups.items() if len(group) >= 3]
        if not valid_groups:
            return None

        loc, ind, group = random.choice(valid_groups)
        industry_kr = random.choice(INDUSTRY_DISPLAY.get(ind, ["가게"]))

        # 평점 범위
        rating_range = random.choice(["3점", "3-4점", "4점"])
        if rating_range == "3점":
            min_r, max_r = 0, 3.5
        elif rating_range == "3-4점":
            min_r, max_r = 3.0, 4.0
        else:
            min_r, max_r = 4.0, 4.5

        template = random.choice([t for t in TEMPLATES["problem_solving"] if "rating_range" in t])
        query = template.format(location=loc, industry_kr=industry_kr, rating_range=rating_range)

        # 정답: 같은 지역 + 업종 + 비슷한 평점
        relevant_docs = [
            {"doc_id": d["doc_id"], "relevance": 2}
            for d in group
            if min_r <= d["rating"] <= max_r
        ]

        return {
            "query": query,
            "intent": "problem_solving",
            "relevant_docs": relevant_docs,
            "filters": {
                "location": loc,
                "industry": ind,
                "rating_range": [min_r, max_r],
            },
        }

    else:  # 40% 리뷰 적음
        loc, ind, group = random.choice([(loc, ind, group) for (loc, ind), group in loc_ind_groups.items() if len(group) >= 3])
        industry_kr = random.choice(INDUSTRY_DISPLAY.get(ind, ["가게"]))

        review_range = random.choice(["100개 미만", "100-500개"])
        min_r, max_r = (0, 100) if "미만" in review_range else (100, 500)

        template = random.choice([t for t in TEMPLATES["problem_solving"] if "review_range" in t or "리뷰가 적" in t])
        query = template.format(location=loc, industry_kr=industry_kr, review_range=review_range)

        # 정답: 같은 지역 + 업종 + 리뷰 적은 곳
        relevant_docs = [
            {"doc_id": d["doc_id"], "relevance": 2}
            for d in group
            if min_r <= d["review_count"] <= max_r
        ]

        return {
            "query": query,
            "intent": "problem_solving",
            "relevant_docs": relevant_docs,
            "filters": {
                "location": loc,
                "industry": ind,
                "review_count_range": [min_r, max_r],
            },
        }


def generate_industry_trend(docs):
    """의도 5: 업종 전반 트렌드"""
    industry = random.choice(["cafe", "restaurant"])
    industry_kr = random.choice(INDUSTRY_DISPLAY[industry])
    template = random.choice(TEMPLATES["industry_trend"])

    query = template.format(industry_kr=industry_kr)

    # 정답: 해당 업종 모두
    relevant_docs = [
        {"doc_id": d["doc_id"], "relevance": 2}
        for d in docs
        if d["industry"] == industry
    ]

    return {
        "query": query,
        "intent": "industry_trend",
        "relevant_docs": relevant_docs,
        "filters": {
            "industry": industry,
        },
    }


def generate_complex_condition(docs, loc_ind_groups):
    """의도 6: 복합 조건 사례"""
    valid_groups = [(loc, ind, group) for (loc, ind), group in loc_ind_groups.items() if len(group) >= 5]
    if not valid_groups:
        return None

    loc, ind, group = random.choice(valid_groups)
    industry_kr = random.choice(INDUSTRY_DISPLAY.get(ind, ["가게"]))
    template = random.choice(TEMPLATES["complex_condition"])

    # 리뷰 또는 평점 조건 추가
    if random.random() < 0.7:  # 리뷰
        review_range = random.choice(["1000개", "500개", "2000개"])
        min_r = int(review_range.replace("개", ""))

        query = template.format(
            location=loc,
            industry_kr=industry_kr,
            review_range=review_range,
            rating_range="4점",  # placeholder
        )

        # 정답: location + industry + 리뷰 많음
        relevant_docs = [
            {"doc_id": d["doc_id"], "relevance": 2}
            for d in group
            if d["review_count"] >= min_r
        ]

        return {
            "query": query,
            "intent": "complex_condition",
            "relevant_docs": relevant_docs,
            "filters": {
                "location": loc,
                "industry": ind,
                "review_count_min": min_r,
            },
        }

    else:  # 평점
        rating_range = random.choice(["4점", "4.5점"])
        min_r = 4.0 if rating_range == "4점" else 4.5

        query = template.format(
            location=loc,
            industry_kr=industry_kr,
            review_range="많은",  # placeholder
            rating_range=rating_range,
        )

        # 정답: location + industry + 평점 높음
        relevant_docs = [
            {"doc_id": d["doc_id"], "relevance": 2}
            for d in group
            if d["rating"] >= min_r
        ]

        return {
            "query": query,
            "intent": "complex_condition",
            "relevant_docs": relevant_docs,
            "filters": {
                "location": loc,
                "industry": ind,
                "rating_min": min_r,
            },
        }


# --------------------------------------------
# 메인
# --------------------------------------------
def main():
    random.seed(42)

    # 문서 로드
    docs = load_documents()
    print(f"문서 로드: {len(docs)}개")

    # location + industry 그룹핑
    loc_ind_groups = defaultdict(list)
    for d in docs:
        loc = d.get("location", "")
        ind = d.get("industry", "")
        if not loc or loc == "기타" or not ind:
            continue
        loc_ind_groups[(loc, ind)].append(d)

    print(f"(location, industry) 그룹: {len(loc_ind_groups)}개")

    # 쿼리 생성
    results = []
    intent_counts = {k: 0 for k in INTENT_COUNTS.keys()}

    generators = {
        "location_based": lambda: generate_location_based(docs, loc_ind_groups),
        "scale_based": lambda: generate_scale_based(docs),
        "channel_strategy": lambda: generate_channel_strategy(docs),
        "problem_solving": lambda: generate_problem_solving(docs, loc_ind_groups),
        "industry_trend": lambda: generate_industry_trend(docs),
        "complex_condition": lambda: generate_complex_condition(docs, loc_ind_groups),
    }

    max_attempts = 1000
    attempts = 0

    while len(results) < TOTAL and attempts < max_attempts:
        attempts += 1

        # 아직 목표 개수 안 찬 의도 선택
        available_intents = [
            intent for intent, target in INTENT_COUNTS.items()
            if intent_counts[intent] < target
        ]

        if not available_intents:
            break

        intent = random.choice(available_intents)
        generator = generators[intent]

        item = generator()
        if item and len(item["relevant_docs"]) >= 1:  # 최소 1개 정답 있어야 함
            results.append(item)
            intent_counts[intent] += 1

            if len(results) % 20 == 0:
                print(f"생성: {len(results)}/{TOTAL}")

    print(f"\n총 {len(results)}개 쿼리 생성")
    print(f"의도별 분포: {intent_counts}")

    # 통계
    avg_relevant = sum(len(r["relevant_docs"]) for r in results) / len(results)
    print(f"평균 정답 문서 수: {avg_relevant:.1f}개")

    # 저장
    output = {
        "version": "v5.9",
        "description": "다중 정답, 상담 챗봇 의도 반영 (쉬운 난이도)",
        "total_count": len(results),
        "intent_counts": intent_counts,
        "avg_relevant_docs": round(avg_relevant, 2),
        "queries": results,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {OUTPUT_PATH}")

    # 샘플 출력
    print("\n=== 샘플 (의도별 1개씩) ===")
    for intent in INTENT_COUNTS.keys():
        samples = [r for r in results if r["intent"] == intent]
        if samples:
            s = samples[0]
            print(f"\n[{intent}]")
            print(f"  Query: {s['query']}")
            print(f"  정답 문서: {len(s['relevant_docs'])}개")
            print(f"  필터: {s.get('filters', {})}")


if __name__ == "__main__":
    main()
