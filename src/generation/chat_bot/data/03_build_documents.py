"""
문서 빌더 - 정제 강화 & 구분 신호 강화

주요 기능:
- 노이즈 제거 강화(동영상/알림/참여 수치 등)
- 제목 변형 + 지역 확장 + 주소 단서 + 전화 끝자리로 구분력 강화
- 특징 키워드/후기 요약은 깨끗하게만 포함

사용법:
  python 03_build_documents.py

환경변수(옵션):
  CORE_PATH   : 입력 코어 레코드 JSON (default: processed/core_records.json)
  OUT_DIR     : 출력 디렉토리 (default: processed)
  DOCS_JSONL  : 출력 파일 (default: processed/documents.jsonl)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List

# --------------------------------------------
# 설정
# --------------------------------------------
CORE_PATH = os.getenv("CORE_PATH", "processed/core_records.json")
OUT_DIR = os.getenv("OUT_DIR", "processed")
DOCS_JSONL = os.getenv("DOCS_JSONL", os.path.join(OUT_DIR, "documents.jsonl"))
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------
# 지역 확장 (수도권 + 주요 광역시)
# --------------------------------------------
LOCATION_EXPANSIONS = {
    # 서울/수도권
    "강남구": "강남구 강남 서울",
    "서초구": "서초구 서초 강남 서울",
    "송파구": "송파구 송파 잠실 서울",
    "마포구": "마포구 마포 홍대 서울",
    "성동구": "성동구 성수동 성수 서울",
    "용산구": "용산구 이태원 한남동 서울",
    "종로구": "종로구 종로 서울",
    "중구": "중구 명동 을지로 서울",
    "영등포구": "영등포구 여의도 서울",
    "강서구": "강서구 화곡 목동 서울",
    "광진구": "광진구 건대 서울",
    "서대문구": "서대문구 신촌 서울",
    "동대문구": "동대문구 청량리 서울",
    "노원구": "노원구 노원 서울",
    "분당": "분당 정자동 판교 경기 성남",
    "판교": "판교 경기 성남",
    "수원": "수원 인계동 영통 경기",
    "일산": "일산 라페스타 경기 고양",
    "부천": "부천 상동 경기",
    "김포": "김포 구래 경기",
    "안양": "안양 평촌 경기",
    "용인": "용인 수지 경기",
    "인천": "인천 송도 청라",
    # 광역시/지방
    "부산": "부산 서면 해운대 광안리",
    "대구": "대구 동성로 수성구",
    "대전": "대전 은행동 둔산동",
    "광주": "광주 상무지구 충장로",
    "울산": "울산 삼산동",
    "창원": "창원 상남동",
    "전주": "전주 객리단길",
    "청주": "청주 성안길",
    "춘천": "춘천 명동",
    "강릉": "강릉 경포 안목항",
    "여수": "여수 이순신광장",
    "포항": "포항 죽도시장",
    "제주시": "제주시 제주도 제주",
    "서귀포시": "서귀포시 서귀포 제주도 제주",
}


# --------------------------------------------
# 유틸
# --------------------------------------------
NOISE_PATTERNS = [
    r"이런 점이 좋았어요[^\\n]*",
    r"안내\\s*\\d+[회명,\\s\\d]*참여",
    r"\\d+회\\d+명\\s*참여",
    r"더보기[^\\\"]*",
    r"알림받기",
    r"동영상\\s*이미지\\s*갯수\\s*\\d+\\+?\\s*동영상",
    r"동영상\\s*/\\s*",
    r"\\d+개\\s*평점\\([^)]*\\)\\s*안내",
    r"방문자\\s*리뷰\\s*[\\d,]+블로그\\s*리뷰\\s*[\\d,]+",
    r"농림축산식품부\\s*제공\\s*안심식당",
    r"블로그\\s*리뷰\\s*[\\d,]+",
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    for pat in NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_category(category: str) -> str:
    if not category:
        return ""
    category = category.replace(">", " ").replace(",", " ")
    category = re.sub(r"\s+", " ", category).strip()
    return category


def expand_location(location: str) -> str:
    for key, expanded in LOCATION_EXPANSIONS.items():
        if key in location:
            return expanded
    return location


def extract_keywords(description: str, limit: int = 5) -> List[str]:
    if not description:
        return []
    description = clean_text(description)
    matches = re.findall(r"\"([^\"]{2,20})\"", description)
    keywords: List[str] = []
    for kw in matches:
        if any(noise in kw for noise in ["안내", "참여", "더보기", "알림"]):
            continue
        kw = kw.strip()
        if kw and kw not in keywords:
            keywords.append(kw)
    return keywords[:limit]


def title_variations(title: str) -> str:
    if not title:
        return ""
    parts = [title]
    compact = title.replace(" ", "")
    if compact and compact not in parts:
        parts.append(compact)
    clean = re.sub(r"[^\w가-힣 ]", " ", title)
    tokens = clean.split()
    if len(tokens) >= 2:
        parts.append(" ".join(tokens))
    if len(title) >= 4:
        parts.append(title[:4])
    return " ".join(parts)


def short_addr(road_address: str, address: str) -> str:
    base = road_address or address
    if not base:
        return ""
    tokens = base.split()
    return " ".join(tokens[-2:]) if len(tokens) >= 2 else base


def phone_tail(phone: str) -> str:
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)
    return digits[-4:] if len(digits) >= 4 else digits


# --------------------------------------------
# 문서 생성
# --------------------------------------------
def build_doc(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    ctx = record.get("context", {}) or {}
    np = record.get("performance_data", {}).get("naver_place", {}) or {}
    bi = record.get("business_info", {}) or {}
    cv = record.get("customer_voice", {}) or {}

    title = (ctx.get("title") or "").strip()
    industry = ctx.get("industry", "")
    location = (ctx.get("location") or "").strip()
    category = normalize_category(np.get("category", ""))
    road_address = np.get("road_address", "")
    address = np.get("address", "")
    phone = np.get("telephone", "") or ctx.get("phone", "")
    review_count = np.get("review_count", 0)
    rating = np.get("rating") or np.get("estimated_rating") or 0

    desc = bi.get("description", "")
    keywords = extract_keywords(desc)

    # 후기 텍스트 정제
    reviews = cv.get("sample_reviews", [])
    clean_reviews: List[str] = []
    for r in reviews[:2]:
        text = r.get("text", "") if isinstance(r, dict) else str(r)
        text = clean_text(text)
        if text and len(text) > 5:
            clean_reviews.append(text)

    # 고유 토큰
    unique_bits: List[str] = []
    tail = phone_tail(phone)
    if tail:
        unique_bits.append(f"TEL:{tail}")

    id_base = f"{title}_{location}_{road_address}_{index}"
    doc_id = hashlib.md5(id_base.encode()).hexdigest()[:12]

    text_parts: List[str] = []
    text_parts.append(title_variations(title))
    text_parts.append(f"{expand_location(location)} {category}")
    if rating:
        text_parts.append(f"평점 {rating}점")
    if review_count:
        text_parts.append(f"리뷰 {review_count}개")
    if keywords:
        text_parts.append(f"특징 {', '.join(keywords)}")
    addr_token = short_addr(road_address, address)
    if addr_token:
        text_parts.append(f"주소 {addr_token}")
    if clean_reviews:
        text_parts.append(f"후기 {' '.join(clean_reviews)[:120]}")
    if unique_bits:
        text_parts.append(" ".join(unique_bits))

    text = " | ".join([t for t in text_parts if t])

    return {
        "doc_id": doc_id,
        "text": text,
        "metadata": {
            "platform": "naver",
            "place_id": doc_id,
            "title": title,
            "industry": industry,
            "location": location,
            "category": category,
            "road_address": road_address,
            "address": address,
            "review_count": review_count,
            "rating": rating,
            "keywords": keywords,
            "phone_tail": tail,
        },
    }


def main() -> None:
    print("=" * 70)
    print("문서 빌더 (정제+구분 강화)")
    print("=" * 70)

    # 1. 데이터 로드
    print(f"\n[1/3] 데이터 로드: {CORE_PATH}")
    with open(CORE_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"레코드 수: {len(records)}개")

    # 2. 문서 생성
    print("\n[2/3] 문서 생성")
    documents: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        doc = build_doc(rec, idx)
        if doc["text"].strip():
            documents.append(doc)
    print(f"생성된 문서: {len(documents)}개")

    # 3. 저장
    print(f"\n[3/3] 저장: {DOCS_JSONL}")
    with open(DOCS_JSONL, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    lengths = [len(d["text"]) for d in documents]
    print("\n" + "=" * 70)
    print("결과")
    print("=" * 70)
    if lengths:
        print(f"총 문서: {len(documents)}개")
        print(f"텍스트 길이: 평균 {sum(lengths)/len(lengths):.0f}자, 최소 {min(lengths)}자, 최대 {max(lengths)}자")
    print("\n샘플 3개:")
    for i in range(min(3, len(documents))):
        print(f"\n--- 문서 {i+1} ---")
        print(documents[i]["text"][:300])

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
