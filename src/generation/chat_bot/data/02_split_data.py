import os
import json
import hashlib
from typing import Any, Dict, List, Tuple

# =========================
# 설정
# =========================
RAW_PATH = os.getenv("RAW_PATH", "data/raw/naver_complete_20260116_142140.json")
OUT_CORE = os.getenv("OUT_CORE", "data/processed/core_records.json")
OUT_AUX = os.getenv("OUT_AUX", "data/processed/aux_records.json")

MIN_REVIEW_COUNT = int(os.getenv("MIN_REVIEW_COUNT", "500"))

def norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()

def make_dedup_key(rec: Dict[str, Any]) -> str:
    ctx = rec.get("context", {}) or {}
    np = rec.get("performance_data", {}).get("naver_place", {}) or {}

    title = norm(ctx.get("title")).lower()
    road = norm(np.get("road_address")).lower()
    addr = norm(np.get("address")).lower()
    loc = norm(ctx.get("location")).lower()

    # road가 없으면 address 사용, 둘 다 없으면 location까지라도 넣기
    base = f"{title}||{road or addr}||{loc}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def record_quality_score(rec: Dict[str, Any]) -> int:
    # 중복 충돌 시 "더 완성도 높은" 레코드 선택용 점수
    score = 0
    np = rec.get("performance_data", {}).get("naver_place", {}) or {}
    bi = rec.get("business_info", {}) or {}
    cv = rec.get("customer_voice", {}) or {}
    va = rec.get("visual_assets", {}) or {}

    if np.get("review_count") is not None: score += 2
    if np.get("rating") is not None or np.get("estimated_rating") is not None: score += 2
    if norm(bi.get("description")): score += 1
    if (cv.get("sample_reviews") or []): score += 1
    if (va.get("business_photos") or []): score += 1
    return score

def is_core(rec: Dict[str, Any]) -> bool:
    np = rec.get("performance_data", {}).get("naver_place", {}) or {}
    review_count = np.get("review_count") or 0
    rating = np.get("rating")
    est = np.get("estimated_rating")
    return (review_count >= MIN_REVIEW_COUNT) and (rating is not None or est is not None)

def safe_mkdir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def main():
    # 1) Load
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError("RAW_PATH JSON은 리스트(list) 형태여야 합니다.")

    print(f"✅ Loaded raw records: {len(data)}")
    print(f"📌 RAW_PATH: {RAW_PATH}")

    # 2) Dedup
    best = {}  # key -> (score, rec)
    for rec in data:
        k = make_dedup_key(rec)
        sc = record_quality_score(rec)
        if k not in best or sc > best[k][0]:
            best[k] = (sc, rec)

    deduped = [v[1] for v in best.values()]
    print(f"🧹 Deduped records: {len(deduped)} (removed {len(data) - len(deduped)})")

    # 3) Split
    core = []
    aux = []
    for rec in deduped:
        if is_core(rec):
            core.append(rec)
        else:
            aux.append(rec)

    print(f"🔥 Core records (review_count>={MIN_REVIEW_COUNT} & rating/est exists): {len(core)}")
    print(f"🧊 Aux records: {len(aux)}")

    # 4) Basic stats (optional but helpful)
    def count_field(records: List[Dict[str, Any]], path: Tuple[str, ...]) -> int:
        c = 0
        for r in records:
            cur = r
            ok = True
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if ok and cur:
                c += 1
        return c

    # 사진/리뷰/설명 커버리지 확인
    core_desc = count_field(core, ("business_info", "description"))
    core_reviews = 0
    for r in core:
        arr = (r.get("customer_voice", {}) or {}).get("sample_reviews") or []
        if len(arr) > 0:
            core_reviews += 1
    core_photos = 0
    for r in core:
        arr = (r.get("visual_assets", {}) or {}).get("business_photos") or []
        if len(arr) > 0:
            core_photos += 1

    print("\n📊 Core coverage:")
    if core:
        print(f"  - description: {core_desc}/{len(core)} ({core_desc/len(core)*100:.1f}%)")
        print(f"  - sample_reviews: {core_reviews}/{len(core)} ({core_reviews/len(core)*100:.1f}%)")
        print(f"  - business_photos: {core_photos}/{len(core)} ({core_photos/len(core)*100:.1f}%)")

    # 5) Save
    safe_mkdir(OUT_CORE)
    safe_mkdir(OUT_AUX)

    with open(OUT_CORE, "w", encoding="utf-8") as f:
        json.dump(core, f, ensure_ascii=False, indent=2)

    with open(OUT_AUX, "w", encoding="utf-8") as f:
        json.dump(aux, f, ensure_ascii=False, indent=2)

    print("\n💾 Saved:")
    print(f"  - {OUT_CORE}")
    print(f"  - {OUT_AUX}")

if __name__ == "__main__":
    main()
