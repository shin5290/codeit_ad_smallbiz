import re
from typing import Iterable, Optional

_UTENSIL_MARKERS = (
    "수저 추가",
    "수저 넣",
    "수저 올려",
    "수저 그려",
    "숟가락",
    "스푼",
    "젓가락",
)

_OVERLAY_MARKERS = (
    "이미지에", "이미지 위", "이미지위", "사진에", "그림에", "포스터에", "배너에",
    "사진 위", "그림 위", "포스터 위", "배너 위",
)

_IMAGE_KEYWORDS = (
    "이미지", "사진", "그림", "포스터", "배너", "썸네일", "일러스트",
    "image", "photo", "picture", "poster", "banner", "thumbnail", "illustration",
)

_TEXT_KEYWORDS = (
    "문구", "카피", "슬로건", "텍스트", "문장",
    "copy", "slogan", "text", "caption",
)

_DUAL_MARKERS = (
    "둘다", "둘 다", "두개", "두 개", "같이", "한꺼번에", "그리고", "및", "and", "랑", "겸",
)

_MODIFICATION_IMAGE_HINTS = (
    "image", "photo", "picture", "background", "color", "font", "text",
    "이미지", "사진", "그림", "배경", "색상", "색깔", "폰트", "글씨",
    "포스터", "배너", "구도", "레이아웃",
)

_INDUSTRY_ALIAS = {
    "restaurant": ("restaurant", "food", "dining", "eatery", "korean_food", "한식", "식당", "음식점"),
    "cafe": ("cafe", "coffee", "카페", "커피"),
    "bakery": ("bakery", "빵집", "베이커리"),
    "dessert": ("dessert", "디저트"),
    "bar": ("bar", "술집", "바"),
    "retail": ("retail", "shop", "store", "소매", "매장", "상점"),
    "service": ("service", "서비스"),
}


def _has_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def normalize_modification_typo(text: str) -> str:
    if not text:
        return text
    if "수저해" not in text:
        return text
    if _has_any(text, _UTENSIL_MARKERS):
        return text
    return text.replace("수저해", "수정해")


def detect_dual_generation_request(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if _has_any(lowered, _OVERLAY_MARKERS):
        return False
    has_image = _has_any(lowered, _IMAGE_KEYWORDS)
    has_text = _has_any(lowered, _TEXT_KEYWORDS)
    if not (has_image and has_text):
        return False
    if _has_any(lowered, _DUAL_MARKERS):
        return True

    proximity_pattern = (
        r"(이미지|사진|그림|포스터|배너|썸네일|image|photo|picture|poster|banner|thumbnail)"
        r".{0,12}"
        r"(문구|카피|슬로건|텍스트|문장|copy|slogan|text|caption)"
    )
    reverse_pattern = (
        r"(문구|카피|슬로건|텍스트|문장|copy|slogan|text|caption)"
        r".{0,12}"
        r"(이미지|사진|그림|포스터|배너|썸네일|image|photo|picture|poster|banner|thumbnail)"
    )
    return bool(re.search(proximity_pattern, lowered) or re.search(reverse_pattern, lowered))


def should_force_image_modification(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return _has_any(lowered, _MODIFICATION_IMAGE_HINTS)


def normalize_industry_label(label: Optional[str]) -> Optional[str]:
    if not isinstance(label, str):
        return None
    cleaned = label.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in ("none", "null", "unknown"):
        return None
    for canonical, aliases in _INDUSTRY_ALIAS.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    return cleaned


def resolve_hashtag_count_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"해시\\s*태그\\s*(\\d+)\\s*개|해시태그\\s*(\\d+)\\s*개|해시태그\\s*(\\d+)", text)
    if not match:
        match = re.search(r"(\\d+)\\s*개\\s*(?:이상)?\\s*해시태그", text)
        if not match:
            return None
    for group in match.groups():
        if group and group.isdigit():
            return int(group)
    return None


def normalize_hashtag_placeholder(
    refined_input: str,
    chat_history: list[dict],
    default_count: int = 5,
) -> str:
    if not refined_input:
        return refined_input
    if "해시태그" not in refined_input:
        return refined_input
    placeholder_pattern = r"해시태그\\s*[Nn]\\s*개|해시태그\\s*N개|해시태그\\s*n개"
    if not re.search(placeholder_pattern, refined_input):
        return refined_input

    count = None
    for msg in reversed(chat_history or []):
        count = resolve_hashtag_count_from_text(msg.get("content") or "")
        if count:
            break
    if not count:
        count = default_count

    return re.sub(placeholder_pattern, f"해시태그 {count}개", refined_input)


def _extract_recent_hashtag_requirements(chat_history: list[dict]) -> tuple[Optional[int], list[str]]:
    if not chat_history:
        return None, []
    for msg in reversed(chat_history):
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        count = resolve_hashtag_count_from_text(text)
        tags = re.findall(r"#[0-9A-Za-z가-힣_]+", text)
        if not count and not tags:
            continue
        deduped: list[str] = []
        seen = set()
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        return count, deduped
    return None, []


def _user_disallowed_hashtags(text: str) -> bool:
    if not text:
        return False
    lowered = text.replace(" ", "")
    if "해시태그" not in lowered and "#" not in lowered:
        return False
    return any(token in lowered for token in ("없", "빼", "제외", "말고"))


def apply_hashtag_requirements(
    refined_input: str,
    latest_user_message: str,
    chat_history: list[dict],
) -> str:
    if not refined_input:
        return refined_input
    if _user_disallowed_hashtags(latest_user_message):
        return refined_input

    history_count, history_tags = _extract_recent_hashtag_requirements(chat_history)
    if not history_count and not history_tags:
        return refined_input

    explicit_count = resolve_hashtag_count_from_text(latest_user_message or "")
    explicit_tags = re.findall(r"#[0-9A-Za-z가-힣_]+", latest_user_message or "")

    updated = refined_input
    if history_count and not explicit_count:
        if re.search(r"해시태그\\s*\\d+\\s*개", updated):
            updated = re.sub(
                r"해시태그\\s*\\d+\\s*개",
                f"해시태그 {history_count}개",
                updated,
            )
        elif "해시태그" in updated:
            updated = f"{updated} 해시태그 {history_count}개 포함"

    if history_tags and not explicit_tags:
        if not re.search(r"#[0-9A-Za-z가-힣_]+", updated):
            updated = f"{updated} 필수 해시태그: {' '.join(history_tags)}"

    return updated
