import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ImageReference:
    role: Optional[str] = None  # "input", "output"
    position: Optional[str] = None  # "first", "last", "previous", "nth"
    index: Optional[int] = None


# 정규식 사전 컴파일
FIRST_PATTERN = re.compile(r"(첫번째|첫 번째|1번째|1번|첫 이미지|첫 사진|처음)")
SECOND_PATTERN = re.compile(r"(두번째|두 번째|2번째|2번)")
THIRD_PATTERN = re.compile(r"(세번째|세 번째|3번째|3번)")
NTH_PATTERN = re.compile(r"(\d+)\s*(?:번째|번)")


def contains_any(text: str, keywords: List[str]) -> bool:
    """키워드 리스트 중 하나라도 텍스트에 포함되는지 확인"""
    return any(keyword in text for keyword in keywords)


def detect_content_type(text: str) -> str:
    """사용자 입력에서 요청하는 콘텐츠 타입을 감지"""
    if not text.strip():
        return "both"
    
    both_keywords = [
        "둘 다", "둘다", "이미지와 문구", "이미지랑 문구", "이미지+텍스트", 
        "이미지+문구", "사진과 문구", "사진+문구", "이미지와 카피", 
        "이미지랑 카피", "전부", "모두",
    ]
    if contains_any(text, both_keywords):
        return "both"
    
    text_only_keywords = ["카피만", "문구만", "텍스트만", "설명만", "소개만", "슬로건만"]
    image_only_keywords = ["이미지만", "사진만", "포스터만", "배너만", "썸네일만"]
    
    if contains_any(text, text_only_keywords):
        return "text"
    if contains_any(text, image_only_keywords):
        return "image"
    
    text_keywords = [
        "카피", "문구", "텍스트", "설명", "소개", "슬로건", 
        "캐치프레이즈", "문장", "광고 문구", "헤드라인"
    ]
    image_keywords = [
        "이미지", "사진", "포스터", "배너", "썸네일", "디자인", 
        "그림", "일러스트", "그래픽", "홍보물", "전단", "전단지", "비주얼"
    ]

    wants_text = contains_any(text, text_keywords)
    wants_image = contains_any(text, image_keywords)

    if wants_text and wants_image:
        return "both"
    if wants_text:
        return "text"
    if wants_image:
        return "image"
    
    return "both"


def detect_image_references(text: str) -> List[ImageReference]:
    """텍스트에서 이미지 참조 정보를 여러 개 추출"""
    image_tokens = ["이미지", "사진", "그림", "샘플", "결과", "생성물", "비주얼"]

    role_input_explicit = [
        "내가 보낸", "내가 올린", "내가 첨부", "내 사진", "내 이미지",
        "제가 보낸", "제가 올린"
    ]
    role_output_explicit = [
        "너가 보낸", "네가 보낸", "당신이 보낸", "ai가 보낸",
        "봇이 보낸", "시스템이 보낸", "claude가", "생성해준"
    ]

    role_input_tokens = [
        "보내준", "보낸", "첨부", "올린", "원본", "처음에 보낸",
        "처음 보낸", "맨처음에 보낸", "이전에 보낸", "입력"
    ]
    role_output_tokens = [
        "생성된", "만든", "결과", "출력", "생성물", "만들어준", "그려진"
    ]

    role = None
    if contains_any(text, role_input_explicit):
        role = "input"
    elif contains_any(text, role_output_explicit):
        role = "output"
    elif contains_any(text, role_input_tokens):
        role = "input"
    elif contains_any(text, role_output_tokens):
        role = "output"

    positions: list[tuple[str, Optional[int]]] = []

    if FIRST_PATTERN.search(text):
        positions.append(("first", 1))
    if SECOND_PATTERN.search(text):
        positions.append(("nth", 2))
    if THIRD_PATTERN.search(text):
        positions.append(("nth", 3))

    for num_match in NTH_PATTERN.finditer(text):
        index = int(num_match.group(1))
        positions.append(("nth" if index > 1 else "first", index))

    if contains_any(text, ["마지막", "끝", "최종", "라스트", "최근"]):
        positions.append(("last", None))
    if contains_any(text, ["이전", "직전", "바로 전", "그 이미지", "그 사진", "아까", "방금"]):
        positions.append(("previous", None))

    has_image_context = (
        contains_any(text, image_tokens) or
        contains_any(text, role_input_explicit + role_output_explicit)
    )
    if not has_image_context:
        return []

    if not positions:
        if role is None:
            return []
        return [ImageReference(role=role, position=None, index=None)]

    deduped = []
    seen = set()
    for position, index in positions:
        key = (position, index, role)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ImageReference(role=role, position=position, index=index))

    return deduped
