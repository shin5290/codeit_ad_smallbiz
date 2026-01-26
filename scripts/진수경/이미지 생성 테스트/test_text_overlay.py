from PIL import Image, ImageDraw, ImageFont

def wrap_text(draw, text, font, max_width):
    """
    단순 줄바꿈: bbox 폭을 넘지 않도록 공백 기준 줄바꿈
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    cur = words[0]
    for w in words[1:]:
        test = cur + " " + w
        bbox = draw.textbbox((0,0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def fit_font_to_box(draw, text, font_path, box_w, box_h, max_size=120, min_size=14, line_spacing=1.15):
    """
    bbox 안에 들어가도록 폰트 크기 자동으로 내림
    """
    best = None

    for size in range(max_size, min_size-1, -2):
        font = ImageFont.truetype(font_path, size)
        lines = wrap_text(draw, text, font, box_w)

        # 전체 높이 계산
        line_heights = []
        max_line_w = 0
        for line in lines:
            bb = draw.textbbox((0,0), line, font=font)
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            max_line_w = max(max_line_w, w)
            line_heights.append(h)

        total_h = int(sum(line_heights) * line_spacing)
        if max_line_w <= box_w and total_h <= box_h:
            best = (font, lines, total_h)
            break

    return best  # (font, lines, total_h) or None

def draw_text_in_box(
    img: Image.Image,
    bbox,                  # [x0,y0,x1,y1]
    text: str,
    font_path: str,
    fill=(20, 60, 30),
    align="center",
    stroke_width=6,
    stroke_fill=(255,255,255),
    add_bg_box=False,
    bg_fill=(255,255,255,140),   # 반투명 박스
    padding=16,
    max_font_size=120
):
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)

    x0, y0, x1, y1 = bbox
    box_w = (x1 - x0) - 2*padding
    box_h = (y1 - y0) - 2*padding

    # 폰트 크기 자동 맞춤
    fitted = fit_font_to_box(d, text, font_path, box_w, box_h, max_size=max_font_size)
    if fitted is None:
        # 마지막 수단: 작은 폰트 고정
        font = ImageFont.truetype(font_path, 18)
        lines = [text]
        total_h = 18
    else:
        font, lines, total_h = fitted

    # 배경 박스(옵션)
    if add_bg_box:
        d.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=bg_fill)

    # 텍스트 시작 y
    cur_y = y0 + padding + (box_h - total_h) // 2

    for line in lines:
        bb = d.textbbox((0,0), line, font=font)
        lw = bb[2] - bb[0]
        lh = bb[3] - bb[1]

        if align == "center":
            tx = x0 + padding + (box_w - lw)//2
        elif align == "left":
            tx = x0 + padding
        else:  # right
            tx = x1 - padding - lw

        d.text(
            (tx, cur_y),
            line,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill
        )
        cur_y += int(lh * 1.15)

    # 합성
    out = Image.alpha_composite(img, overlay)
    return out.convert("RGB")

if __name__ == "__main__":
    from src.utils.config import PROJECT_ROOT
    from .test_controlnet_bbox import _resize_if_needed
    from .test_controlnet_bbox import make_layout_control_image
    from .test_controlnet_bbox import default_boxes_for_this_template
    from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops, UnidentifiedImageError, ImageEnhance, ImageColor, ImageSequence
    from pathlib import Path
    from typing import Literal
    import os, sys
    import numpy as np
    # 2단계에서 만든 배경 이미지 불러오기 (bg_locked.png)
    base = Image.open(f"{PROJECT_ROOT}/src/generation/image_generation/test_images/bg_locked.png").convert("RGB")
    boxes = default_boxes_for_this_template(base)

    # 폰트 경로
    FONT_DIR = f"{PROJECT_ROOT}/data/assets/font/"
    FONT_TITLE = f"{FONT_DIR}Cafe24Ssurround-v2.0.ttf"
    FONT_DATE  = f"{FONT_DIR}NotoSansKR-Medium.ttf"
    FONT_MSG   = f"{FONT_DIR}NanumPenScript-Regular.ttf"  # or SsurroundAir

    # 텍스트 내용
    top_msg_text = "잠시! 휴가 다녀오겠습니다!"
    title_text   = "휴가 일정 안내"
    date_text    = "2026.02.01 ~ 2026.02.05"

    # bbox 매핑
    box_map = {b["name"]: b["bbox"] for b in boxes}

    out = base.convert("RGB")

    # 상단 멘트(손글씨 느낌)
    out = draw_text_in_box(
        out,
        box_map["top_msg"],
        top_msg_text,
        FONT_MSG,
        fill=(20, 70, 40),
        stroke_width=4,
        stroke_fill=(255,255,255),
        add_bg_box=False,
        max_font_size=64
    )

    # 제목(굵은 헤드라인)
    out = draw_text_in_box(
        out,
        box_map["title"],
        title_text,
        FONT_TITLE,
        fill=(15, 60, 25),
        stroke_width=8,
        stroke_fill=(255,255,255),
        add_bg_box=False,
        max_font_size=120
    )

    # 날짜(가독성 우선)
    out = draw_text_in_box(
        out,
        box_map["date"],
        date_text,
        FONT_DATE,
        fill=(20, 70, 40),
        stroke_width=5,
        stroke_fill=(255,255,255),
        add_bg_box=True,                 # 날짜는 반투명 박스 깔면 안정적
        bg_fill=(255,255,255,120),
        max_font_size=64
    )

    out.save("final_poster.png")
    print("saved: final_poster.png")
