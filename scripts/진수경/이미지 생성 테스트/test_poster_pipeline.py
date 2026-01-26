"""
포스터 생성 파이프라인 테스트
1) test_poster_prompt -> 2) test_controlnet_bbox -> 3) test_text_overlay
"""

import sys
from pathlib import Path

from PIL import Image

# 프로젝트 루트를 sys.path에 추가 (import 가능하게)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generation.image_generation.test_poster_prompt import (
    OUTPUT_DIR as POSTER_OUTPUT_DIR,
    test_poster_prompt,
)
from src.generation.image_generation.test_controlnet_bbox import (
    default_boxes_for_this_template,
    run_sdxl_controlnet_lock_text_areas,
)
from src.generation.image_generation.test_text_overlay import draw_text_in_box


OUTPUT_DIR = Path(__file__).parent / "test_images"
OUTPUT_DIR.mkdir(exist_ok=True)

FONT_DIR = project_root / "data/assets/font"
FONT_TITLE = FONT_DIR / "Cafe24Ssurround-v2.0.ttf"
FONT_DATE = FONT_DIR / "NotoSansKR-Medium.ttf"
FONT_MSG = FONT_DIR / "NanumPenScript-Regular.ttf"


def run_poster_pipeline() -> bool:
    """
    1) SDXL 프롬프트 이미지 생성
    2) ControlNet으로 텍스트 영역 고정
    3) 텍스트 오버레이
    """
    print("\n" + "=" * 60)
    print("POSTER PIPELINE")
    print("=" * 60)

    # 1) 프롬프트 이미지 생성
    print("\n[1/3] Poster prompt generation")
    if not test_poster_prompt():
        print("❌ Poster prompt step failed")
        return False

    poster_path = POSTER_OUTPUT_DIR / "test_poster_prompt.png"
    if not poster_path.exists():
        print(f"❌ Poster image not found: {poster_path}")
        return False

    # 2) ControlNet으로 텍스트 영역 고정
    print("\n[2/3] ControlNet lock text areas")
    base_img = Image.open(poster_path).convert("RGB")
    boxes = default_boxes_for_this_template(base_img)
    bg_locked_path = OUTPUT_DIR / "bg_locked.png"

    run_sdxl_controlnet_lock_text_areas(
        init_image_path=str(poster_path),
        out_path=str(bg_locked_path),
        boxes=boxes,
        strength=0.15,
        cfg=5.5,
        steps=25,
        seed=1000,
        max_side=None,  # box 좌표 불일치 방지
    )

    # 3) 텍스트 오버레이
    print("\n[3/3] Text overlay")
    out = Image.open(bg_locked_path).convert("RGB")
    box_map = {b["name"]: b["bbox"] for b in boxes}

    top_msg_text = "잠시! 휴가 다녀오겠습니다!"
    title_text = "휴가 일정 안내"
    date_text = "2026.02.01 ~ 2026.02.05"

    out = draw_text_in_box(
        out,
        box_map["top_msg"],
        top_msg_text,
        str(FONT_MSG),
        fill=(20, 70, 40),
        stroke_width=4,
        stroke_fill=(255, 255, 255),
        add_bg_box=False,
        max_font_size=64,
    )

    out = draw_text_in_box(
        out,
        box_map["title"],
        title_text,
        str(FONT_TITLE),
        fill=(15, 60, 25),
        stroke_width=8,
        stroke_fill=(255, 255, 255),
        add_bg_box=False,
        max_font_size=120,
    )

    out = draw_text_in_box(
        out,
        box_map["date"],
        date_text,
        str(FONT_DATE),
        fill=(20, 70, 40),
        stroke_width=5,
        stroke_fill=(255, 255, 255),
        add_bg_box=True,
        bg_fill=(255, 255, 255, 120),
        max_font_size=64,
    )

    final_path = OUTPUT_DIR / "final_poster_pipeline.png"
    out.save(final_path)
    print(f"\n✅ Final poster saved: {final_path}")
    return True


if __name__ == "__main__":
    ok = run_poster_pipeline()
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n{status}: Poster Pipeline")
