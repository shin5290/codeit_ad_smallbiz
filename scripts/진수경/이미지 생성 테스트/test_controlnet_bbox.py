"""
test_controlnet_bbox.py (drop-in replacement)

목표
- SDXL + ControlNet(Img2Img)로 "텍스트 영역(빈 공간)" 레이아웃을 더 안정적으로 고정
- 메모리 터짐 방지:
  - xformers 있으면 사용, 없으면 자동 스킵
  - VAE slicing/tiling
  - attention slicing
  - (옵션) CPU offload / model cpu offload
  - (옵션) 이미지 해상도 리사이즈
"""

import os
import gc
import torch
from PIL import Image, ImageDraw
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from src.utils.config import PROJECT_ROOT


# ----------------------------
# Memory / Environment helpers
# ----------------------------
def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _maybe_enable_xformers(pipe):
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[INFO] xformers enabled")
    except Exception as e:
        print(f"[INFO] xformers not available, skipping ({type(e).__name__})")


def _enable_memory_savers(pipe):
    # diffusers 내장 메모리 절약 옵션들
    try:
        pipe.enable_attention_slicing("auto")
        print("[INFO] attention slicing enabled")
    except Exception:
        pass

    try:
        pipe.enable_vae_slicing()
        print("[INFO] VAE slicing enabled")
    except Exception:
        pass

    # SDXL에서 tiling이 지원되면 더 안정적
    try:
        pipe.enable_vae_tiling()
        print("[INFO] VAE tiling enabled")
    except Exception:
        pass


def _maybe_cpu_offload(pipe, device: str, mode: str = "sequential"):
    """
    mode:
      - "sequential": 가장 메모리 절약(속도 느림)
      - "model": model_cpu_offload (조금 빠름)
      - "none": offload 안 함
    """
    if device != "cuda":
        return

    mode = (mode or "none").lower()
    if mode == "none":
        return

    try:
        if mode == "sequential":
            pipe.enable_sequential_cpu_offload()
            print("[INFO] sequential CPU offload enabled")
        elif mode == "model":
            pipe.enable_model_cpu_offload()
            print("[INFO] model CPU offload enabled")
        else:
            print("[WARN] unknown offload mode, skipping:", mode)
    except Exception as e:
        print(f"[WARN] CPU offload not available, skipping ({type(e).__name__}: {e})")


def _load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resize_if_needed(img: Image.Image, max_side: int | None) -> Image.Image:
    """
    VRAM 터질 때 가장 효과 좋은 방법: 입력 해상도 제한.
    SDXL은 기본적으로 1024 근처가 안정적.
    """
    if not max_side:
        return img

    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img

    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    print(f"[INFO] resize {w}x{h} -> {nw}x{nh} (max_side={max_side})")
    return img.resize((nw, nh), Image.LANCZOS)


# ----------------------------
# Control image (layout guide)
# ----------------------------
def make_layout_control_image(base_img: Image.Image, boxes, line_width: int = 6) -> Image.Image:
    """
    base_img: 원본 이미지 (PIL)
    boxes: [{"name":..., "bbox":[x0,y0,x1,y1]}, ...]  (픽셀좌표)
    return: ControlNet conditioning image (PIL RGB)
    """
    w, h = base_img.size
    ctrl = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(ctrl)

    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=line_width)

    return ctrl


def default_boxes_for_this_template(img: Image.Image):
    w, h = img.size

    notebook = [int(0.18*w), int(0.14*h), int(0.82*w), int(0.78*h)]
    top_msg  = [int(0.22*w), int(0.18*h), int(0.78*w), int(0.28*h)]
    title    = [int(0.22*w), int(0.32*h), int(0.78*w), int(0.50*h)]
    date     = [int(0.22*w), int(0.52*h), int(0.78*w), int(0.62*h)]

    return [
        {"name": "notebook", "bbox": notebook},
        {"name": "top_msg", "bbox": top_msg},
        {"name": "title", "bbox": title},
        {"name": "date", "bbox": date},
    ]


# ----------------------------
# Main: SDXL + ControlNet Img2Img
# ----------------------------
def run_sdxl_controlnet_lock_text_areas(
    init_image_path: str,
    out_path: str,
    boxes,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet_model: str = "/opt/ai-models/sdxl/diffusers--controlnet-canny-sdxl-1.0",
    prompt: str | None = None,
    negative_prompt: str | None = None,
    strength: float = 0.20,
    controlnet_conditioning_scale: float = 1.0,
    steps: int = 25,
    cfg: float = 6.0,
    seed: int = 1234,
    device: str | None = None,
    # ★ 메모리 안정화 옵션
    offload_mode: str = "none",  # "none" | "model" | "sequential"
    max_side: int | None = 1024, # VRAM 부족하면 896/768로 내려
):
    """
    - init 이미지 기반으로 약하게 재합성하면서
    - ControlNet layout guide(사각형)로 텍스트 영역을 '비워두는 레이아웃'을 고정
    """
    device = device or _get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    init_img = _load_image_rgb(init_image_path)
    init_img = _resize_if_needed(init_img, max_side=max_side)

    # boxes는 리사이즈되면 좌표도 같이 스케일해야 함
    # -> max_side로 줄였을 때만 스케일 적용
    # (현재 코드는 "기본 박스"를 init_img 리사이즈 이후에 생성하도록 main에서 처리하는 게 안전)
    # 여기서는 boxes를 이미 맞는 것으로 가정.

    ctrl_img = make_layout_control_image(init_img, boxes)

    _clear_cuda()

    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )

    # offload 사용 여부에 따라 .to(cuda) 타이밍 다름
    if offload_mode == "none":
        pipe = pipe.to(device)

    # 메모리 절약 옵션
    _maybe_enable_xformers(pipe)      # 있으면 켜고 없으면 스킵
    _enable_memory_savers(pipe)       # slicing/tiling 등
    _maybe_cpu_offload(pipe, device=device, mode=offload_mode)

    # (옵션) 토치 컴파일은 환경마다 이슈가 있어 기본 비활성
    # try:
    #     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
    # except Exception:
    #     pass

    if prompt is None:
        prompt = (
            "cute illustrated announcement poster, flat vector illustration style, "
            "green pastel background, notebook paper frame with flowers and cats, "
            "clean layout, large empty space for headline and date, "
            "blank paper area reserved for text, no printed text"
        )

    if negative_prompt is None:
        negative_prompt = (
            "letters, words, typography, watermark, logo, illegible text, text artifacts, "
            "photo realistic, 3d render, depth of field, cinematic lighting"
        )

    gen = torch.Generator(device=device) if device == "cuda" else torch.Generator()
    gen = gen.manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_img,
            control_image=ctrl_img,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=gen,
        )

    out_img = result.images[0]
    out_img.save(out_path)
    print("[INFO] saved:", out_path)

    # 메모리 정리
    del pipe
    del controlnet
    _clear_cuda()

    return out_path


if __name__ == "__main__":
    init_path = f"{PROJECT_ROOT}/src/generation/image_generation/test_images/test_poster_prompt.png"

    # 이미지 로드 + (필요시) 리사이즈 적용을 여기서 먼저 하고 boxes를 그 크기에 맞춰 잡는 게 안전
    base_img = _load_image_rgb(init_path)
    base_img = _resize_if_needed(base_img, max_side=1024)  # 필요시 896/768로 낮추기
    boxes = default_boxes_for_this_template(base_img)

    # 임시로 리사이즈된 이미지를 저장해 init으로 사용해도 됨
    # (boxes가 리사이즈 이미지 기준이므로)
    resized_init_path = "tmp_resized_init.png"
    base_img.save(resized_init_path)

    bg_out = run_sdxl_controlnet_lock_text_areas(
        init_image_path=resized_init_path,
        out_path="bg_locked.png",
        boxes=boxes,
        strength=0.15,
        cfg=5.5,
        steps=25,
        seed=1000,
        offload_mode="none",   # VRAM 부족하면 "model" 또는 "sequential"
        max_side=None,         # 위에서 이미 리사이즈했으니 여기선 None 권장
    )

    print("saved:", bg_out)
