"""\
Drink Poster Pipeline (SDXL + ControlNet + Auto layout + Text/Sticker overlay)

Goal
- User uploads a drink/product photo (e.g., iced matcha).
- Generate a poster like the sample: dark chalkboard background, central drink, starburst accents,
  speech bubbles with short copy, big script headline + product name, brand handle.

What this script does
1) Detect the main product region (subject bbox) automatically from the uploaded photo (no hardcoded example boxes)
   - Uses a simple, robust saliency-style heuristic (edge + saturation) and finds the largest connected component.
   - Falls back to a center bbox if detection fails.
2) Derive text/sticker boxes from the detected subject bbox (layout engine)
3) Build ControlNet conditioning images:
   - Canny edges to preserve the drink silhouette
   - Layout guide (rectangles) to reserve text areas
4) SDXL ControlNet Img2Img to create the poster background while keeping subject and empty text regions
5) Overlay typography + speech bubbles + starbursts with Pillow (reliable text rendering)

Run
  python -m src.generation.image_generation.drink_poster_pipeline \
    --input /path/to/user_drink.jpg \
    --outdir ./outputs \
    --title "Craving" \
    --product "Matcha!" \
    --bubble1 "Creamy\nIced\nMatcha" \
    --bubble2 "Matcha\nMilk\nVanilla Bliss" \
    --brand "@yourbrandname"

Notes
- This script assumes:
  - diffusers + torch installed
  - A SDXL base model available (HF id or local path)
  - A SDXL ControlNet canny model available (HF id or local path)
- If xformers isn't installed, it will skip it safely.
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline


# ---------------------------
# Paths / Fonts
# ---------------------------
DEFAULT_FONT_DIR = Path("data/assets/font")

# You can swap these with your project fonts.
DEFAULT_FONTS = {
    "script": DEFAULT_FONT_DIR / "NanumPenScript-Regular.ttf",        # hand-written
    "headline": DEFAULT_FONT_DIR / "Cafe24Ssurround-v2.0.ttf",        # bold cute headline
    "body": DEFAULT_FONT_DIR / "NotoSansKR-Medium.ttf",              # readable body
}


# ---------------------------
# Helpers: memory
# ---------------------------

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

    try:
        pipe.enable_vae_tiling()
        print("[INFO] VAE tiling enabled")
    except Exception:
        pass


def _maybe_cpu_offload(pipe, device: str, mode: str = "none"):
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
    except Exception as e:
        print(f"[WARN] CPU offload not available, skipping ({type(e).__name__}: {e})")


# ---------------------------
# Helpers: image
# ---------------------------

def load_image_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def resize_max_side(img: Image.Image, max_side: int | None) -> Tuple[Image.Image, float]:
    if not max_side:
        return img, 1.0
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img, 1.0
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.LANCZOS), scale


# ---------------------------
# 1) Subject bbox detection (no hardcoded example)
# ---------------------------

def _to_gray_np(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    # luminance
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    # Lightweight Sobel without OpenCV
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    def conv2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
        # pad
        ap = np.pad(a, 1, mode="edge")
        out = np.zeros_like(a)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                patch = ap[y : y + 3, x : x + 3]
                out[y, x] = float(np.sum(patch * k))
        return out

    gx = conv2d(gray, kx)
    gy = conv2d(gray, ky)
    mag = np.sqrt(gx * gx + gy * gy)
    # normalize
    mag = mag / (mag.max() + 1e-6)
    return mag


def _rgb_to_hsv_np(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    # hue
    h = np.zeros_like(mx)
    mask = diff > 1e-6
    # where max is r
    idx = mask & (mx == r)
    h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    # where max is g
    idx = mask & (mx == g)
    h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
    # where max is b
    idx = mask & (mx == b)
    h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360

    s = np.zeros_like(mx)
    s[mx > 1e-6] = diff[mx > 1e-6] / mx[mx > 1e-6]
    v = mx

    hsv = np.stack([h / 360.0, s, v], axis=-1)
    return hsv


def _largest_component_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    # Simple flood-fill labeling (no OpenCV dependency)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)

    best_area = 0
    best_bbox = None

    def neighbors(y: int, x: int):
        if y > 0:
            yield y - 1, x
        if y + 1 < h:
            yield y + 1, x
        if x > 0:
            yield y, x - 1
        if x + 1 < w:
            yield y, x + 1

    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = 1

                minx = maxx = x
                miny = maxy = y
                area = 0

                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    minx = min(minx, cx)
                    maxx = max(maxx, cx)
                    miny = min(miny, cy)
                    maxy = max(maxy, cy)

                    for ny, nx in neighbors(cy, cx):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = 1
                            stack.append((ny, nx))

                if area > best_area:
                    best_area = area
                    best_bbox = (minx, miny, maxx + 1, maxy + 1)

    return best_bbox


def detect_subject_bbox(img: Image.Image) -> Tuple[int, int, int, int]:
    """Detect main object bbox using a saliency-ish heuristic.

    Works reasonably for product photos (glass/cup/bottle) even on simple backgrounds.
    If it fails, returns a safe center bbox.
    """
    w, h = img.size

    gray = _to_gray_np(img)
    edges = _sobel_edges(gray)

    hsv = _rgb_to_hsv_np(img)
    sat = hsv[..., 1]

    # Saliency map: edges + saturation (product tends to have strong edges / saturation)
    sal = 0.65 * edges + 0.35 * sat
    sal = sal / (sal.max() + 1e-6)

    # Threshold at percentile to create a binary mask
    thr = float(np.quantile(sal, 0.90))
    mask = sal >= thr

    # Remove borders (often noisy)
    pad = int(0.02 * min(w, h))
    if pad > 0:
        mask[:pad, :] = False
        mask[-pad:, :] = False
        mask[:, :pad] = False
        mask[:, -pad:] = False

    bbox = _largest_component_bbox(mask)

    if bbox is None:
        # fallback center bbox
        cx0 = int(0.30 * w)
        cy0 = int(0.18 * h)
        cx1 = int(0.70 * w)
        cy1 = int(0.92 * h)
        return (cx0, cy0, cx1, cy1)

    x0, y0, x1, y1 = bbox

    # Expand bbox a bit
    dx = int(0.06 * (x1 - x0))
    dy = int(0.06 * (y1 - y0))
    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = min(w, x1 + dx)
    y1 = min(h, y1 + dy)

    # If bbox is too small/large, fallback
    area = (x1 - x0) * (y1 - y0)
    if area < 0.02 * w * h or area > 0.90 * w * h:
        cx0 = int(0.30 * w)
        cy0 = int(0.18 * h)
        cx1 = int(0.70 * w)
        cy1 = int(0.92 * h)
        return (cx0, cy0, cx1, cy1)

    return (x0, y0, x1, y1)


# ---------------------------
# 2) Layout engine (derive text boxes from subject bbox)
# ---------------------------

@dataclass
class Layout:
    subject_bbox: Tuple[int, int, int, int]
    boxes: Dict[str, Tuple[int, int, int, int]]


def derive_matcha_poster_layout(canvas_size: Tuple[int, int], subject_bbox: Tuple[int, int, int, int]) -> Layout:
    """Layout similar to sample:

    - Big script at top (Craving)
    - Big script at bottom (Matcha!)
    - Two speech bubbles (left-bottom / right-top)
    - Brand handle bottom-right

    Boxes are derived from subject bbox so it adapts to different photos.
    """
    w, h = canvas_size
    sx0, sy0, sx1, sy1 = subject_bbox
    sw = sx1 - sx0
    sh = sy1 - sy0

    # margins based on canvas
    mx = int(0.06 * w)
    my = int(0.06 * h)

    # Top title box: full width, above subject
    top_h = int(0.18 * h)
    top = (mx, my, w - mx, my + top_h)

    # Bottom product name box
    bot_h = int(0.22 * h)
    bottom = (mx, h - my - bot_h, w - mx, h - my)

    # Right-top bubble: to the right of subject upper half
    rb_w = int(0.30 * w)
    rb_h = int(0.22 * h)
    rb_x0 = min(w - mx - rb_w, sx1 + int(0.03 * w))
    rb_y0 = max(my + int(0.10 * h), sy0 + int(0.08 * sh))
    right_bubble = (rb_x0, rb_y0, rb_x0 + rb_w, rb_y0 + rb_h)

    # Left-bottom bubble: to the left of subject lower half
    lb_w = int(0.30 * w)
    lb_h = int(0.22 * h)
    lb_x0 = max(mx, sx0 - int(0.03 * w) - lb_w)
    lb_y0 = min(h - my - bot_h - int(0.06 * h) - lb_h, sy0 + int(0.45 * sh))
    left_bubble = (lb_x0, lb_y0, lb_x0 + lb_w, lb_y0 + lb_h)

    # Brand handle
    br_w = int(0.35 * w)
    br_h = int(0.08 * h)
    brand = (w - mx - br_w, h - my - int(0.02*h) - br_h, w - mx, h - my - int(0.02*h))

    boxes = {
        "top_title": top,
        "bottom_title": bottom,
        "bubble_right": right_bubble,
        "bubble_left": left_bubble,
        "brand": brand,
    }

    return Layout(subject_bbox=subject_bbox, boxes=boxes)


# ---------------------------
# 3) Control images
# ---------------------------

def make_layout_control_image(canvas_size: Tuple[int, int], boxes: Dict[str, Tuple[int, int, int, int]], line_width: int = 6) -> Image.Image:
    w, h = canvas_size
    ctrl = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(ctrl)

    for _, (x0, y0, x1, y1) in boxes.items():
        d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=line_width)

    return ctrl


def make_canny_control_image(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Canny control image.

    Uses OpenCV if available; otherwise falls back to an edge-map approximation.
    """
    try:
        import cv2  # type: ignore

        arr = np.asarray(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        edges_rgb = np.stack([edges, edges, edges], axis=-1)
        return Image.fromarray(edges_rgb)
    except Exception:
        # fallback: sobel magnitude
        gray = _to_gray_np(img)
        edges = _sobel_edges(gray)
        edges_u8 = (edges * 255).astype(np.uint8)
        edges_rgb = np.stack([edges_u8, edges_u8, edges_u8], axis=-1)
        return Image.fromarray(edges_rgb)


# ---------------------------
# 4) SDXL ControlNet composition
# ---------------------------

def build_pipe(
    base_model: str,
    controlnet_path: str,
    device: str,
    dtype: torch.dtype,
    offload_mode: str = "none",
) -> StableDiffusionXLControlNetImg2ImgPipeline:
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )

    if offload_mode == "none":
        pipe = pipe.to(device)

    _maybe_enable_xformers(pipe)
    _enable_memory_savers(pipe)
    _maybe_cpu_offload(pipe, device=device, mode=offload_mode)

    return pipe


def run_poster_generation(
    user_photo: Image.Image,
    layout: Layout,
    out_path: Path,
    base_model: str,
    controlnet_canny: str,
    prompt: str,
    negative: str,
    steps: int = 30,
    cfg: float = 6.0,
    strength: float = 0.55,
    control_scale: float = 0.9,
    seed: int = 1234,
    device: str | None = None,
    offload_mode: str = "none",
) -> Image.Image:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    init = user_photo.convert("RGB")

    # Make controls
    canny = make_canny_control_image(init)
    layout_ctrl = make_layout_control_image(init.size, layout.boxes)

    # We combine two conditioning images into one by blending (simple but effective):
    # - canny preserves subject
    # - layout enforces empty text areas
    ctrl = Image.blend(canny, layout_ctrl, alpha=0.45)

    _clear_cuda()

    pipe = build_pipe(
        base_model=base_model,
        controlnet_path=controlnet_canny,
        device=device,
        dtype=dtype,
        offload_mode=offload_mode,
    )

    gen = torch.Generator(device=device) if device == "cuda" else torch.Generator()
    gen.manual_seed(seed)

    with torch.inference_mode():
        res = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=init,
            control_image=ctrl,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=control_scale,
            generator=gen,
        )

    out = res.images[0]
    save_image(out, out_path)

    # cleanup
    del pipe
    _clear_cuda()

    return out


# ---------------------------
# 5) Overlay: text + speech bubbles + starbursts
# ---------------------------

def resolve_font_path(font_path: Path) -> Path | None:
    if font_path.is_file():
        return font_path
    # allow relative from cwd
    p2 = Path.cwd() / font_path
    if p2.is_file():
        return p2
    return None


def load_font(font_path: Path | None, size: int) -> ImageFont.ImageFont:
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception:
            pass
    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bb = draw.textbbox((0, 0), text, font=font)
    return (bb[2] - bb[0], bb[3] - bb[1])


def fit_font(draw: ImageDraw.ImageDraw, text: str, font_path: Path | None, box_w: int, box_h: int, max_size: int, min_size: int = 14):
    for s in range(max_size, min_size - 1, -2):
        font = load_font(font_path, s)
        w, h = text_bbox(draw, text, font)
        if w <= box_w and h <= box_h:
            return font
    return load_font(font_path, min_size)


def draw_glow_text(
    base: Image.Image,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill=(255, 255, 255, 255),
    stroke_width: int = 3,
    stroke_fill=(0, 0, 0, 255),
    glow_radius: int = 6,
    glow_alpha: int = 120,
):
    base = base.convert("RGBA")
    w, h = base.size

    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text(xy, text, font=font, fill=(255, 255, 255, glow_alpha), stroke_width=stroke_width, stroke_fill=(255, 255, 255, glow_alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(glow_radius))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

    out = Image.alpha_composite(base, glow)
    out = Image.alpha_composite(out, overlay)
    return out


def draw_starburst(img: Image.Image, center: Tuple[int, int], r_inner: int, r_outer: int, spikes: int, fill: Tuple[int, int, int, int]):
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    cx, cy = center
    pts = []
    for i in range(spikes * 2):
        ang = (i * np.pi) / spikes
        r = r_outer if i % 2 == 0 else r_inner
        x = cx + int(np.cos(ang) * r)
        y = cy + int(np.sin(ang) * r)
        pts.append((x, y))

    d.polygon(pts, fill=fill)
    return Image.alpha_composite(img, overlay)


def draw_cloud_bubble(img: Image.Image, bbox: Tuple[int, int, int, int], outline=(255, 255, 255, 255), width: int = 6, fill=None):
    """Simple cloud bubble: multiple circles along the border."""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    # base fill
    if fill is not None:
        d.rounded_rectangle([x0, y0, x1, y1], radius=int(min(w, h) * 0.25), fill=fill)

    # cloud outline: circles
    n = 10
    rad = int(min(w, h) * 0.18)

    # top
    for i in range(n):
        x = x0 + int((i + 0.5) * w / n)
        d.ellipse([x - rad, y0 - rad, x + rad, y0 + rad], outline=outline, width=width)

    # bottom
    for i in range(n):
        x = x0 + int((i + 0.5) * w / n)
        d.ellipse([x - rad, y1 - rad, x + rad, y1 + rad], outline=outline, width=width)

    # left
    for i in range(n):
        y = y0 + int((i + 0.5) * h / n)
        d.ellipse([x0 - rad, y - rad, x0 + rad, y + rad], outline=outline, width=width)

    # right
    for i in range(n):
        y = y0 + int((i + 0.5) * h / n)
        d.ellipse([x1 - rad, y - rad, x1 + rad, y + rad], outline=outline, width=width)

    return Image.alpha_composite(img, overlay)


def overlay_matcha_poster(
    bg: Image.Image,
    layout: Layout,
    title_top: str,
    title_bottom: str,
    bubble_right: str,
    bubble_left: str,
    brand: str,
    fonts: Dict[str, Path],
) -> Image.Image:
    bg = bg.convert("RGBA")
    w, h = bg.size

    # Resolve fonts (fallback to default)
    f_script = resolve_font_path(fonts["script"]) or resolve_font_path(DEFAULT_FONTS["script"])  # type: ignore
    f_head = resolve_font_path(fonts["headline"]) or resolve_font_path(DEFAULT_FONTS["headline"])  # type: ignore
    f_body = resolve_font_path(fonts["body"]) or resolve_font_path(DEFAULT_FONTS["body"])  # type: ignore

    d = ImageDraw.Draw(bg)

    # Accent starbursts (positions relative to subject)
    sx0, sy0, sx1, sy1 = layout.subject_bbox
    star_color = (170, 210, 80, 255)
    bg = draw_starburst(bg, (sx0 - int(0.08*w), sy0 + int(0.30*(sy1-sy0))), 22, 60, 12, star_color)
    bg = draw_starburst(bg, (sx1 + int(0.08*w), sy0 + int(0.62*(sy1-sy0))), 22, 70, 12, star_color)

    # Speech bubbles
    b1 = layout.boxes["bubble_right"]
    b2 = layout.boxes["bubble_left"]
    bg = draw_cloud_bubble(bg, b1, outline=(255, 255, 255, 255), width=5, fill=None)
    bg = draw_cloud_bubble(bg, b2, outline=(255, 255, 255, 255), width=5, fill=None)

    # Bubble texts (centered)
    for bbox, txt in [(b1, bubble_right), (b2, bubble_left)]:
        x0, y0, x1, y1 = bbox
        box_w, box_h = (x1 - x0), (y1 - y0)
        font = fit_font(d, txt, f_body, int(box_w * 0.75), int(box_h * 0.65), max_size=48)
        tw, th = text_bbox(d, txt, font)
        tx = x0 + (box_w - tw) // 2
        ty = y0 + (box_h - th) // 2
        bg = draw_glow_text(bg, (tx, ty), txt, font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255), glow_radius=5, glow_alpha=80)

    # Top title (script)
    tbox = layout.boxes["top_title"]
    x0, y0, x1, y1 = tbox
    font_top = fit_font(d, title_top, f_script, int((x1-x0)*0.95), int((y1-y0)*0.85), max_size=120)
    tw, th = text_bbox(d, title_top, font_top)
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 - th) // 2
    bg = draw_glow_text(bg, (tx, ty), title_top, font_top, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255), glow_radius=8, glow_alpha=110)

    # Bottom title (script)
    bbox = layout.boxes["bottom_title"]
    x0, y0, x1, y1 = bbox
    font_bot = fit_font(d, title_bottom, f_script, int((x1-x0)*0.95), int((y1-y0)*0.85), max_size=150)
    tw, th = text_bbox(d, title_bottom, font_bot)
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 - th) // 2
    bg = draw_glow_text(bg, (tx, ty), title_bottom, font_bot, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255), glow_radius=10, glow_alpha=120)

    # Brand handle (bottom-right)
    bbox = layout.boxes["brand"]
    x0, y0, x1, y1 = bbox
    font_brand = fit_font(d, brand, f_head, int((x1-x0)*0.98), int((y1-y0)*0.90), max_size=48)
    tw, th = text_bbox(d, brand, font_brand)
    tx = x1 - tw
    ty = y0 + (y1 - y0 - th) // 2
    bg = draw_glow_text(bg, (tx, ty), brand, font_brand, fill=(255, 255, 255, 230), stroke_width=1, stroke_fill=(0, 0, 0, 220), glow_radius=4, glow_alpha=60)

    return bg.convert("RGB")


# ---------------------------
# CLI: pipeline
# ---------------------------

def build_default_prompt() -> Tuple[str, str]:
    prompt = (
        "social media drink poster, iced matcha latte in a tall glass centered, "
        "dark green chalkboard background, studio product lighting, high contrast, "
        "clean composition with empty space for typography and stickers, "
        "modern cafe poster design, minimal clutter, "
        "no printed text"
    )

    negative = (
        "letters, words, watermark, logo, illegible text, text artifacts, "
        "bad anatomy, extra objects, messy layout, low quality"
    )

    return prompt, negative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to user drink photo")
    ap.add_argument("--outdir", default="outputs", help="Output directory")

    ap.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--controlnet_canny", default="/opt/ai-models/sdxl/diffusers--controlnet-canny-sdxl-1.0")

    ap.add_argument("--title", default="Craving")
    ap.add_argument("--product", default="Matcha!")
    ap.add_argument("--bubble1", default="Creamy\nIced\nMatcha")
    ap.add_argument("--bubble2", default="Matcha\nMilk\nVanilla Bliss")
    ap.add_argument("--brand", default="@yourbrandname")

    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--control_scale", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--max_side", type=int, default=1024, help="Resize max side for VRAM safety")
    ap.add_argument("--offload", default="none", choices=["none", "model", "sequential"], help="CPU offload mode")

    ap.add_argument("--font_script", default=str(DEFAULT_FONTS["script"]))
    ap.add_argument("--font_headline", default=str(DEFAULT_FONTS["headline"]))
    ap.add_argument("--font_body", default=str(DEFAULT_FONTS["body"]))

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    user = load_image_rgb(args.input)
    user, scale = resize_max_side(user, args.max_side)
    print(f"[INFO] input size={user.size} scale={scale:.3f}")

    # 1) subject bbox detection
    subject_bbox = detect_subject_bbox(user)
    print("[INFO] subject bbox:", subject_bbox)

    # debug bbox
    dbg = user.copy()
    dd = ImageDraw.Draw(dbg)
    dd.rectangle(subject_bbox, outline=(255, 0, 0), width=5)
    save_image(dbg, outdir / "debug_subject_bbox.png")

    # 2) layout derivation
    layout = derive_matcha_poster_layout(user.size, subject_bbox)

    # debug layout boxes
    dbg2 = user.copy()
    dd2 = ImageDraw.Draw(dbg2)
    for k, b in layout.boxes.items():
        dd2.rectangle(b, outline=(0, 255, 0), width=4)
        dd2.text((b[0] + 6, b[1] + 6), k, fill=(0, 255, 0))
    save_image(dbg2, outdir / "debug_layout_boxes.png")

    # 3-4) SDXL generation (keep subject + reserve text boxes)
    prompt, negative = build_default_prompt()
    bg = run_poster_generation(
        user_photo=user,
        layout=layout,
        out_path=outdir / "bg_generated.png",
        base_model=args.base_model,
        controlnet_canny=args.controlnet_canny,
        prompt=prompt,
        negative=negative,
        steps=args.steps,
        cfg=args.cfg,
        strength=args.strength,
        control_scale=args.control_scale,
        seed=args.seed,
        offload_mode=args.offload,
    )

    # 5) overlay typography + bubbles + starbursts
    fonts = {
        "script": Path(args.font_script),
        "headline": Path(args.font_headline),
        "body": Path(args.font_body),
    }

    final = overlay_matcha_poster(
        bg=bg,
        layout=layout,
        title_top=args.title,
        title_bottom=args.product,
        bubble_right=args.bubble1,
        bubble_left=args.bubble2,
        brand=args.brand,
        fonts=fonts,
    )

    save_image(final, outdir / "final_matcha_poster.png")
    print("[DONE]", outdir / "final_matcha_poster.png")


if __name__ == "__main__":
    main()
