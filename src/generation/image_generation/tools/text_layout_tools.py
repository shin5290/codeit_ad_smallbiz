"""
Text Layout Tools for GPT-4V
GPT-4V가 이미지를 분석하고 텍스트 레이아웃을 결정하기 위한 Tool Definition

작성자: 이현석
"""

from typing import Dict, List, Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ==============================================================================
# 동적 폰트 목록 가져오기
# ==============================================================================

def get_font_enum() -> List[str]:
    """
    사용 가능한 폰트 목록을 동적으로 가져옴
    /mnt/fonts의 커스텀 폰트 포함
    """
    try:
        from .font_loader import get_available_fonts
        return get_available_fonts()
    except Exception as e:
        # Fallback: 기본 폰트만 반환
        logger.warning(f"⚠️ Failed to load font list: {e}")
        return [
            "NanumGothic",
            "NanumGothicBold",
            "NanumMyeongjo",
            "NotoSansKR",
            "NotoSerifKR"
        ]

# ==============================================================================
# OpenAI Function Calling Tool Definition
# ==============================================================================

def get_text_overlay_tool() -> Dict[str, Any]:
    """
    동적 폰트 목록을 포함한 TEXT_OVERLAY_TOOL 생성
    """
    available_fonts = get_font_enum()

    return {
        "type": "function",
        "function": {
            "name": "apply_text_overlay",
            "description": "이미지에 텍스트를 오버레이하기 위한 레이아웃 명세. 각 텍스트 레이어의 위치, 폰트, 색상, 효과를 지정합니다.",
            "parameters": {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "description": "텍스트 레이어 배열 (여러 텍스트를 다른 위치에 배치 가능)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "표시할 텍스트 내용 (예: '딸기라떼', '신메뉴 출시')"
                            },
                            "position": {
                                "type": "object",
                                "description": "텍스트 위치 (0.0-1.0 정규화 좌표)",
                                "properties": {
                                    "x": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "가로 위치 (0.0=왼쪽, 0.5=중앙, 1.0=오른쪽)"
                                    },
                                    "y": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "세로 위치 (0.0=상단, 0.5=중앙, 1.0=하단)"
                                    },
                                    "anchor": {
                                        "type": "string",
                                        "enum": ["top_left", "top_center", "top_right",
                                                "center_left", "center", "center_right",
                                                "bottom_left", "bottom_center", "bottom_right"],
                                        "description": "텍스트 앵커 포인트 (텍스트의 어느 부분을 기준으로 배치할지)"
                                    }
                                },
                                "required": ["x", "y", "anchor"]
                            },
                            "font": {
                                "type": "object",
                                "description": "폰트 설정",
                                "properties": {
                                    "family": {
                                        "type": "string",
                                        "enum": available_fonts,
                                        "description": "폰트 패밀리. 이미지 분위기와 레이어 역할에 맞춰 선택 (일반 레이어와 강조 레이어에 다른 폰트 사용)"
                                    },
                                    "size": {
                                        "type": "integer",
                                        "minimum": 18,
                                        "maximum": 200,
                                        "description": "폰트 크기 (픽셀). CRITICAL: Use dynamically calculated size ranges from IMAGE DIMENSIONS section above!"
                                    }
                                },
                                "required": ["family", "size"]
                            },
                            "color": {
                                "type": "object",
                                "description": "텍스트 색상 (RGBA)",
                                "properties": {
                                    "r": {"type": "integer", "minimum": 0, "maximum": 255},
                                    "g": {"type": "integer", "minimum": 0, "maximum": 255},
                                    "b": {"type": "integer", "minimum": 0, "maximum": 255},
                                    "a": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "투명도 (0.0=투명, 1.0=불투명)"
                                    }
                                },
                                "required": ["r", "g", "b", "a"]
                            },
                            "effects": {
                                "type": "object",
                                "description": "텍스트 효과 (가독성 향상)",
                                "properties": {
                                    "stroke": {
                                        "type": "object",
                                        "description": "외곽선 (복잡한 배경에서 가독성 향상)",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "width": {
                                                "type": "integer",
                                                "minimum": 1,
                                                "maximum": 10,
                                                "description": "외곽선 두께 (픽셀)"
                                            },
                                            "color": {
                                                "type": "object",
                                                "properties": {
                                                    "r": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "g": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "b": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "a": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                                                },
                                                "required": ["r", "g", "b", "a"]
                                            }
                                        },
                                        "required": ["enabled"]
                                    },
                                    "shadow": {
                                        "type": "object",
                                        "description": "그림자 (입체감, 가독성 향상)",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "offset_x": {
                                                "type": "integer",
                                                "minimum": -20,
                                                "maximum": 20,
                                                "description": "그림자 X 오프셋 (픽셀)"
                                            },
                                            "offset_y": {
                                                "type": "integer",
                                                "minimum": -20,
                                                "maximum": 20,
                                                "description": "그림자 Y 오프셋 (픽셀)"
                                            },
                                            "blur": {
                                                "type": "integer",
                                                "minimum": 0,
                                                "maximum": 20,
                                                "description": "그림자 블러 강도"
                                            },
                                            "color": {
                                                "type": "object",
                                                "properties": {
                                                    "r": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "g": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "b": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "a": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                                                },
                                                "required": ["r", "g", "b", "a"]
                                            }
                                        },
                                        "required": ["enabled"]
                                    },
                                    "background_box": {
                                        "type": "object",
                                        "description": "배경 박스 (매우 복잡한 배경에서 사용)",
                                        "properties": {
                                            "enabled": {"type": "boolean"},
                                            "padding": {
                                                "type": "integer",
                                                "minimum": 5,
                                                "maximum": 30,
                                                "description": "텍스트 주변 여백 (픽셀)"
                                            },
                                            "color": {
                                                "type": "object",
                                                "properties": {
                                                    "r": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "g": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "b": {"type": "integer", "minimum": 0, "maximum": 255},
                                                    "a": {
                                                        "type": "number",
                                                        "minimum": 0.0,
                                                        "maximum": 1.0,
                                                        "description": "투명도 (0.6-0.8 권장)"
                                                    }
                                                },
                                                "required": ["r", "g", "b", "a"]
                                            },
                                            "border_radius": {
                                                "type": "integer",
                                                "minimum": 0,
                                                "maximum": 20,
                                                "description": "모서리 둥글기 (픽셀)"
                                            }
                                        },
                                        "required": ["enabled"]
                                    }
                                }
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "이 레이아웃을 선택한 이유 (디버깅/검증용)"
                            }
                        },
                        "required": ["text", "position", "font", "color", "reasoning"]
                    }
                }
            },
            "required": ["layers"]
        }
    }
}

# Backward compatibility: 기본 TEXT_OVERLAY_TOOL (레거시 코드용)
TEXT_OVERLAY_TOOL = get_text_overlay_tool()


# ==============================================================================
# GPT-4V Analysis Prompt Template
# ==============================================================================

def get_analysis_prompt(text_data: Dict[str, str], image_context: str = "", image_size: tuple = None) -> str:
    """
    GPT-4V 이미지 분석 프롬프트 생성

    Args:
        text_data: 오버레이할 텍스트 데이터
            예: {"product_name": "딸기라떼", "tagline": "신메뉴 출시"}
        image_context: 이미지 컨텍스트 (선택, 예: "카페 메뉴 광고")
        image_size: 이미지 크기 (width, height) - 폰트 크기 조정용

    Returns:
        str: GPT-4V에게 전달할 프롬프트
    """

    # 텍스트 데이터를 읽기 쉽게 포맷팅
    text_items = "\n".join([f"- {key}: \"{value}\"" for key, value in text_data.items()])

    # 사용 가능한 폰트 목록 가져오기
    available_fonts = get_font_enum()
    font_list = "\n".join([f"- {font}" for font in available_fonts])

    # 이미지 크기 정보 포맷팅
    if image_size:
        width, height = image_size
        aspect_ratio = width / height
        if aspect_ratio > 1.2:
            orientation = "가로 (Landscape)"
        elif aspect_ratio < 0.8:
            orientation = "세로 (Portrait/Vertical)"
        else:
            orientation = "정사각형 (Square)"

        # 동적 폰트 크기 계산 (이미지의 짧은 쪽 기준)
        # 기준: 1024px 기준으로 제품명=100px, 부가문구=50px
        # 공식: font_size = base_size * (shorter_dimension / 1024) * scaling_factor
        shorter_dim = min(width, height)
        base_scale = shorter_dim / 1024.0

        # 제품명 (Main Text): 이미지 짧은 쪽의 8-13% 크기
        main_min = int(shorter_dim * 0.08)
        main_max = int(shorter_dim * 0.13)

        # 부가문구 (Secondary): 이미지 짧은 쪽의 4-7% 크기
        secondary_min = int(shorter_dim * 0.04)
        secondary_max = int(shorter_dim * 0.07)

        # 작은 텍스트: 이미지 짧은 쪽의 2.5-4% 크기
        fine_min = int(shorter_dim * 0.025)
        fine_max = int(shorter_dim * 0.04)

        # 최소/최대 제한 (너무 작거나 크지 않게)
        main_min = max(40, min(main_min, 160))
        main_max = max(60, min(main_max, 200))
        secondary_min = max(25, min(secondary_min, 90))
        secondary_max = max(35, min(secondary_max, 120))
        fine_min = max(18, min(fine_min, 50))
        fine_max = max(25, min(fine_max, 70))

        size_info = f"""
## IMAGE DIMENSIONS (CRITICAL FOR FONT SIZING!)
- **Width**: {width}px
- **Height**: {height}px
- **Aspect Ratio**: {aspect_ratio:.2f} ({orientation})
- **Shorter Dimension**: {shorter_dim}px (used as font size baseline)

### DYNAMIC FONT SIZE CALCULATION (SMART SCALING!)

**📐 Formula**: Font sizes are calculated as **percentage of the shorter dimension** to ensure text fits perfectly regardless of image size.

**Calculated sizes for THIS image**:
- **Product Name / Main Text**: **{main_min}-{main_max}px**
  - Calculation: {shorter_dim}px × 8-13% = optimal size for primary text
  - This ensures the main text is LARGE and DOMINANT but doesn't overflow

- **Tagline / Secondary Text**: **{secondary_min}-{secondary_max}px**
  - Calculation: {shorter_dim}px × 4-7% = supporting text size
  - Clear hierarchy below main text, still very readable

- **Fine Print / Details**: **{fine_min}-{fine_max}px**
  - Calculation: {shorter_dim}px × 2.5-4% = small but legible

**🎯 Sizing Strategy ({orientation})**:
"""

        if aspect_ratio < 0.8:  # 세로 이미지
            size_info += f"""- **VERTICAL** image: Using width ({width}px) as constraint
- Text scales down automatically to fit narrow canvas
- Prioritize readability over impact - Korean text needs breathing room
- **Recommendation**: Use sizes in LOWER half of each range for better fit
"""
        elif aspect_ratio > 1.2:  # 가로 이미지
            size_info += f"""- **HORIZONTAL** image: Using height ({height}px) as constraint
- Wide canvas allows for bold, impactful text
- **Recommendation**: Use sizes in UPPER half of each range for maximum presence
"""
        else:  # 정사방형
            size_info += f"""- **SQUARE** image: Balanced dimensions
- **Recommendation**: Use MID-range sizes for harmonious composition
"""

        size_info += f"""
**⚠️ CRITICAL RULES**:
1. **ALWAYS use the calculated ranges above** - they're optimized for THIS specific image size
2. Longer text (10+ characters) → use LOWER end of range
3. Shorter text (3-5 characters) → use UPPER end of range
4. Korean text is denser → add 10-15% more size than you'd use for English
5. When in doubt, test mentally: "Would {main_max}px text fit comfortably in {width}px width?"
"""
    else:
        size_info = ""

    prompt = f"""You are an expert graphic designer specializing in advertising image composition.

## YOUR TASK
Analyze the provided advertising image and determine the optimal layout for overlaying the following text:

{text_items}

## IMAGE CONTEXT
{image_context if image_context else "General advertising image"}

{size_info}

## ANALYSIS GUIDELINES

### 1. POSITIONING STRATEGY
- **Identify empty/spacious areas**: Avoid placing text over important subjects (products, faces, key elements)
- **Rule of thirds**: Consider placing text at intersection points or along grid lines
- **Visual flow**: Text should guide the viewer's eye naturally (top→bottom, left→right for Korean)
- **Hierarchy**: Primary text (product name) should be more prominent than secondary text (tagline)

### 2. COLOR SELECTION (CRITICAL FOR VISIBILITY!)
- **Contrast ratio**: Ensure WCAG AA compliance (contrast ratio > 4.5:1 for readability)
- **Background analysis**:
  - Dark background (avg brightness < 128) → Use WHITE text (RGB: 255, 255, 255) with dark stroke
  - Light background (avg brightness > 128) → Use BLACK text (RGB: 0, 0, 0) with white stroke
  - Complex/medium background → ALWAYS use strong stroke (width: 4-6px) with contrasting color
- **CRITICAL**: When in doubt, use WHITE text with thick BLACK stroke (works on 90% of images)
- **Brand harmony**: Colors should complement the overall image mood

### 3. FONT SELECTION (SMART LAYER-BASED STRATEGY)

**CRITICAL: Use DIFFERENT fonts for regular vs. emphasized layers!**

#### Available Fonts:
{font_list}

#### Font Personality Guide (MATCH FONT TO IMAGE MOOD!):

**🎨 Cute/Playful/Character** (desserts, mascots, children):
- **BMJUA_ttf**: Round, cute, bouncy - Pokemon, character products
- **BMDOHYEON_ttf**: Bold, fun, energetic - snacks, playful brands
- **Cafe24Ssurround**: Rounded, friendly - cafes, bakeries
- **NanumPenScript-Regular**: Handwritten - personal, warm

**💼 Modern/Clean/Professional** (tech, fashion, corporate):
- **Pretendard-Bold**: Modern, sharp - tech/startups
- **SUIT-Bold**: Contemporary - business, fashion
- **SpoqaHanSansNeo-Bold**: Clean sans - apps, services
- **NotoSansKR-Medium**: Neutral, versatile

**💥 Bold/Impact/Promotional** (sales, events):
- **BlackHanSans-Regular**: Ultra-bold - SALE, events
- **GmarketSansTTFBold**: Strong impact - promotions
- **SCDream9**: Very bold - strong emphasis
- **KBO Dia Gothic_bold**: Sports, dynamic

**✨ Elegant/Traditional/Premium**:
- **NanumMyeongjo**: Traditional serif - heritage
- **NotoSerifKR**: Classic serif - editorial
- **SCDream7**: Elegant - premium

**Font Pairing Examples**:
- Pokemon/character bread: Base=**BMJUA_ttf**, Emphasis=**BMDOHYEON_ttf**
- Modern cafe: Base=**Pretendard-Bold**, Emphasis=**Cafe24Ssurround**
- Sale event: Base=**NanumGothic**, Emphasis=**BlackHanSans-Regular**

**CRITICAL RULES**:
1. **Analyze image first** - cute? modern? traditional?
2. **Choose matching fonts** - not just defaults!
3. **Different fonts per layer** - create hierarchy!

### 4. FONT SIZE HIERARCHY
**⚠️ CRITICAL: IGNORE any generic size recommendations - ONLY use the DYNAMICALLY CALCULATED sizes from the IMAGE DIMENSIONS section above!**

The sizes in that section are **scientifically calculated** based on:
- Image dimensions (width × height)
- Aspect ratio (landscape/portrait/square)
- Text density (Korean vs English)
- Canvas constraints (what physically fits)

**Why dynamic sizing matters**:
- A 100px font on a 768px wide vertical image = 13% of width (too large!)
- The same 100px font on a 1344px wide horizontal image = 7% of width (perfect!)
- **Solution**: Use percentage-based sizing from IMAGE DIMENSIONS section

**Your job**: Simply pick a size from the calculated range based on text length and emphasis level.

### 5. EFFECTS DECISION TREE (ALWAYS PRIORITIZE READABILITY!)
- **Clean, simple background (sky, solid color, blur)**:
  - Add subtle shadow (offset: 3-4px, blur: 6-8px) for depth

- **Medium complexity (patterns, gradients, food photography)**:
  - ALWAYS add thick stroke (width: 4-6px, contrasting color) + shadow (offset: 3-5px, blur: 6-10px)

- **High complexity (busy scene, multiple objects)**:
  - Use THICK stroke (width: 6-8px) + shadow, OR semi-transparent background_box (alpha: 0.7-0.85)

- **DEFAULT SAFE OPTION**: White text + thick black stroke (6px) + subtle shadow works on 90% of images!

### 6. KOREAN TEXT CONSIDERATIONS
- Korean text is denser than Latin alphabet → needs slightly larger font size
- Ensure adequate letter spacing for readability
- Korean reads top-to-bottom or left-to-right → position accordingly

### 7. KEYWORD EMPHASIS (MANDATORY LAYER SPLITTING!)

**⚠️ CRITICAL: ALWAYS split text into multiple layers for visual hierarchy!**

**Step 1: Analyze image mood/style**
- Cute/playful → Use custom decorative fonts from font list
- Modern/clean → Use sans-serif fonts
- Traditional/elegant → Use serif fonts
- Food/bakery → Use rounded, friendly fonts

**Step 2: Split text into layers**
- Create AT LEAST 2 layers if text contains 2+ words
- Identify main keyword (제품명, 브랜드명, 핵심단어)

**Step 3: Font pairing based on image**
- Layer 1 (supporting text): Choose from available fonts matching image mood
- Layer 2 (emphasis): Choose DIFFERENT font from Layer 1, stronger/bolder variant

**Font Selection Priority**:
1. Check available font list above - prefer custom fonts for unique styling
2. Match font personality to image content (cute → rounded fonts, professional → clean sans-serif)
3. Use DIFFERENT fonts for different layers

**RULE: Single-layer text is BORING. Always create visual hierarchy!**

## OUTPUT REQUIREMENTS
You MUST call the `apply_text_overlay` function with a complete layout specification.

For each text item:
1. Choose a position that doesn't obscure important visual elements
2. Select a font that matches the image mood and industry
3. Pick colors with sufficient contrast (calculate approximate brightness if needed)
4. Decide on effects based on background complexity
5. Provide clear reasoning for your decisions

## EXAMPLE ANALYSIS FLOW
1. "The image shows a strawberry latte on a white marble table with soft natural lighting"
2. "Upper 20% of image is empty space (white background) - good for text"
3. "Background is light (brightness ~240) - use dark text (RGB: 30, 30, 30)"
4. "This is a cafe product - use NanumGothicBold for friendly, readable style"
5. "Background is simple - only need subtle shadow for depth"

Now analyze the image and call the `apply_text_overlay` function.
"""

    return prompt


# ==============================================================================
# Validation Helpers
# ==============================================================================

def validate_layout_spec(layout_spec: Dict[str, Any]) -> bool:
    """
    GPT-4V가 반환한 레이아웃 명세 검증

    Args:
        layout_spec: GPT-4V tool call 결과

    Returns:
        bool: 유효하면 True

    Raises:
        ValueError: 필수 필드 누락 또는 값 범위 오류
    """
    if "layers" not in layout_spec:
        raise ValueError("Missing 'layers' field in layout spec")

    layers = layout_spec["layers"]
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("'layers' must be a non-empty list")

    for i, layer in enumerate(layers):
        # 필수 필드 체크
        required_fields = ["text", "position", "font", "color"]
        for field in required_fields:
            if field not in layer:
                raise ValueError(f"Layer {i}: Missing required field '{field}'")

        # 위치 범위 체크
        pos = layer["position"]
        if not (0.0 <= pos["x"] <= 1.0 and 0.0 <= pos["y"] <= 1.0):
            raise ValueError(f"Layer {i}: Position coordinates must be in range [0.0, 1.0]")

        # 폰트 크기 체크
        font_size = layer["font"]["size"]
        if not (20 <= font_size <= 200):
            raise ValueError(f"Layer {i}: Font size must be in range [20, 200]")

        # 색상 범위 체크
        color = layer["color"]
        for channel in ["r", "g", "b"]:
            if not (0 <= color[channel] <= 255):
                raise ValueError(f"Layer {i}: Color channel '{channel}' must be in range [0, 255]")

        if not (0.0 <= color["a"] <= 1.0):
            raise ValueError(f"Layer {i}: Alpha channel must be in range [0.0, 1.0]")

    return True


def get_default_layout(text_data: Dict[str, str], image_size: tuple) -> Dict[str, Any]:
    """
    Fallback: GPT-4V 호출 실패 시 기본 레이아웃 생성

    Args:
        text_data: 오버레이할 텍스트
        image_size: (width, height)

    Returns:
        dict: 기본 레이아웃 명세
    """
    layers = []

    # 텍스트를 순서대로 상단에 배치 (간단한 폴백)
    y_positions = [0.15, 0.85]  # 상단, 하단

    for i, (key, text) in enumerate(text_data.items()):
        if i >= len(y_positions):
            break

        layer = {
            "text": text,
            "position": {
                "x": 0.5,
                "y": y_positions[i],
                "anchor": "center"
            },
            "font": {
                "family": "NanumGothicBold",
                "size": 80 if i == 0 else 40  # 첫 번째가 제일 크게
            },
            "color": {
                "r": 255,
                "g": 255,
                "b": 255,
                "a": 1.0
            },
            "effects": {
                "stroke": {
                    "enabled": True,
                    "width": 3,
                    "color": {"r": 0, "g": 0, "b": 0, "a": 1.0}
                },
                "shadow": {
                    "enabled": True,
                    "offset_x": 2,
                    "offset_y": 2,
                    "blur": 4,
                    "color": {"r": 0, "g": 0, "b": 0, "a": 0.6}
                },
                "background_box": {
                    "enabled": False
                }
            },
            "reasoning": "Fallback layout - GPT-4V analysis unavailable"
        }
        layers.append(layer)

    return {"layers": layers}
