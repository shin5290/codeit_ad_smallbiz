"""
Text Layout Tools for GPT-4V
GPT-4V가 이미지를 분석하고 텍스트 레이아웃을 결정하기 위한 Tool Definition

작성자: 이현석
"""

from typing import Dict, List, Any

# ==============================================================================
# OpenAI Function Calling Tool Definition
# ==============================================================================

TEXT_OVERLAY_TOOL = {
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
                                        "enum": [
                                            "NanumGothic",      # 고딕체 (기본, 중성적)
                                            "NanumGothicBold",  # 고딕체 볼드 (강조)
                                            "NanumMyeongjo",    # 명조체 (고급스러운, 전통적)
                                            "NotoSansKR",       # 산세리프 (현대적, 깔끔)
                                            "NotoSerifKR"       # 세리프 (우아한, 고전적)
                                        ],
                                        "description": "폰트 패밀리. 이미지 분위기에 맞춰 선택 (캐주얼→고딕, 고급→명조/세리프)"
                                    },
                                    "size": {
                                        "type": "integer",
                                        "minimum": 20,
                                        "maximum": 200,
                                        "description": "폰트 크기 (픽셀). 제품명→60-100, 부가문구→30-50 권장"
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


# ==============================================================================
# GPT-4V Analysis Prompt Template
# ==============================================================================

def get_analysis_prompt(text_data: Dict[str, str], image_context: str = "") -> str:
    """
    GPT-4V 이미지 분석 프롬프트 생성

    Args:
        text_data: 오버레이할 텍스트 데이터
            예: {"product_name": "딸기라떼", "tagline": "신메뉴 출시"}
        image_context: 이미지 컨텍스트 (선택, 예: "카페 메뉴 광고")

    Returns:
        str: GPT-4V에게 전달할 프롬프트
    """

    # 텍스트 데이터를 읽기 쉽게 포맷팅
    text_items = "\n".join([f"- {key}: \"{value}\"" for key, value in text_data.items()])

    prompt = f"""You are an expert graphic designer specializing in advertising image composition.

## YOUR TASK
Analyze the provided advertising image and determine the optimal layout for overlaying the following text:

{text_items}

## IMAGE CONTEXT
{image_context if image_context else "General advertising image"}

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

### 3. FONT SELECTION
- **NanumGothic / NanumGothicBold**:
  - Use for: Casual, friendly, everyday products (cafes, restaurants, retail)
  - Style: Clean, neutral, highly readable
- **NanumMyeongjo**:
  - Use for: Traditional, elegant, premium products (luxury goods, traditional Korean)
  - Style: Serif, sophisticated, classic
- **NotoSansKR**:
  - Use for: Modern, tech-savvy, minimalist products (apps, services, fashion)
  - Style: Sans-serif, contemporary, clean
- **NotoSerifKR**:
  - Use for: Artistic, editorial, high-end products (magazines, galleries, premium services)
  - Style: Serif, refined, elegant

### 4. FONT SIZE HIERARCHY
- **Product Name / Main Text**: 80-140px (needs to be LARGE and immediately visible, DOMINANT presence)
- **Tagline / Secondary Text**: 40-70px (supporting information, but still clearly readable)
- **Fine Print / Details**: 25-40px (additional info, disclaimers)
- **CRITICAL**: Korean advertising text must be BOLD and LARGE - err on the larger side!

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

### 7. KEYWORD EMPHASIS (SPLIT LAYERS FOR STYLING)
- **Identify keywords to emphasize**: product names, brand names, numbers, key actions
- **Split into separate layers** if different styling is needed:
  - Example: "삼겹살엔 역시 소주" → Layer 1: "삼겹살엔 역시 " (regular) + Layer 2: "소주" (emphasized)
  - Position layers adjacent to each other (calculate x offset based on text width)
- **Emphasis techniques**:
  - Larger font size (1.5-2x the base layer)
  - Different font family (e.g., NanumMyeongjo for elegance, NanumGothicBold for impact)
  - Different color (brand color, contrasting color, gradient effect via background_box)
  - Stronger effects (thicker stroke, brighter color)

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
