"""
ControlNet Node for Image-to-Image Generation
제품 이미지의 구조를 유지하면서 스타일 변환
"""

from typing import Any, Dict, Optional, Literal
import torch
from PIL import Image
import numpy as np
from pathlib import Path

from .base import BaseNode
from ..config import model_config

# ControlNet 전처리 라이브러리
try:
    from controlnet_aux import CannyDetector, OpenposeDetector, MidasDetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    print("[WARNING] controlnet_aux not installed. Install with: pip install controlnet-aux")

# Diffusers ControlNet
try:
    from diffusers import ControlNetModel
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False
    print("[WARNING] diffusers ControlNet not available")


# 모델 캐싱 디렉토리
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class ControlNetPreprocessorNode(BaseNode):
    """
    ControlNet Preprocessor Node
    입력 이미지를 ControlNet이 사용할 수 있는 형태로 변환

    지원 타입:
    - canny: 윤곽선 추출 (제품 형태 유지에 최적)
    - depth: 깊이 맵 추출 (공간 구조 유지)
    - openpose: 포즈 추출 (사람 이미지용)
    """

    def __init__(
        self,
        control_type: Literal["canny", "depth", "openpose"] = "canny",
        node_name: Optional[str] = None
    ):
        """
        Args:
            control_type: ControlNet 타입
            node_name: 노드 이름
        """
        super().__init__(node_name)
        self.control_type = control_type
        self.processor = None

        if not CONTROLNET_AUX_AVAILABLE:
            raise ImportError(
                "controlnet_aux is required. Install with: pip install controlnet-aux"
            )

        self._load_processor()

    def _load_processor(self):
        """전처리 모델 로드"""
        print(f"[{self.node_name}] Loading {self.control_type} preprocessor...")

        if self.control_type == "canny":
            self.processor = CannyDetector()
        elif self.control_type == "depth":
            self.processor = MidasDetector.from_pretrained("lllyasviel/Annotators")
        elif self.control_type == "openpose":
            self.processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        else:
            raise ValueError(f"Unsupported control_type: {self.control_type}")

        print(f"[{self.node_name}] Preprocessor loaded successfully")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        이미지를 ControlNet용으로 전처리

        Args:
            inputs: {
                "image": PIL.Image - 입력 이미지 (제품 사진 등)
            }

        Returns:
            {
                "control_image": PIL.Image - 전처리된 이미지 (Canny edge 등)
                "control_type": str - ControlNet 타입
            }
        """
        image = inputs["image"]

        # PIL Image를 NumPy 배열로 변환 (일부 processor 요구사항)
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # 전처리 실행
        print(f"[{self.node_name}] Processing {self.control_type}...")

        if self.control_type == "canny":
            # Canny edge detection
            control_image = self.processor(image, low_threshold=100, high_threshold=200)
        elif self.control_type == "depth":
            # Depth estimation
            control_image = self.processor(image)
        elif self.control_type == "openpose":
            # Pose estimation
            control_image = self.processor(image)

        return {
            "control_image": control_image,
            "control_type": self.control_type
        }

    def get_required_inputs(self) -> list:
        return ["image"]

    def get_output_keys(self) -> list:
        return ["control_image", "control_type"]


class ControlNetLoaderNode(BaseNode):
    """
    ControlNet 모델 로더 노드
    SDXL 호환 ControlNet 모델 로드
    """

    def __init__(
        self,
        control_type: Literal["canny", "depth", "openpose"] = "canny",
        node_name: Optional[str] = None
    ):
        """
        Args:
            control_type: ControlNet 타입
            node_name: 노드 이름
        """
        super().__init__(node_name)
        self.control_type = control_type
        self.controlnet_model = None

        if not CONTROLNET_AVAILABLE:
            raise ImportError(
                "diffusers ControlNet not available. Update diffusers: pip install --upgrade diffusers"
            )

    def _get_controlnet_model_id(self) -> str:
        """ControlNet 모델 ID 반환"""
        # SDXL ControlNet 모델 매핑
        model_map = {
            "canny": "diffusers/controlnet-canny-sdxl-1.0",
            "depth": "diffusers/controlnet-depth-sdxl-1.0",
            "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
        }

        model_id = model_map.get(self.control_type)
        if not model_id:
            raise ValueError(f"Unsupported control_type: {self.control_type}")

        return model_id

    def _load_controlnet(self):
        """ControlNet 모델 로드"""
        model_id = self._get_controlnet_model_id()

        print(f"[{self.node_name}] Loading ControlNet: {model_id}")

        # 로컬 캐시 경로
        local_model_path = MODELS_DIR / f"controlnet-{self.control_type}-sdxl"

        # 모델 로드
        if local_model_path.exists():
            print(f"[{self.node_name}] Loading from local cache: {local_model_path}")
            self.controlnet_model = ControlNetModel.from_pretrained(
                str(local_model_path),
                torch_dtype=getattr(torch, model_config.DTYPE)
            )
        else:
            print(f"[{self.node_name}] Downloading from HuggingFace: {model_id}")
            self.controlnet_model = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=getattr(torch, model_config.DTYPE),
                cache_dir=str(MODELS_DIR)
            )
            # 로컬 저장
            self.controlnet_model.save_pretrained(str(local_model_path))

        print(f"[{self.node_name}] ControlNet loaded successfully")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        ControlNet 모델 로드 및 반환

        Args:
            inputs: {} (입력 불필요, 설정만 사용)

        Returns:
            {
                "controlnet": ControlNetModel - 로드된 ControlNet 모델
                "control_type": str - ControlNet 타입
            }
        """
        if self.controlnet_model is None:
            self._load_controlnet()

        return {
            "controlnet": self.controlnet_model,
            "control_type": self.control_type
        }

    def get_required_inputs(self) -> list:
        return []  # 입력 불필요

    def get_output_keys(self) -> list:
        return ["controlnet", "control_type"]

    def unload(self):
        """메모리 해제"""
        if self.controlnet_model is not None:
            del self.controlnet_model
            self.controlnet_model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[{self.node_name}] ControlNet unloaded")
