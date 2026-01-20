"""
Save Image Node
이미지를 파일로 저장하는 노드

역할:
- PIL Image를 JPEG로 저장
- 해시 기반 파일명 생성 (중복 방지)
- 2글자 서브디렉토리 구조 (파일 분산)
"""

from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import io

from PIL import Image
from .base import BaseNode


class SaveImageNode(BaseNode):
    """
    이미지를 파일로 저장하는 노드

    Inputs:
        - image (PIL.Image): 저장할 이미지

    Outputs:
        - image_path (str): 저장된 파일의 절대 경로
        - filename (str): 파일명 (해시값, 확장자 제외)

    Example:
        node = SaveImageNode(storage_dir=Path("./output"))
        result = node.execute({"image": pil_image})
        # result = {
        #     "image_path": "/path/to/ab/abc123...def.jpg",
        #     "filename": "abc123...def"
        # }
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        quality: int = 95,
        format: str = "JPEG"
    ):
        """
        Args:
            storage_dir: 저장 디렉토리 (None이면 기본값 사용)
            quality: JPEG 품질 (1-100)
            format: 이미지 포맷 (JPEG, PNG 등)
        """
        super().__init__("SaveImageNode")

        # 기본 저장 경로: /mnt/data/generated
        if storage_dir is None:
            storage_dir = Path("/mnt/data/generated")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.format = format

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        이미지를 파일로 저장

        Args:
            inputs: {
                "image": PIL.Image  # 필수
            }

        Returns:
            {
                "image_path": str,  # 저장된 파일의 절대 경로
                "filename": str     # 파일명 (해시값)
            }
        """
        image: Image.Image = inputs["image"]

        # JPEG는 RGBA를 지원하지 않으므로 RGB로 변환
        if self.format.upper() == "JPEG" and image.mode == "RGBA":
            # 흰색 배경에 알파 채널 합성
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # 알파 채널을 마스크로 사용
            image = rgb_image

        # 1. 이미지를 바이트로 변환 (해시 계산용)
        buffer = io.BytesIO()
        image.save(buffer, format=self.format, quality=self.quality)
        image_bytes = buffer.getvalue()

        # 2. 해시 기반 파일명 생성
        filename = hashlib.sha256(image_bytes).hexdigest()

        # 3. 서브디렉토리 생성 (처음 2글자로 분산)
        subdir = filename[:2]
        save_dir = self.storage_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)

        # 4. 파일 확장자 결정
        ext = "jpg" if self.format.upper() == "JPEG" else self.format.lower()
        save_path = save_dir / f"{filename}.{ext}"

        # 5. 저장
        image.save(save_path, format=self.format, quality=self.quality)

        return {
            "image_path": str(save_path.absolute()),
            "filename": filename
        }

    def get_required_inputs(self) -> list:
        return ["image"]

    def get_output_keys(self) -> list:
        return ["image_path", "filename"]
