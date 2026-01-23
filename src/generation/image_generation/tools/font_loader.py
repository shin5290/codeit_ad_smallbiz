"""
Font Loader with Fallback Strategy
시스템 폰트 자동 검색 및 로딩

작성자: 이현석

GCP 서버 폰트 현황:
✅ /usr/share/fonts/truetype/nanum/ - Nanum 폰트 패밀리
✅ /usr/share/fonts/opentype/noto/   - Noto CJK 폰트
✅ /usr/share/fonts/truetype/dejavu/ - DejaVu (폴백)
"""

from pathlib import Path
from typing import Optional, Dict, List
from PIL import ImageFont

# ==============================================================================
# 폰트 경로 및 매핑
# ==============================================================================

# 시스템 폰트 직접 경로 매핑 (GCP 서버 기준)
FONT_PATHS = {
    # Nanum 계열
    "NanumGothic": "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "NanumGothicBold": "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "NanumBarunGothic": "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "NanumBarunGothicBold": "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
    "NanumMyeongjo": "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
    "NanumMyeongjoBold": "/usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf",

    # Noto CJK (TTC - TrueType Collection)
    "NotoSansKR": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "NotoSansKRBold": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "NotoSerifKR": "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "NotoSerifKRBold": "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",

    # 폴백
    "DejaVu": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}

# 커스텀 폰트 디렉토리 (선택)
CUSTOM_FONT_DIR = Path("/mnt/fonts")


class FontLoader:
    """
    폰트 로더: 시스템 폰트 직접 접근 + 커스텀 폰트 검색
    """

    def __init__(self):
        """초기화: 폰트 가용성 체크"""
        self._font_cache: Dict[str, Path] = {}
        self._scan_fonts()

    def _scan_fonts(self):
        """시스템 폰트 및 커스텀 폰트 스캔 (커스텀 폰트 우선)"""
        print("[FontLoader] Scanning fonts...")

        # 1. 시스템 폰트 먼저 로드 (fallback)
        for family, path_str in FONT_PATHS.items():
            path = Path(path_str)
            if path.exists():
                self._font_cache[family] = path
            else:
                print(f"⚠️ Font not found: {family} at {path}")

        # 2. 커스텀 폰트로 덮어쓰기 (/mnt/fonts 우선)
        if CUSTOM_FONT_DIR.exists():
            print(f"[FontLoader] Scanning custom fonts: {CUSTOM_FONT_DIR} (overrides system fonts)")
            self._scan_custom_fonts()

        # 로그 출력
        print(f"[FontLoader] Loaded {len(self._font_cache)} fonts:")
        for family in sorted(self._font_cache.keys()):
            print(f"  ✅ {family}")

    def _scan_custom_fonts(self):
        """커스텀 폰트 디렉토리 스캔 (TTF/OTF/TTC), 시스템 폰트 덮어쓰기"""
        for ext in ["*.ttf", "*.otf", "*.ttc"]:
            for font_file in CUSTOM_FONT_DIR.glob(ext):
                # 파일명을 family 이름으로 사용 (확장자 제거)
                family_name = font_file.stem

                # 덮어쓰기 여부 로그
                if family_name in self._font_cache:
                    print(f"  🔄 Override: {family_name} (custom overrides system)")
                else:
                    print(f"  ✅ Custom: {family_name}")

                # 무조건 커스텀 폰트로 덮어쓰기
                self._font_cache[family_name] = font_file

    def get_font_path(self, family: str) -> Optional[Path]:
        """
        폰트 패밀리 이름으로 경로 반환

        Args:
            family: 폰트 패밀리 이름 (예: "NanumGothicBold")

        Returns:
            Path: 폰트 파일 경로 (없으면 None)
        """
        return self._font_cache.get(family)

    def load_font(self, family: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
        """
        PIL ImageFont 객체 생성

        Args:
            family: 폰트 패밀리 이름
            size: 폰트 크기 (픽셀)
            index: TTC 파일 내 폰트 인덱스 (Noto CJK용)
                   0=Korean, 1=Japanese, 2=Simplified Chinese, 3=Traditional Chinese

        Returns:
            ImageFont.FreeTypeFont: PIL 폰트 객체
        """
        font_path = self.get_font_path(family)

        if font_path is None:
            # Fallback 체인
            print(f"⚠️ Font '{family}' not found, trying fallbacks")

            # 1. DejaVu 시도
            font_path = self.get_font_path("DejaVu")
            if font_path is None:
                # 2. 아무 한글 폰트나 사용
                korean_fonts = ["NanumGothic", "NanumBarunGothic", "NotoSansKR"]
                for fallback in korean_fonts:
                    font_path = self.get_font_path(fallback)
                    if font_path:
                        print(f"   Using fallback: {fallback}")
                        break

            if font_path is None:
                # 3. 최후의 수단: PIL 기본 폰트
                print(f"⚠️ No fonts available, using PIL default")
                return ImageFont.load_default()

        try:
            # TTC 파일 처리 (Noto CJK)
            if font_path.suffix == ".ttc":
                return ImageFont.truetype(str(font_path), size=size, index=index)
            else:
                return ImageFont.truetype(str(font_path), size=size)
        except Exception as e:
            print(f"⚠️ Failed to load font '{family}' from {font_path}: {e}")
            return ImageFont.load_default()

    def get_available_fonts(self) -> List[str]:
        """사용 가능한 폰트 패밀리 목록 반환"""
        return sorted(self._font_cache.keys())

    def is_font_available(self, family: str) -> bool:
        """특정 폰트가 사용 가능한지 확인"""
        return family in self._font_cache


# ==============================================================================
# 전역 인스턴스 (싱글톤)
# ==============================================================================

_GLOBAL_FONT_LOADER: Optional[FontLoader] = None


def get_font_loader() -> FontLoader:
    """전역 FontLoader 인스턴스 반환 (Lazy Initialization)"""
    global _GLOBAL_FONT_LOADER

    if _GLOBAL_FONT_LOADER is None:
        _GLOBAL_FONT_LOADER = FontLoader()

    return _GLOBAL_FONT_LOADER


# ==============================================================================
# 편의 함수
# ==============================================================================

def load_font(family: str, size: int) -> ImageFont.FreeTypeFont:
    """
    폰트 로드 (전역 FontLoader 사용)

    Args:
        family: 폰트 패밀리 이름
        size: 폰트 크기 (픽셀)

    Returns:
        ImageFont.FreeTypeFont: PIL 폰트 객체
    """
    loader = get_font_loader()
    return loader.load_font(family, size)


def get_available_fonts() -> List[str]:
    """사용 가능한 폰트 목록 반환"""
    loader = get_font_loader()
    return loader.get_available_fonts()


def check_font_availability():
    """
    폰트 가용성 체크 및 리포트 출력 (디버깅용)
    """
    loader = get_font_loader()

    print("\n" + "="*80)
    print("FONT AVAILABILITY REPORT")
    print("="*80)

    required_fonts = [
        "NanumGothic",
        "NanumGothicBold",
        "NanumMyeongjo",
        "NotoSansKR",
        "NotoSerifKR"
    ]

    for font_family in required_fonts:
        is_available = loader.is_font_available(font_family)
        status = "✅ Available" if is_available else "❌ Missing"

        if is_available:
            path = loader.get_font_path(font_family)
            print(f"{status:20} {font_family:20} → {path.name}")
        else:
            print(f"{status:20} {font_family:20}")

    print("\n" + "="*80)
    print("ALL AVAILABLE FONTS:")
    for font in loader.get_available_fonts():
        print(f"  - {font}")
    print("="*80 + "\n")


# ==============================================================================
# 메인 실행 (테스트)
# ==============================================================================

if __name__ == "__main__":
    # 폰트 가용성 체크
    check_font_availability()

    # 테스트: 폰트 로드
    print("\nTesting font loading...")

    test_fonts = ["NanumGothicBold", "NotoSansKR", "NonExistentFont"]
    for font_name in test_fonts:
        try:
            font = load_font(font_name, 48)
            print(f"✅ Loaded {font_name} at 48px")
        except Exception as e:
            print(f"❌ Failed to load {font_name}: {e}")
