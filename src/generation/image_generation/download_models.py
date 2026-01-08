"""
SDXL 모델 다운로드 스크립트
/opt/ai-models/sdxl에 필요한 모델들을 다운로드

사용법:
  cd /opt/ai-models/sdxl
  python download_models.py
"""

from huggingface_hub import snapshot_download
import os
from pathlib import Path

# 모델 저장 경로
MODELS_DIR = Path("/opt/ai-models/sdxl")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 다운로드할 모델 목록
MODELS = {
    "SG161222/RealVisXL_V4.0": "Ultra Realistic 스타일",
    "cagliostrolab/animagine-xl-3.1": "Anime 스타일",
    "John6666/bss-equinox-il-semi-realistic-model-v25-sdxl": "Semi Realistic 스타일",
    "madebyollin/sdxl-vae-fp16-fix": "개선된 VAE (공통)",
}

# ControlNet 모델 (선택사항)
CONTROLNET_MODELS = {
    "diffusers/controlnet-canny-sdxl-1.0": "Canny ControlNet",
    "diffusers/controlnet-depth-sdxl-1.0": "Depth ControlNet",
    "thibaud/controlnet-openpose-sdxl-1.0": "Openpose ControlNet",
}


def download_model(repo_id: str, description: str):
    """HuggingFace에서 모델 다운로드"""
    model_name = repo_id.replace("/", "--")
    local_path = MODELS_DIR / model_name

    if local_path.exists():
        print(f"✓ 이미 존재: {model_name}")
        return

    print(f"\n{'='*60}")
    print(f"다운로드 시작: {description}")
    print(f"모델: {repo_id}")
    print(f"저장 경로: {local_path}")
    print(f"{'='*60}\n")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # 불필요한 파일 제외
        )
        print(f"✅ 완료: {model_name}\n")
    except Exception as e:
        print(f"❌ 실패: {e}\n")


def main():
    """메인 함수"""
    print("\n" + "="*60)
    print("SDXL 모델 다운로드 스크립트")
    print("="*60)
    print(f"저장 위치: {MODELS_DIR}")
    print(f"예상 용량: 약 25-30GB (모든 모델 포함)")
    print("="*60 + "\n")

    # 디스크 공간 확인
    import shutil
    total, used, free = shutil.disk_usage("/opt")
    free_gb = free // (2**30)
    print(f"남은 디스크 공간: {free_gb}GB")

    if free_gb < 35:
        print("⚠️  경고: 디스크 공간이 부족할 수 있습니다 (35GB 이상 권장)")
        response = input("계속하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("다운로드 취소됨")
            return

    print("\n[1/2] 필수 모델 다운로드 시작...\n")

    # 필수 모델 다운로드
    for repo_id, description in MODELS.items():
        download_model(repo_id, description)

    # ControlNet 모델 다운로드 (선택)
    print("\n" + "="*60)
    print("ControlNet 모델 다운로드 (선택사항)")
    print("I2I 기능을 사용하려면 필요합니다")
    print("="*60 + "\n")

    response = input("ControlNet 모델도 다운로드하시겠습니까? (y/n): ")

    if response.lower() == 'y':
        print("\n[2/2] ControlNet 모델 다운로드 시작...\n")
        for repo_id, description in CONTROLNET_MODELS.items():
            download_model(repo_id, description)
    else:
        print("ControlNet 모델 다운로드 건너뜀")

    # 완료 메시지
    print("\n" + "="*60)
    print("✅ 모든 다운로드 완료!")
    print("="*60)

    # 다운로드된 모델 확인
    print("\n다운로드된 모델:")
    for item in MODELS_DIR.iterdir():
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            print(f"  - {item.name}: {size_gb:.1f}GB")

    # 총 용량
    import subprocess
    result = subprocess.run(['du', '-sh', str(MODELS_DIR)], capture_output=True, text=True)
    print(f"\n총 용량: {result.stdout.split()[0]}")

    print("\n다음 단계:")
    print("  1. 코드에서 모델 경로 확인:")
    print("     MODELS_DIR = /opt/ai-models/sdxl")
    print("  2. 이미지 생성 테스트:")
    print("     python test_workflow.py")


if __name__ == "__main__":
    main()
