"""
ZIT 모델 다운로드 스크립트 (최적화 버전)
/opt/ai-models/zit 경로에 필요한 모델과 특정 파일들을 다운로드합니다.
"""

from huggingface_hub import snapshot_download, hf_hub_download
import os
from pathlib import Path
import sys

# ==========================================
# 1. 설정 및 경로
# ==========================================
BASE_DIR = Path("src/generation/image_generation/models/zit")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# [A] 전체 저장소 다운로드 (베이스 모델용)
# 폴더 구조가 필요한 모델들입니다.
REPO_DOWNLOADS = {
    "Tongyi-MAI/Z-Image-Turbo": {
        "desc": "ZIT 베이스 모델 (전체)",
        "folder": "Z-Image-Turbo-Base"  # 로컬에 저장될 폴더명
    }
}

# [B] 단일 파일 다운로드 (ControlNet, LoRA용)
# 특정 .safetensors 파일 하나만 딱 집어서 다운로드합니다.
FILE_DOWNLOADS = [
    # ControlNet Union (사용자분이 픽한 8steps 최신 버전)
    {
        "repo_id": "bubbliiiing/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        "filename": "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors",
        "desc": "ControlNet Union 2.1 (8-Steps)",
        "subfolder": "controlnet" # 저장될 하위 폴더
    },
    # 예시 LoRA (ZIT 전용 LoRA가 있다면 여기에 추가)
    {
        "repo_id": "strangerzonehf/Anime-Z", 
        "filename": "Anime-Z.safetensors", # 실제 파일명을 정확히 알아야 합니다
        "desc": "Anime 스타일 LoRA",
        "subfolder": "lora"
    },
    {
        "repo_id": "falgasdev/ob-semi-realistic-portrait-painting", 
        "filename": "OB半写实肖像画2.0 OB Semi-Realistic Portraits z- image turbo(1).safetensors", # 실제 파일명을 정확히 알아야 합니다
        "desc": "Semi-Realistic 스타일 LoRA",
        "subfolder": "lora"
    }
]

# ==========================================
# 2. 다운로드 함수 정의
# ==========================================
def download_repo(repo_id, config):
    """저장소 전체(스냅샷) 다운로드"""
    local_path = BASE_DIR / config['folder']
    print(f"\n[REPO] 다운로드 시작: {config['desc']}")
    print(f" - 대상: {repo_id}")
    print(f" - 경로: {local_path}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.ckpt"], # 불필요 파일 제외
        )
        print(f"✅ 완료\n")
    except Exception as e:
        print(f"❌ 실패: {e}\n")

def download_file(config):
    """단일 파일 다운로드"""
    local_dir = BASE_DIR / config['subfolder']
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[FILE] 다운로드 시작: {config['desc']}")
    print(f" - 파일: {config['filename']}")
    print(f" - 경로: {local_dir}")

    try:
        hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ 완료\n")
    except Exception as e:
        print(f"❌ 실패: {e}\n")

# ==========================================
# 3. 메인 실행
# ==========================================
def main():
    print("\n" + "="*60)
    print("🚀 Z-Image Turbo 통합 다운로더")
    print("="*60)
    print(f"메인 저장 경로: {BASE_DIR}")
    
    # 1. 베이스 모델 다운로드
    print("\n[단계 1/2] 베이스 모델 다운로드")
    for repo_id, config in REPO_DOWNLOADS.items():
        download_repo(repo_id, config)

    # 2. ControlNet & LoRA 다운로드
    print("\n" + "="*60)
    print("ControlNet 및 LoRA (단일 파일)")
    print("="*60)
    
    if input("추가 파일들을 다운로드하시겠습니까? (y/n): ").lower() == 'y':
        print("\n[단계 2/2] 추가 파일 다운로드")
        for config in FILE_DOWNLOADS:
            download_file(config)
    
    # 마무리
    print("\n" + "="*60)
    print("🎉 모든 작업이 완료되었습니다.")
    print(f"총 용량 확인: du -sh {BASE_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()