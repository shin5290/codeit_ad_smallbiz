"""
OpenAI API 연결 및 .env 파일 테스트
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드 (프로젝트 루트에서)
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

def test_api_connection():
    """OpenAI API 연결 테스트"""
    
    print("=" * 60)
    print("🔍 OpenAI API 연결 테스트")
    print("=" * 60)
    
    # 1. 환경 확인
    print(f"\n📁 프로젝트 루트: {project_root}")
    print(f"📁 .env 파일 위치: {env_path}")
    print(f"📁 .env 파일 존재: {env_path.exists()}")
    
    # 2. API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n❌ API 키를 찾을 수 없습니다!")
        print("   .env 파일 내용을 확인하세요.")
        return False
    
    print(f"\n✅ API 키 로드 성공")
    print(f"   키 앞 15자: {api_key[:15]}...")
    
    # 3. GPT API 호출 테스트
    print("\n🤖 GPT API 호출 테스트 중...")
    
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "안녕하세요! 간단히 인사해주세요."}
            ],
            max_tokens=50
        )
        
        answer = response.choices[0].message.content
        print(f"\n✅ GPT 응답 성공:")
        print(f"   💬 {answer}")
        
        print("\n" + "=" * 60)
        print("🎉 모든 테스트 통과! API 연결 정상 작동합니다.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ GPT 호출 실패:")
        print(f"   오류: {e}")
        print("\n💡 해결 방법:")
        print("   1. API 키가 올바른지 확인")
        print("   2. 인터넷 연결 확인")
        print("   3. OpenAI 계정 크레딧 확인")
        return False

if __name__ == "__main__":
    success = test_api_connection()
    sys.exit(0 if success else 1)