# src/utils/config.py의 역할

`src/utils/config.py`는 프로젝트 전체의 설정값을 중앙에서 관리하는 핵심 파일입니다. 팀 프로젝트에서 이 파일이 수행하는 주요 역할은 다음과 같습니다.

## 1. 중앙 집중식 설정 관리

각 팀원이 개발하는 기능 모듈들이 공통으로 사용하는 설정값을 한 곳에서 관리합니다. 예를 들어 데이터베이스 연결 정보, API 엔드포인트, 파일 경로, 로깅 레벨 등을 이 파일에 정의하면, 팀원들은 자신의 모듈에서 이 설정을 import해서 사용할 수 있습니다.

## 2. 환경별 설정 분리

개발(development), 테스트(testing), 운영(production) 환경에 따라 다른 설정값을 사용할 수 있도록 관리합니다. 환경 변수를 읽어서 적절한 설정을 로드하는 방식으로 구현되는 경우가 많습니다.

## 3. 설정 변경의 용이성

설정값이 여러 파일에 흩어져 있으면 수정할 때 모든 파일을 찾아다녀야 하지만, config.py에 모아두면 한 번의 수정으로 전체 프로젝트에 반영됩니다. 예를 들어 데이터베이스 포트 번호가 바뀌면 이 파일 하나만 수정하면 됩니다.

## 4. 코드의 재사용성과 유지보수성 향상

하드코딩된 값들을 config 파일로 분리하면 코드가 더 깔끔해지고, 나중에 설정을 변경하거나 다른 프로젝트에서 재사용하기도 쉬워집니다.

## 5. 보안 강화

API 키, 비밀번호 같은 민감한 정보를 환경 변수나 별도의 보안 파일에서 읽어오도록 config.py에서 처리할 수 있습니다. 이렇게 하면 민감한 정보가 코드 저장소에 직접 노출되지 않습니다.

## 전형적인 사용 예시

```python
# src/utils/config.py
import os

class Config:
    # 데이터베이스 설정
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    
    # API 설정
    API_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # 파일 경로
    DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
    LOG_DIR = os.path.join(os.path.dirname(__file__), '../../logs')
```

팀원들은 자신의 모듈에서 `from src.utils.config import Config`로 이 설정들을 가져와 사용하게 됩니다.

# config.py 구성 순서에 대한 권장사항

일반적으로 **역할 군(기능별) 분리**가 더 이해하기 쉽고 유지보수하기 좋습니다. 로직 처리 순서는 실제 코드 실행 파일(main.py 등)에서 고려할 사항이고, config.py는 "설정값 저장소"이기 때문에 **찾기 쉽고 관리하기 쉬운 구조**가 우선입니다.

## 권장하는 config.py 구성 순서

### 1. 기본 설정 (가장 위)
- 프로젝트 기본 정보
- 환경 변수 (개발/테스트/운영)
- 루트 경로

### 2. 외부 연결 설정
- 데이터베이스
- API 엔드포인트
- 외부 서비스 (AWS, Redis 등)

### 3. 경로 설정
- 데이터 디렉토리
- 로그 디렉토리
- 캐시 디렉토리
- 임시 파일 경로

### 4. 기능별 설정
- 로깅 설정
- 인증/보안 설정
- 데이터 처리 설정
- UI/프론트엔드 설정

### 5. 상수 및 제약사항
- 타임아웃
- 재시도 횟수
- 페이지네이션 크기
- 파일 크기 제한

## 실전 예시 코드

```python
# src/utils/config.py
import os
from pathlib import Path

# ============================================================
# 1. 기본 설정
# ============================================================
class BaseConfig:
    """프로젝트 기본 설정"""
    # 프로젝트 정보
    PROJECT_NAME = "MyTeamProject"
    VERSION = "1.0.0"
    
    # 환경 설정
    ENV = os.getenv('ENVIRONMENT', 'development')  # development, testing, production
    DEBUG = ENV == 'development'
    
    # 루트 경로
    BASE_DIR = Path(__file__).parent.parent.parent
    SRC_DIR = BASE_DIR / 'src'


# ============================================================
# 2. 외부 연결 설정
# ============================================================
class DatabaseConfig:
    """데이터베이스 연결 설정"""
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME', 'myproject_db')
    DB_USER = os.getenv('DB_USER', 'admin')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    # 연결 풀 설정
    DB_POOL_SIZE = 10
    DB_MAX_OVERFLOW = 20


class APIConfig:
    """외부 API 설정"""
    # 외부 API 엔드포인트
    EXTERNAL_API_URL = os.getenv('EXTERNAL_API_URL', 'https://api.example.com')
    API_KEY = os.getenv('API_KEY', '')
    
    # API 호출 설정
    API_TIMEOUT = 30
    API_MAX_RETRIES = 3


# ============================================================
# 3. 경로 설정
# ============================================================
class PathConfig:
    """프로젝트 내 경로 설정"""
    # 데이터 관련 경로
    DATA_DIR = BaseConfig.BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    
    # 로그 경로
    LOG_DIR = BaseConfig.BASE_DIR / 'logs'
    
    # 캐시 및 임시 파일
    CACHE_DIR = BaseConfig.BASE_DIR / 'cache'
    TEMP_DIR = BaseConfig.BASE_DIR / 'temp'
    
    # 출력 파일
    OUTPUT_DIR = BaseConfig.BASE_DIR / 'output'
    REPORT_DIR = OUTPUT_DIR / 'reports'
    
    # 디렉토리 자동 생성
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리들을 자동으로 생성"""
        for dir_path in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
                        cls.LOG_DIR, cls.CACHE_DIR, cls.TEMP_DIR,
                        cls.OUTPUT_DIR, cls.REPORT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. 기능별 설정
# ============================================================
class LoggingConfig:
    """로깅 설정"""
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = PathConfig.LOG_DIR / 'app.log'
    
    # 로그 파일 로테이션
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5


class SecurityConfig:
    """보안 및 인증 설정"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    JWT_EXPIRATION_HOURS = 24
    
    # 비밀번호 정책
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_REQUIRE_SPECIAL_CHAR = True


class DataProcessingConfig:
    """데이터 처리 관련 설정"""
    # 배치 처리
    BATCH_SIZE = 1000
    MAX_WORKERS = 4
    
    # 데이터 검증
    ALLOW_NULL_VALUES = False
    DATA_ENCODING = 'utf-8'
    
    # 파일 포맷
    CSV_DELIMITER = ','
    DATE_FORMAT = '%Y-%m-%d'


# ============================================================
# 5. 상수 및 제약사항
# ============================================================
class ConstraintsConfig:
    """시스템 제약사항 및 상수"""
    # 타임아웃 설정
    REQUEST_TIMEOUT = 30  # seconds
    TASK_TIMEOUT = 300  # seconds
    
    # 재시도 설정
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1  # seconds
    
    # 페이지네이션
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    
    # 파일 업로드
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.txt', '.csv', '.json', '.xlsx'}


# ============================================================
# 통합 Config 클래스 (선택사항)
# ============================================================
class Config(BaseConfig, DatabaseConfig, APIConfig, PathConfig,
            LoggingConfig, SecurityConfig, DataProcessingConfig,
            ConstraintsConfig):
    """
    모든 설정을 통합한 Config 클래스
    
    사용법:
        from src.utils.config import Config
        
        print(Config.PROJECT_NAME)
        print(Config.DB_HOST)
    """
    pass


# ============================================================
# 환경별 설정 오버라이드 (선택사항)
# ============================================================
class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """운영 환경 설정"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """테스트 환경 설정"""
    TESTING = True
    DB_NAME = 'test_db'


# 환경에 따른 Config 선택
config_by_env = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

# 현재 환경의 설정 가져오기
current_config = config_by_env.get(BaseConfig.ENV, DevelopmentConfig)
```

## 이런 구조의 장점

1. **찾기 쉬움**: "데이터베이스 설정 어디 있지?" → DatabaseConfig 섹션으로 바로 이동
2. **수정 용이**: 관련된 설정들이 한 곳에 모여있어 함께 수정 가능
3. **팀 협업**: 각자 담당 기능에 해당하는 섹션만 수정하면 되므로 충돌 최소화
4. **확장 가능**: 새로운 기능 추가 시 해당 섹션에 클래스만 추가하면 됨
5. **주석과 구분선**: 섹션별로 명확히 구분되어 가독성이 높음

## 추가 팁

- **주석을 적극 활용**: 각 설정값이 무엇을 의미하는지 간단히 설명
- **기본값 제공**: `os.getenv('KEY', 'default_value')` 형태로 기본값 설정
- **타입 명시**: Python 3.6+ 사용 시 타입 힌트 추가 권장
- **문서화**: README에 config.py 사용법과 필수 환경변수 목록 작성

이런 구조로 작성하면 팀원 누구나 쉽게 이해하고 사용할 수 있습니다!

# config.py 설정 통합 시 주의사항과 해결 방법

네, 기본적으로는 가능하지만 **몇 가지 중요한 사항**을 확인해야 합니다. 단순히 import 경로만 바꾸면 문제가 생길 수 있습니다.

## 발생 가능한 문제들

### 1. 상대 경로 문제
각 디렉토리의 임시 config.py가 **상대 경로**를 사용했다면 경로가 깨집니다.

```python
# 예: src/feature_a/config.py 에서
DATA_DIR = './data'  # 현재 디렉토리 기준
LOG_FILE = '../logs/app.log'  # 상위 디렉토리 기준

# src/utils/config.py로 이동하면
# 기준점이 바뀌어서 경로가 틀어집니다!
```

### 2. 같은 이름의 설정 변수 충돌
여러 담당자가 같은 변수명을 사용했다면 충돌이 발생합니다.

```python
# feature_a/config.py
BATCH_SIZE = 100

# feature_b/config.py
BATCH_SIZE = 500  # 충돌!
```

### 3. 파일 참조 경로
config.py 내부에서 다른 파일을 참조하는 경우 문제가 생깁니다.

```python
# feature_a/config.py
with open('settings.json', 'r') as f:  # 같은 폴더의 settings.json
    settings = json.load(f)
```

## 안전한 통합 방법

### 단계별 가이드

```python
# ============================================================
# Step 1: 기존 임시 config.py 분석
# ============================================================

# 예: src/feature_a/config.py (기존)
# 상대 경로 사용 예시
DATA_DIR = './data'
MODEL_PATH = '../models/model.pkl'
BATCH_SIZE = 100

# 예: src/feature_b/config.py (기존)
# 상대 경로 사용 예시
OUTPUT_DIR = './output'
BATCH_SIZE = 500  # feature_a와 충돌!
```

```python
# ============================================================
# Step 2: src/utils/config.py 에 통합 (수정 버전)
# ============================================================
import os
from pathlib import Path

class BaseConfig:
    """프로젝트 기본 설정"""
    # 절대 경로로 기준점 설정
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 프로젝트 루트
    SRC_DIR = BASE_DIR / 'src'


# 기능별로 클래스를 분리하여 충돌 방지
class FeatureAConfig:
    """Feature A 전용 설정"""
    # 상대 경로 → 절대 경로로 변환
    DATA_DIR = BaseConfig.SRC_DIR / 'feature_a' / 'data'
    MODEL_PATH = BaseConfig.BASE_DIR / 'models' / 'model.pkl'
    BATCH_SIZE = 100  # Feature A용 배치 크기


class FeatureBConfig:
    """Feature B 전용 설정"""
    # 상대 경로 → 절대 경로로 변환
    OUTPUT_DIR = BaseConfig.SRC_DIR / 'feature_b' / 'output'
    BATCH_SIZE = 500  # Feature B용 배치 크기 (충돌 해결)


# 공통 설정
class CommonConfig:
    """모든 기능이 공유하는 설정"""
    LOG_DIR = BaseConfig.BASE_DIR / 'logs'
    DEBUG = True
```

```python
# ============================================================
# Step 3: 기존 코드에서 import 수정
# ============================================================

# src/feature_a/main.py (수정 전)
from config import DATA_DIR, BATCH_SIZE

def process_data():
    print(f"Loading from {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")


# src/feature_a/main.py (수정 후)
from src.utils.config import FeatureAConfig

def process_data():
    print(f"Loading from {FeatureAConfig.DATA_DIR}")
    print(f"Batch size: {FeatureAConfig.BATCH_SIZE}")
```

## 실전 통합 체크리스트

### ✅ 통합 전 확인사항

```python
# 1. 모든 임시 config.py 파일 찾기
# 터미널에서 실행:
# find ./src -name "config.py" -type f

# 2. 각 config.py의 내용 비교
#    - 변수명 중복 체크
#    - 경로 방식 확인 (상대/절대)
#    - 외부 파일 참조 확인
```

### ✅ 통합 후 검증 코드

```python
# src/utils/config_validator.py
"""config.py 통합 후 검증용 스크립트"""

from pathlib import Path
from src.utils.config import BaseConfig, FeatureAConfig, FeatureBConfig

def validate_paths():
    """모든 경로가 올바르게 설정되었는지 확인"""
    print("=== 경로 검증 ===")
    
    configs = [
        ("BASE_DIR", BaseConfig.BASE_DIR),
        ("FeatureA DATA_DIR", FeatureAConfig.DATA_DIR),
        ("FeatureB OUTPUT_DIR", FeatureBConfig.OUTPUT_DIR),
    ]
    
    for name, path in configs:
        exists = path.exists() if hasattr(path, 'exists') else Path(path).exists()
        status = "✓" if exists else "✗ (생성 필요)"
        print(f"{name}: {path} {status}")


def validate_no_conflicts():
    """설정 충돌 확인"""
    print("\n=== 충돌 검증 ===")
    
    # 같은 용도의 설정값들이 제대로 분리되었는지 확인
    print(f"FeatureA BATCH_SIZE: {FeatureAConfig.BATCH_SIZE}")
    print(f"FeatureB BATCH_SIZE: {FeatureBConfig.BATCH_SIZE}")
    
    if FeatureAConfig.BATCH_SIZE != FeatureBConfig.BATCH_SIZE:
        print("✓ 배치 크기가 기능별로 분리됨")
    else:
        print("⚠ 배치 크기가 동일함 - 의도된 것인지 확인 필요")


def test_import_from_features():
    """각 feature에서 import가 제대로 되는지 테스트"""
    print("\n=== Import 테스트 ===")
    
    try:
        # Feature A에서 사용하는 방식
        from src.utils.config import FeatureAConfig
        print(f"✓ FeatureAConfig import 성공")
        print(f"  DATA_DIR: {FeatureAConfig.DATA_DIR}")
        
        # Feature B에서 사용하는 방식
        from src.utils.config import FeatureBConfig
        print(f"✓ FeatureBConfig import 성공")
        print(f"  OUTPUT_DIR: {FeatureBConfig.OUTPUT_DIR}")
        
    except ImportError as e:
        print(f"✗ Import 실패: {e}")


if __name__ == "__main__":
    validate_paths()
    validate_no_conflicts()
    test_import_from_features()
```

## 마이그레이션 스크립트 예시

```python
# migrate_config.py
"""
기존 config.py들을 src/utils/config.py로 통합하는 헬퍼 스크립트
"""

import re
from pathlib import Path

def find_all_configs():
    """프로젝트 내 모든 config.py 찾기"""
    src_dir = Path('./src')
    config_files = list(src_dir.rglob('config.py'))
    
    # src/utils/config.py는 제외
    config_files = [f for f in config_files if 'utils' not in f.parts]
    
    return config_files


def analyze_config_file(filepath):
    """config.py 파일 분석"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 변수 찾기
    variables = re.findall(r'^([A-Z_]+)\s*=', content, re.MULTILINE)
    
    # 상대 경로 사용 여부
    has_relative_paths = bool(re.search(r'[\'"]\.\.?/', content))
    
    return {
        'filepath': filepath,
        'variables': variables,
        'has_relative_paths': has_relative_paths,
        'content': content
    }


def suggest_migration(config_info):
    """마이그레이션 제안"""
    print(f"\n파일: {config_info['filepath']}")
    print(f"변수: {', '.join(config_info['variables'])}")
    print(f"상대 경로 사용: {'예' if config_info['has_relative_paths'] else '아니오'}")
    
    if config_info['has_relative_paths']:
        print("⚠ 경고: 상대 경로를 절대 경로로 수정해야 합니다!")


def main():
    """실행"""
    print("=== Config 마이그레이션 분석 ===\n")
    
    config_files = find_all_configs()
    print(f"발견된 config.py 파일: {len(config_files)}개\n")
    
    all_variables = set()
    
    for config_file in config_files:
        info = analyze_config_file(config_file)
        suggest_migration(info)
        all_variables.update(info['variables'])
    
    print(f"\n=== 전체 변수 목록 ===")
    print(f"총 {len(all_variables)}개: {', '.join(sorted(all_variables))}")
    
    # 중복 변수 확인
    from collections import Counter
    all_vars_list = []
    for config_file in config_files:
        info = analyze_config_file(config_file)
        all_vars_list.extend(info['variables'])
    
    duplicates = [var for var, count in Counter(all_vars_list).items() if count > 1]
    
    if duplicates:
        print(f"\n⚠ 중복된 변수명: {', '.join(duplicates)}")
        print("→ 클래스로 분리하거나 접두사를 추가하세요!")


if __name__ == "__main__":
    main()
```

## 권장 통합 순서

1. **백업 생성**: 기존 config.py 파일들을 모두 백업
2. **분석**: 위 스크립트로 충돌 및 경로 문제 파악
3. **통합**: src/utils/config.py에 클래스별로 통합
4. **검증**: config_validator.py로 경로 확인
5. **단계적 수정**: 한 feature씩 import 경로 수정 및 테스트
6. **정리**: 기존 임시 config.py 파일 삭제

## 결론

단순히 `from config`를 `from src.utils.config`로 바꾸는 것만으로는 **부족**합니다. 특히 상대 경로와 변수명 충돌 문제를 반드시 해결해야 합니다. 위의 체크리스트와 검증 스크립트를 활용하면 안전하게 통합할 수 있습니다.