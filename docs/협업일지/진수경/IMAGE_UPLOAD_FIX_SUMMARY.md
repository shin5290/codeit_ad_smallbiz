# 이미지 업로드 문제 해결 요약

## 문제 분석

사용자가 이미지를 업로드할 때 `/data/uploads` 폴더에 저장이 안 되고, DB 저장 및 이미지 생성 로직이 실행되지 않는 문제가 발생했습니다.

## 발견된 문제점

### 1. src/utils/image.py - save_uploaded_image() 함수 버그

**문제**:
- 56번째 줄에서 정상적으로 `return {"file_hash": ..., "file_directory": ...}` 반환
- 58번째 줄에 도달할 수 없는 `return None` 코드가 존재
- 예외 처리가 없어 에러 발생 시 원인 파악이 어려움
- 로깅이 없어 실행 흐름 추적 불가

**해결책**:
- 도달할 수 없는 `return None` 제거
- 모든 단계에 로깅 추가 (디버깅 용이)
- 예외 처리 추가 (try-except)
- 각 단계별로 상세한 정보 로깅

### 2. src/backend/services.py - ingest_user_message() 함수

**문제**:
- 이미지 처리 과정에 로깅이 없어 어느 단계에서 실패하는지 파악 불가
- 에러가 발생해도 조용히 실패할 가능성

**해결책**:
- 각 단계별 로깅 추가
- 세션 생성, 이미지 저장, DB 저장 각 단계별로 상세 로그 출력

### 3. main.py - /generate 엔드포인트

**문제**:
- 요청 수신 시 로깅이 없어 이미지가 제대로 전달되는지 확인 불가
- 에러 발생 시 스택 트레이스가 기록되지 않음

**해결책**:
- 요청 수신 시 이미지 정보 로깅
- 예외 발생 시 전체 스택 트레이스 로깅 (exc_info=True)

### 4. src/backend/services.py - generate_contents() 함수

**문제**:
- input_image가 전달되어도 로드 과정에서 실패할 가능성
- 로깅이 없어 reference_image 로드 성공 여부 확인 불가

**해결책**:
- input_image 로드 과정 로깅
- PIL Image 로드 성공/실패 여부 로깅
- 이미지 크기 등 상세 정보 로깅

## 수정된 파일 목록

1. **src/utils/image.py**
   - save_uploaded_image() 함수 수정
   - 로깅 추가 (import logging)
   - 예외 처리 추가
   - 도달 불가능한 코드 제거

2. **src/backend/services.py**
   - ingest_user_message() 함수 로깅 추가
   - generate_contents() 함수 로깅 추가

3. **main.py**
   - 로깅 설정 추가 (basicConfig)
   - /generate 엔드포인트 로깅 추가

## 워크플로우 (수정 후)

```
1. 사용자 요청 → main.py:/generate 엔드포인트
   └─ [로그] 요청 수신, 이미지 유무 확인

2. handle_generate_pipeline() 호출
   └─ ingest_user_message() 호출
      ├─ [로그] 세션 생성/확인
      ├─ save_uploaded_image() 호출
      │  ├─ [로그] 베이스 디렉토리 생성
      │  ├─ [로그] 이미지 데이터 읽기 (바이트 수)
      │  ├─ [로그] 해시 계산, 확장자 결정
      │  ├─ [로그] 서브디렉토리 생성
      │  ├─ [로그] 파일 저장 완료
      │  └─ return {"file_hash": ..., "file_directory": ...}
      ├─ [로그] 이미지 디스크 저장 완료
      ├─ process_db.save_image_from_hash() 호출
      │  └─ [로그] 이미지 DB 저장 완료
      └─ process_db.save_chat_message() 호출
         └─ [로그] 채팅 메시지 저장 완료

3. generate_contents() 호출
   ├─ [로그] generation_type, input_image 유무
   ├─ load_image_from_payload() 호출
   │  └─ [로그] reference image 로드 성공/실패, 크기
   └─ 이미지 생성 로직 실행
```

## 테스트 방법

### 1. 서버 실행
```bash
cd /home/spai0416/codeit_ad_smallbiz
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 이미지 업로드 테스트
프론트엔드에서 이미지 포함한 요청 전송 또는 curl 사용:

```bash
curl -X POST http://localhost:8000/generate \
  -F "input_text=카페 광고 생성" \
  -F "image=@test_image.jpg" \
  -F "generation_type=image" \
  -F "style=ultra_realistic" \
  -F "aspect_ratio=1:1"
```

### 3. 로그 확인
서버 터미널에서 다음과 같은 로그 확인:

```
[/generate] Request received: input_text=카페 광고 생성...
[/generate] Image details: filename=test_image.jpg, content_type=image/jpeg
ingest_user_message: session_id=xxx, user_id=xxx, has_image=True
save_uploaded_image: base_dir=/home/spai0416/codeit_ad_smallbiz/data/uploads
save_uploaded_image: read 12345 bytes from test_image.jpg
save_uploaded_image: file_hash=abc123..., ext=.jpg
save_uploaded_image: saved file to /home/spai0416/codeit_ad_smallbiz/data/uploads/ab/abc123.jpg
ingest_user_message: image saved to DB with id=1
generate_contents: generation_type=image, has_input_image=True
generate_contents: reference image loaded successfully: size=(1024, 768)
```

### 4. 파일 저장 확인
```bash
ls -la /home/spai0416/codeit_ad_smallbiz/data/uploads/
# 서브디렉토리가 생성되고 파일이 저장되었는지 확인
```

### 5. DB 확인
데이터베이스에서 ImageMatching 테이블 확인:
```sql
SELECT * FROM image_matching ORDER BY created_at DESC LIMIT 10;
SELECT * FROM chat_history WHERE image_id IS NOT NULL ORDER BY created_at DESC LIMIT 10;
```

## 추가 개선 사항 (선택)

1. **디렉토리 권한 확인**
   ```bash
   chmod 755 /home/spai0416/codeit_ad_smallbiz/data/uploads
   ```

2. **환경 변수로 업로드 경로 관리**
   - .env에 `UPLOAD_DIR` 추가
   - settings에 설정 추가

3. **이미지 검증 강화**
   - 파일 크기 제한
   - MIME 타입 검증
   - PIL로 이미지 유효성 검증

4. **에러 핸들링 개선**
   - 디스크 공간 부족 시 처리
   - 권한 에러 처리
   - 사용자에게 친절한 에러 메시지

## 예상되는 문제 및 해결책

### 문제 1: 여전히 이미지가 저장되지 않음
**원인**: 디렉토리 권한 문제
**해결**:
```bash
chmod -R 755 /home/spai0416/codeit_ad_smallbiz/data
chown -R spai0416:spai0416 /home/spai0416/codeit_ad_smallbiz/data
```

### 문제 2: 로그에 아무것도 출력되지 않음
**원인**: 로깅 레벨 설정 문제
**해결**: main.py에서 로깅 레벨을 DEBUG로 변경
```python
logging.basicConfig(level=logging.DEBUG)
```

### 문제 3: DB에 저장되지만 파일이 없음
**원인**: save_uploaded_image()에서 파일은 저장했지만 경로가 잘못됨
**해결**: 로그에서 file_directory 경로 확인하고 실제 파일 존재 여부 확인

### 문제 4: 이미지는 저장되는데 생성 로직이 실행 안 됨
**원인**: generate_contents()에서 input_image 로드 실패
**해결**:
- load_image_from_payload() 함수 확인
- file_directory 경로가 절대 경로인지 확인
- PIL Image.open() 에러 확인

## 결론

이번 수정으로 다음이 개선되었습니다:

1. ✅ 이미지 업로드 로직의 버그 수정
2. ✅ 전체 워크플로우에 상세한 로깅 추가
3. ✅ 예외 처리 강화
4. ✅ 디버깅 용이성 향상

이제 서버를 재시작하고 테스트하면 로그를 통해 정확히 어느 단계에서 문제가 발생하는지 파악할 수 있습니다.
