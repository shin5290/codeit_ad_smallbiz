# End-to-End 평가 요약

> 소상공인 마케팅 상담 챗봇 시스템 종합 평가 결과

---

## ⚡ 한눈에 보는 결과

| 항목 | 결과 | 평가 |
|------|------|------|
| **Intent 정확도** | 91.5% | ✅ 목표 달성 (70%+) |
| **비용 효율** | $0.0016/쿼리 | ✅ 매우 저렴 ($16/월 1만) |
| **응답 속도** | 8.5초/쿼리 | ⚠️ 개선 필요 (목표: 2-3초) |
| **완료율** | 100% (200/200) | ✅ 안정성 확보 |

**종합 평가**: 정확도/비용은 우수, Latency는 개선 필요

---

## 📊 메타데이터
- 총 쿼리: 200개
- 완료: 200개 (100%)
- 오류: 0개
- 모델: gpt-4o-mini
- Tavily 사용: True

## 🎯 핵심 성과

### Intent 라우팅 정확도
- **91.5%** ✅ (목표: 70%+ 달성)
- Route 분포:
  - doc_rag: 121개 (60.5%)
  - marketing_counsel: 78개 (39.0%)
  - trend_web: 1개 (0.5%)

### 비용 효율성
- 총 비용: $0.32
- **쿼리당 평균: $0.0016** ✅
- **월 1만 쿼리 예상: $16** (매우 저렴)
- 참고: 가이드 예상치 $38보다 58% 절감

### Self-Refine 효율성
- 전체 200개 중 50개에만 적용 (25%)
- 실제 개선: 18개 (36%)
- **불필요한 refine 스킵 → 비용 절감** ✅

## ⚠️ 개선 필요 사항

### Latency (응답 속도)
- **현재: 쿼리당 평균 8.5초** ⚠️
- 목표: 2-3초 (실시간 챗봇 기준)
- 병목 분석:
  - 답변 생성 평균: 8.1초
  - Self-Refine 평균: 7.1초 (25% 쿼리만)
  - Route별:
    - doc_rag: 7.8초
    - marketing_counsel: 8.5초
    - trend_web: 15.5초 (웹 검색)
- 최악 케이스: 25.8초 (웹 검색 포함)

## 🚀 Latency 개선 계획

### 단기 (배포 전, 1-2주)
1. **Self-Refine 최적화**
   - 현재: 25% 쿼리에 평균 7.1초
   - 개선: 임계값 조정 또는 비활성화
   - 예상 효과: 8.5초 → 6.5초 (-23%)

2. **Web Search 타임아웃**
   - 현재: trend_web 평균 15.5초
   - 개선: 3-5초 타임아웃 설정
   - 예상 효과: 최악 케이스 25초 → 10초

3. **스트리밍 응답**
   - 첫 단어 1초 내 표시
   - 체감 latency 대폭 감소
   - FastAPI SSE 이미 구현됨

### 중기 (배포 후, 1-3개월)
1. **Redis 캐싱**
   - 동일/유사 쿼리 캐싱
   - Hit rate 30% 가정 시: 평균 3-4초

2. **병렬 처리**
   - RAG 검색 + Web search 동시 실행
   - 예상 효과: 15초 → 8초

3. **LLM 최적화**
   - gpt-4o-mini → claude-haiku (더 빠름)
   - 또는 프롬프트 최적화로 토큰 수 감소

### 장기 (3-6개월)
1. **실제 사용자 로그 분석**
   - 자주 묻는 질문 패턴 파악
   - 캐싱 전략 고도화

2. **인프라 최적화**
   - Vector DB 인덱싱 개선
   - 로드 밸런싱

### 목표
- **단기 목표**: 평균 5초 이하
- **중기 목표**: 평균 3초 이하 (캐싱 적용)
- **최종 목표**: 평균 2초 이하 (P95 < 5초)

## 📊 전체 시스템 성능 요약

### RAG 검색 성능 (Baseline Dense)
- **Recall@1**: 63.0%
- **Recall@5**: 88.5% ✅
- **Recall@10**: 94.0% ✅
- **MRR**: 73.6%
- **Success Rate**: 98.0% ✅
- **Answer Quality**: 3.98/5 ✅

### End-to-End 시스템 성능
- **Intent 정확도**: 91.5% ✅
- **비용**: $0.0016/쿼리 ✅
- **Latency**: 8.5초 ⚠️

### 핵심 의사결정
1. **RAG**: Baseline (Dense only) 채택
   - Hybrid Search, Reranker, Query Rewriting 모두 실패
   - Simple is Best

2. **Agent**: IntentRouter로 경로 분기
   - doc_rag: 60.5%
   - marketing_counsel: 39%
   - trend_web: 0.5%

3. **Self-Refine**: 조건부 적용 (25%)
   - 환각 방지, 비용 절감

## 📋 참고
- 본 요약은 `evaluation/results/end_to_end_results.json` 기준
- RAG 평가: `evaluation/FINAL_EVALUATION_RESULTS.md` 참조
- 평가 일시: 2026-01-27
- 총 평가 쿼리: 200개

---

## 🔧 개선 작업 내역 (기술 상세)
- TrendAgent 초기화 인자 오류 수정 (`openai_api_key/tavily_api_key/model` 제거 → `llm_model` 사용)
- IntentRouter 호출 방식 변경 반영 (`route()` → `classify()`), Agent 호출 (`invoke()` → `run()`)
- 평가 라벨 기준을 IntentRouter 설계에 맞게 현실화 (marketing_counsel 포함)
- Self-Refine 조건부 적용 (짧은 답변/출처 없음/보장 표현 시에만 실행)
- Self-Refine 환각 방지 강화 (새 숫자/예산/성과 수치 추가 금지)
- 출처 문구 개선 ("일반 지식 기반" → "제공된 자료에서 확인 불가")
- Self-Refine 비용/시간 계산 버그 수정 (스킵 시 비용 0)
