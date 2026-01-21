# Prompt Engineering Report (SmallBiz RAG Assistant)

## 개요
프롬프트를 반복 개선하며 LLM-as-a-judge + 룰 기반 평가로 품질을 추적했습니다. 최종 스코어는 Specificity 8.50 / Evidence 9.00 / Structure 10.00 / Safety 10.00, 룰 위반 없음입니다.

## 실행 방법
1) 응답 생성: `python evaluation/build_responses.py --model gpt-4o --output evaluation/responses.jsonl`  
2) 평가 실행: `python evaluation/evaluate_prompts.py --inputs evaluation/responses.jsonl --model gpt-4o --summary`

## 개선 과정 (서술)
- 초기 상태: 숫자/근거 요구가 느슨해 Evidence가 낮고, 구조는 양호했으나 근거 밀도가 부족했습니다.
- 숫자·근거 의무화: 항목마다 숫자를 넣도록 했으나, 근거 표기가 느슨해 Evidence가 충분히 오르지 않았습니다.
- 숫자 상향 + 출처 최소 개수: 숫자 요구를 늘리고 출처 3개 이상을 요구했지만, 항목별 근거 연결이 약했습니다.
- 항목별 출처 부착 + 검색 k 확대: 각 bullet 끝에 `(출처: {제목}({지역}))`를 강제하고 retrieval k를 7로 올리자 Evidence와 Structure가 크게 상승해 최종 스코어를 달성했습니다.

## 점수 히스토리 (LLM-as-judge)
| Iteration | 주요 변경 | Specificity | Evidence | Structure | Safety | Rule Violations |
|-----------|-----------|-------------|----------|-----------|--------|-----------------|
| 1 | 베이스라인 | 7.67 | 6.33 | 9.00 | 9.33 | 0 |
| 2 | 숫자·근거 요구(느슨) | 7.50 | 6.00 | 9.00 | 9.50 | 0 |
| 3 | 숫자 상향, 출처 요구 | 8.25 | 6.75 | 9.25 | 9.25 | 0 |
| 4 | 항목별 출처 부착, k=7 | **8.50** | **9.00** | **10.00** | **10.00** | 0 |

## 효과가 컸던 변경
- 항목별 출처 태깅: 각 실행/아이디어 bullet 끝에 `(출처: {제목}({지역}))` 강제 → Evidence/Structure 개선.
- 검색 폭 확장: k=3→7로 사례 밀도 증가 → 근거 점수 상승.
- 숫자 하한선: 주요 항목에 숫자 2개 이상 요구 → Specificity 개선.
- 출처 중복 금지 + 최소 3개: 누락/중복 방지로 구조 일관성 확보.

## 현재 프롬프트 계약
- 항목당 숫자 2개 이상 + 근거 부착, 출처 섹션 bullet 3개 이상(중복 금지).
- 평점/별점 언급 금지, 상식 배제, 구형 채널(페이스북 등) 제한.
- 프랜차이즈 지양, 로컬 우선.

## 추가 아이디어 (선택)
- 섹션별 길이 상한을 두어 과도한 장황함 방지.
- 트렌드 답변에 날짜/링크 기반 출처를 더 많이 요구해 실시간성 근거 강화.
