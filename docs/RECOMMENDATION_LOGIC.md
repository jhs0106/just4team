# 추천 로직 설명

이 문서는 데스크 셋업 추천 시스템의 내부 동작 방식을 설명합니다.

---

## 전체 흐름 (그냥 테스트용 추가로 튜닝해야 함)

```
사용자 조건 입력 (색감 / 테마 / 용도 / 예산 / 카테고리)
    ↓
카테고리별 다중 쿼리 생성
    ↓
Jina CLIP v2 query embedding
    ↓
pgvector 코사인 유사도 → image_score, text_score 계산
    ↓
후보 집계 (minmax 정규화 + 가중 합산 → product_match_score)
    ↓
카테고리 간 전수 조합 생성
    ↓
예산 초과 조합 제거
    ↓
setup_score 계산 → 정렬 → 상위 N개 반환
```

---

## 1. 사용자 조건 입력

`test.py` → `cli.py`의 `run_setup_recommendation()` 에서 다음 항목을 입력받습니다:

| 항목 | 예시 |
|---|---|
| 색감 | 화이트톤, 블랙, 우드톤 |
| 테마 | 미니멀, 게이밍, 빈티지 |
| 용도 | 개발자 데스크셋업, 재택근무 |
| 예산 | 500000 (원, 비우면 무제한) |
| 카테고리 | keyboard, mouse, lamp 또는 번호 선택 |

카테고리는 번호 또는 텍스트로 입력할 수 있으며, `cli.py`의 `_prompt_categories_selection()`이 파싱합니다.

---

## 2. 카테고리별 다중 쿼리 생성

`recommender.py`의 `build_category_queries()`가 입력 조건으로 카테고리당 최대 4개 쿼리를 생성합니다.

| 쿼리 형태 | 가중치 |
|---|---|
| `"{색감} 데스크셋업에 어울리는 {카테고리}"` | 0.35 |
| `"{테마} 데스크셋업에 어울리는 {카테고리}"` | 0.35 |
| `"{용도}에 적합한 {카테고리}"` | 0.20 |
| `"{색감+테마+용도}에 어울리는 {카테고리}"` | 0.10 |

입력하지 않은 항목은 해당 쿼리가 생략되고, 나머지 가중치가 합이 1.0이 되도록 자동 정규화됩니다.

---

## 3. Jina CLIP v2 Query Embedding

`searcher.py`의 `get_query_embedding()`이 각 쿼리 텍스트를 Jina CLIP v2 `encode_text()`로 1024차원 벡터로 변환하고 L2 정규화합니다.

```python
feat = model.encode_text([query_text])
feat = F.normalize(feat, dim=-1)
```

---

## 4. image_score / text_score 계산

pgvector `<=>` 연산자(코사인 거리)로 쿼리 벡터와 DB 임베딩 간 유사도를 계산합니다.

```sql
1 - (embedding_img <=> query_vector::vector)  AS image_score
1 - (embedding_txt <=> query_vector::vector)  AS text_score
```

### 검색 모드 (mode 파라미터)

| 모드 | 계산식 | 설명 |
|---|---|---|
| `image_only` | `image_score` | 이미지 임베딩 유사도만 사용 **(현재 기본값)** |
| `text_only` | `text_score` | 텍스트 임베딩 유사도만 사용 |
| `raw_sum` | `image + text` | 단순 합산 |
| `weighted_sum` | `w_img×image + w_txt×text` | 가중치 지정 합산 |
| `equal_zsum` | `z(image) + z(text) + 1.5×title_match` | z-score 정규화 + 키워드 매칭 보정 |

`equal_zsum`의 title_match는 쿼리 토큰과 상품명 토큰의 겹침 비율로 계산합니다 (`scoring.py`의 `_title_match_scores()`).

---

## 5. 후보 집계 (product_match_score)

각 카테고리에 대해 여러 쿼리를 실행한 결과를 하나의 점수로 합산합니다 (`search_category_candidates()`).

```python
# 쿼리별로 minmax 정규화 후 가중 합산
product_match_score[id] += minmax(final_score) * query_weight
```

최종 `product_match_score` 기준으로 내림차순 정렬 후 카테고리별 상위 N개(기본 5개)를 후보로 선정합니다.

---

## 6. 카테고리 간 조합 생성

`itertools.product`로 카테고리별 후보를 전수 조합합니다.

```python
for combo in itertools.product(*candidate_lists):
    ...
```

카테고리가 5개 이상이면 조합 폭발을 방지하기 위해 카테고리당 후보를 최대 5개로 제한합니다.

---

## 7. 예산 초과 조합 제거

각 조합의 알려진 가격 합계가 예산을 초과하면 제거합니다.

```python
total_price = sum(item["lprice"] for item in combo if item.get("lprice") is not None)
if pref.budget is not None and total_price > pref.budget:
    continue  # 제거
```

---

## 8. setup_score 계산

```
setup_score = 0.65 × avg_match_score
            + 0.25 × budget_score
            + 0.10 × completeness_score
```

| 항목 | 설명 |
|---|---|
| `avg_match_score` | 조합 내 상품 `product_match_score` 평균 |
| `budget_score` | `1.0 - │남은금액 / 예산│ × 0.3` (예산 없으면 1.0 고정) |
| `completeness_score` | 모든 카테고리 채워진 경우 1.0 (현재 고정) |

가격 미상 상품 포함 시 `-0.05` 페널티 (최대 `-0.15`).

`setup_score` 기준 내림차순 정렬 후 상위 N개(기본 3개)를 반환합니다.

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `config.py` | 검색 기본값 (DEFAULT_SEARCH_MODE, DEFAULT_CANDIDATE_PER_CATEGORY 등) |
| `scoring.py` | z-score, title_match, minmax 계산 함수 |
| `searcher.py` | query embedding + pgvector 검색 |
| `recommender.py` | 다중 쿼리 집계 + 조합 스코어링 |
| `cli.py` | 사용자 입력 파싱 + 결과 출력 |
