# 전체 파이프라인 상세 설명

이 문서는 데스크테리어 추천 시스템의 데이터 흐름을 단계별로 설명합니다.

---

## 전체 흐름 요약

```
네이버 API 수집
    → PostgreSQL DB 적재
    → 상품 이미지 배경 제거
    → Colab: Jina CLIP v2 임베딩 생성
    → npy → DB 적재
    → pgvector 유사도 검색
    → 셋업 추천
```

---

## Step 1. 네이버 쇼핑 API 상품 수집

**실행:**
```bash
python main.py collect
```

**담당 파일:** `collector.py`, `main.py`

**동작:**
- `config.py`의 `CATEGORY_QUERIES`에 정의된 13개 카테고리별 검색어로 네이버 쇼핑 API 호출
- API 1회 최대 100개 / start 최대 1000 제한을 페이징으로 처리
- `CATEGORY_EXCLUDE_KEYWORDS`에 등록된 키워드(식탁, 차량용 등)가 상품명에 포함되면 자동 제외
- `product_id` 기준 UPSERT — 동일 상품이 재수집되면 정보 업데이트 (중복 삽입 없음)
- 실행 시 카테고리별 수집 개수를 입력하는 프롬프트 표시 (엔터 = 기본 10개)

**저장 위치:** PostgreSQL `products` 테이블

---

## Step 2. 상품 이미지 배경 제거

배경 제거는 임베딩 품질을 높이기 위해 수행합니다. 불필요한 배경 없이 제품 자체만 남겨야 이미지 임베딩이 상품 특성을 더 잘 반영합니다.

**방법 A — Colab에서 배경 제거 (권장)**
Colab 노트북에서 rembg를 직접 실행하거나, 배경 제거된 이미지를 생성한 뒤 `data/raw/processed_images/` 에 `{id}.png` 형태로 저장합니다.

**방법 B — 로컬에서 배경 제거 (KoCLIP legacy 경로)**
```bash
python main.py vectorize
```
- rembg BiRefNet 세션으로 배경 제거 → bounding box 크롭 → 정사각형 패딩
- `SKIP_REMBG_CATEGORIES` = {MOUSEPAD, DESK_SHELF} — 납작한 면 제품은 배경 제거 생략

> **현재 프로덕션 경로는 Colab 기반입니다.** `vectorizer.py`는 KoCLIP(512차원) 기반 legacy 코드로, Jina CLIP v2(1024차원)로 전환 전 백업 목적으로 유지합니다.

---

## Step 3. Colab에서 Jina CLIP v2 임베딩 생성

**사전 준비:**
```bash
python scripts/export_for_csv.py
# → data/raw/products.csv 생성 (id, image_url, category, title)
```

이 CSV를 Colab에 업로드하고, Jina CLIP v2 (`jinaai/jina-clip-v2`)로 이미지와 텍스트를 각각 임베딩합니다.

생성되는 파일:
- `product_ids.npy` — 상품 id 배열 (N,)
- `image_embeds.npy` — 이미지 임베딩 (N × 1024)
- `text_embeds.npy` — 텍스트 임베딩 (N × 1024)

> Colab 코드 작성 방법: [COLAB_EMBEDDING.md](COLAB_EMBEDDING.md) 참고

---

## Step 4. 임베딩 DB 적재

생성된 npy 파일 3개를 `data/embeddings/` 에 배치한 뒤:

```bash
python scripts/import_embeddings.py
```

**동작:**
- `product_ids.npy`로 DB의 어느 row에 저장할지 결정
- `image_embeds.npy` → `embedding_img` 컬럼
- `text_embeds.npy` → `embedding_txt` 컬럼
- 하이브리드 벡터 (이미지 × 0.6 + 텍스트 × 0.4, L2 정규화) → `embedding` 컬럼
- DB 컬럼 차원 불일치 감지 시 자동 DROP + 재생성

---

## Step 5. pgvector 유사도 검색

**담당 파일:** `searcher.py`

사용자 텍스트 쿼리를 Jina CLIP v2 `encode_text()`로 1024차원 벡터로 변환하고, pgvector의 `<=>` 연산자(코사인 거리)로 DB 임베딩과 비교합니다.

```sql
1 - (embedding_img <=> query_vector::vector)  -- 이미지 코사인 유사도
1 - (embedding_txt <=> query_vector::vector)  -- 텍스트 코사인 유사도
```

검색 모드 5가지: `image_only`, `text_only`, `raw_sum`, `weighted_sum`, `equal_zsum`
현재 기본값: `image_only`

---

## Step 6. 셋업 추천

**담당 파일:** `recommender.py`, `cli.py`

1. 색감 / 테마 / 용도 → 카테고리별 4개 쿼리 생성
2. 각 쿼리 결과를 minmax 정규화 후 가중 합산 → `product_match_score`
3. `itertools.product`로 카테고리 간 전수 조합
4. 예산 초과 조합 제거
5. `setup_score` 기준 정렬 → 상위 N개 반환

> 추천 로직 상세: [RECOMMENDATION_LOGIC.md](RECOMMENDATION_LOGIC.md) 참고

---

## 실행 순서 체크리스트

```
[ ] docker compose up -d
[ ] python main.py collect
[ ] python scripts/export_for_csv.py
[ ] (Colab) 임베딩 생성 → npy 3개 다운로드
[ ] data/embeddings/ 에 npy 파일 배치
[ ] python scripts/import_embeddings.py
[ ] python test.py
```
