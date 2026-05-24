# 전체 파이프라인 설명

데스크테리어 추천 시스템의 두 가지 사용 경로를 설명합니다.

---

## 경로 A — 팀원 (DB 덤프 복원 후 추천만 사용)

> 상품 수집·임베딩 생성 없이 DB 덤프 파일만 있으면 됩니다.

```
dump.sql 수령
    ↓
Docker DB 실행: docker compose up -d
    ↓
DB 복원: docker exec -i desk-postgres psql -U postgres postgres < dump.sql
    ↓
python main.py recommend
```

### 실행 체크리스트

```
[ ] Docker Desktop 실행
[ ] git clone + pip install -r requirements.txt
[ ] .env 파일 작성 (.env.example 참고)
[ ] docker compose up -d
[ ] docker exec -i desk-postgres psql -U postgres postgres < dump.sql
[ ] python main.py recommend
```

---

## 경로 B — 데이터 제공자 (처음부터 데이터 구축)

```
네이버 쇼핑 API 상품 수집
    ↓
PostgreSQL DB 적재
    ↓
CSV 내보내기 → Colab 업로드
    ↓
Colab: Jina CLIP v2 이미지·텍스트 임베딩 생성
    ↓
npy 3종 다운로드 → data/embeddings/ 배치
    ↓
python scripts/import_embeddings.py (DB 적재)
    ↓
python main.py recommend
    ↓
docker exec desk-postgres pg_dump -U postgres postgres > dump.sql (팀원 공유용)
```

---

## Step 1. 네이버 쇼핑 API 상품 수집

```bash
python main.py collect
```

- `deskterior/core/config.py`의 `CATEGORY_QUERIES`에 정의된 카테고리별 검색어로 네이버 쇼핑 API 호출
- `CATEGORY_EXCLUDE_KEYWORDS`에 등록된 키워드 포함 상품 자동 제외 (TV, 차량용 등)
- `product_id` 기준 UPSERT — 재수집 시 중복 없이 정보 업데이트
- 담당 파일: `deskterior/pipeline/collector.py`

---

## Step 2. CSV 내보내기

```bash
python scripts/export_for_csv.py
# → data/raw/products.csv 생성 (id, image_url, category, title)
```

이 CSV를 Colab에 업로드해 임베딩 생성에 사용합니다.

---

## Step 3. Colab에서 Jina CLIP v2 임베딩 생성

노트북: `notebooks/Jina_CLIP_v2_test.ipynb`

- 상품 이미지 URL에서 이미지를 다운로드해 배경 제거 (rembg)
- Jina CLIP v2 (`jinaai/jina-clip-v2`) 모델로 이미지·텍스트 각각 1024차원 임베딩 생성
- GPU 환경 권장 (Colab T4 이상)

생성되는 파일:
- `product_ids.npy` — 상품 id 배열 (N,)
- `image_embeds.npy` — 이미지 임베딩 (N × 1024)
- `text_embeds.npy` — 텍스트 임베딩 (N × 1024)

---

## Step 4. 임베딩 DB 적재

생성된 npy 3개를 `data/embeddings/`에 배치한 뒤:

```bash
python scripts/import_embeddings.py
```

- `product_ids.npy`로 어느 row에 저장할지 결정
- `image_embeds.npy` → `embedding_img` 컬럼 (vector(1024))
- `text_embeds.npy` → `embedding_txt` 컬럼 (vector(1024))
- DB 컬럼 차원 불일치 감지 시 자동 DROP + 재생성

---

## Step 5. 메타데이터 파싱 (선택)

상품 사이즈 정보(`metadata` JSONB 컬럼)를 파싱합니다.

```bash
python scripts/update_metadata.py
```

- 모니터 인치, 키보드 배열(텐키리스 등), 마우스 무선 여부 등을 상품명에서 파싱
- `SIZE_RELEVANT = {MONITOR, KEYBOARD, MOUSE, HEADSET, MOUSEPAD}` 카테고리에 적용

---

## Step 6. 추천 실행

```bash
python main.py recommend
```

테마(white/black/gaming/wood)와 예산을 입력하면 TOP 3 데스크 셋업 번들이 출력됩니다.

---

## Step 7. 덤프 추출 (팀원 공유용)

```bash
docker exec desk-postgres pg_dump -U postgres postgres > dump.sql
```

`dump.sql`을 Google Drive 등 별도 채널로 공유합니다 (`.gitignore`에 등록되어 있어 Git에는 포함되지 않음).

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `deskterior/pipeline/collector.py` | 네이버 쇼핑 API 수집기 |
| `deskterior/database/manager.py` | PostgreSQL 연결, UPSERT, 임베딩 업데이트 |
| `deskterior/retrieval/searcher.py` | Jina CLIP v2 임베딩 + pgvector 검색 |
| `deskterior/recommender/engine.py` | Beam Search + ThemeEvidence 추천 알고리즘 |
| `deskterior/recommender/config.py` | 테마 프리셋, 카테고리, threshold 설정 |
| `scripts/import_embeddings.py` | npy → DB 적재 |
| `scripts/export_for_csv.py` | DB → CSV 내보내기 |
| `scripts/update_metadata.py` | 사이즈 메타데이터 파싱 업데이트 |
