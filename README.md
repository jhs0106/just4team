# Deskterior — 데스크 셋업 추천 시스템

네이버 쇼핑 API로 수집한 상품 데이터를 Jina CLIP v2 멀티모달 임베딩으로 벡터화하고,
PostgreSQL + pgvector 기반으로 테마·예산에 맞는 데스크 셋업 번들을 추천하는 시스템입니다.

---

## 팀원 빠른 시작 (DB 덤프 받은 경우)

> 상품 수집·임베딩 생성 없이, DB 덤프 파일만 있으면 바로 추천 기능을 사용할 수 있습니다.

### 1. 사전 준비

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치 및 실행
- Python 3.11+
- `dump.sql` 파일 (팀원에게 별도 공유 — Google Drive 등)

### 2. 레포 클론 & 환경 설정

```bash
git clone <repo-url>
cd desk_project

# 가상환경 생성 및 패키지 설치
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env.example`을 복사해 `.env`를 만들고 DB 비밀번호를 입력합니다.

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

`.env` 파일 내용 예시:

```
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
```

> `.env`는 절대 커밋하지 마세요 — `.gitignore`에 등록되어 있습니다.

### 4. Docker DB 실행

```bash
docker compose up -d
```

컨테이너 `desk-postgres`가 실행 중인지 확인:

```bash
docker ps
```

### 5. DB 복원

```bash
docker exec -i desk-postgres psql -U postgres postgres < dump.sql
```

> `dump.sql`은 `.gitignore`에 등록되어 있으므로 Google Drive 등 별도 채널로 공유합니다.

### 6. 추천 실행

```bash
python main.py recommend
```

테마(white / black / gaming / wood)와 예산을 입력하면 TOP 3 데스크 셋업 번들이 출력됩니다.

---

## 주요 명령어

| 명령어 | 설명 |
|---|---|
| `python main.py recommend` | 테마 + 예산 기반 셋업 번들 추천 (신규 엔진) |
| `python main.py search` | 단일 상품 벡터 검색 |
| `python main.py` | 인터랙티브 메뉴 (검색 / 추천) |
| `python main.py collect` | 네이버 쇼핑 상품 수집 (데이터 구축 시) |
| `python main.py vectorize` | 레거시 KoCLIP 벡터화 |

---

## 프로젝트 구조

```
desk_project/
├── deskterior/
│   ├── cli/
│   │   ├── recommend.py        # 테마+예산 추천 CLI (신규 엔진 진입점)
│   │   ├── interactive.py      # 인터랙티브 메뉴
│   │   ├── search.py           # 단일 상품 검색 CLI
│   │   ├── pipeline.py         # 데이터 수집/벡터화 파이프라인 CLI
│   │   └── cli.py              # CLI 공통 헬퍼
│   ├── recommender/
│   │   ├── engine.py           # Beam Search + ThemeEvidence 추천 알고리즘
│   │   └── config.py           # 테마 프리셋, 카테고리, 임계값 설정
│   ├── pipeline/
│   │   ├── collector.py        # 네이버 쇼핑 API 수집기
│   │   └── vectorizer.py       # 레거시 KoCLIP 벡터화
│   ├── retrieval/
│   │   └── searcher.py         # Jina CLIP v2 임베딩 + pgvector 검색
│   ├── database/
│   │   └── manager.py          # PostgreSQL 연결, UPSERT, 임베딩 업데이트
│   ├── core/
│   │   ├── config.py           # 환경설정 (.env 로드)
│   │   └── scoring.py          # z-score, minmax 등 점수 유틸
│   └── legacy/
│       └── old_recommender.py  # 레거시 추천 엔진 (색감·테마·용도 기반)
├── scripts/
│   ├── import_embeddings.py    # npy 임베딩 → DB 적재
│   ├── export_for_csv.py       # DB → CSV 내보내기
│   ├── update_metadata.py      # 상품 사이즈 메타데이터 파싱 업데이트
│   └── update_speaker_metadata.py  # 스피커 사이즈 추정값 업데이트
├── tests/                      # 유닛테스트
├── notebooks/                  # Colab 임베딩/배경제거 노트북
├── docs/                       # 상세 문서
├── main.py                     # 전체 진입점
├── docker-compose.yml          # PostgreSQL + pgvector 컨테이너 설정
├── .env.example                # 환경변수 템플릿
└── requirements.txt
```

---

## 추천 엔진 구조 (요약)

1. 선택한 테마(white/black/gaming/wood)에 맞는 카테고리별 검색 쿼리 생성
2. Jina CLIP v2 텍스트 임베딩 → pgvector 코사인 유사도로 후보 상품 검색
3. ThemeEvidence(키워드 매칭), ImageSim, TextSim, ValueScore로 개별 상품 점수화
4. Beam Search(beam_size=50)로 예산 내 최적 번들 탐색
5. SetupScore(ThemeEvidence + RoleAwareCompatibility + MandatoryCoverage 등) 기준 TOP 3 반환

자세한 로직: [docs/RECOMMENDATION_LOGIC.md](docs/RECOMMENDATION_LOGIC.md)

---

## 데이터 직접 구축 (데이터 제공자용)

팀원이 아닌 데이터 구축 담당자는 아래 순서로 진행합니다.

1. 상품 수집: `python main.py collect`
2. CSV 내보내기: `python scripts/export_for_csv.py`
3. Colab에서 Jina CLIP v2 임베딩 생성 (`notebooks/Jina_CLIP_v2_test.ipynb`)
4. npy 3종(`product_ids.npy`, `image_embeds.npy`, `text_embeds.npy`)을 `data/embeddings/`에 배치
5. DB 적재: `python scripts/import_embeddings.py`
6. 덤프 생성 후 공유: `docker exec desk-postgres pg_dump -U postgres postgres > dump.sql`

자세한 파이프라인: [docs/PROJECT_FLOW.md](docs/PROJECT_FLOW.md)

---

## DB 설정 및 공유

자세한 설명: [docs/DB_SETUP.md](docs/DB_SETUP.md)

```bash
# 덤프 추출 (데이터 제공자)
docker exec desk-postgres pg_dump -U postgres postgres > dump.sql

# 덤프 복원 (팀원)
docker exec -i desk-postgres psql -U postgres postgres < dump.sql
```

---

## 테스트

```bash
python -m pytest tests/ -v
```

---

## 주의사항

- `dump.sql`은 `.gitignore`에 등록 — Git으로 공유하지 말고 Google Drive 등으로 공유
- `.env`는 절대 커밋 금지 (DB 비밀번호 포함)
- `data/` 폴더의 이미지, npy, CSV는 용량이 크므로 커밋하지 않음
- Docker 컨테이너를 중지해도 `pgdata` 볼륨에 데이터 보존됨 (`docker compose stop`)
- 볼륨까지 삭제하려면 `docker compose down -v` (데이터 완전 삭제 — 주의)
