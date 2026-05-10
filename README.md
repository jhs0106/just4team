

---

## 주요 기능

- 네이버 쇼핑 API를 통한 13개 카테고리 상품 자동 수집
- rembg(BiRefNet)를 이용한 상품 이미지 배경 제거 -> 현재는 colab으로 진행
- Google Colab 환경에서 Jina CLIP v2로 이미지/텍스트 임베딩 생성
- PostgreSQL + pgvector 기반 코사인 유사도 검색
- 색감 · 테마 · 용도 · 예산 · 카테고리 조건 기반 데스크 셋업 추천

---

## 전체 처리 흐름

```
[1] 네이버 쇼핑 API → 상품 수집
        ↓
[2] PostgreSQL DB 적재 (products 테이블)
        ↓
[3] 상품 이미지 배경 제거 (rembg BiRefNet)
        ↓
[4] Google Colab — Jina CLIP v2 임베딩 생성
    ├── 이미지 임베딩  (N × 1024)
    └── 텍스트 임베딩  (N × 1024)
        ↓
[5] 임베딩 npy → 로컬 DB 적재 (import_embeddings.py)
        ↓
[6] pgvector 코사인 유사도 검색
        ↓
[7] 카테고리별 후보 검색 → 조합 스코어링 → 셋업 추천
```

> **Jina CLIP v2 모델(`jinaai/jina-clip-v2`)은 1024차원 벡터를 생성하며, GPU 환경(Colab 권장)에서 임베딩을 생성합니다.**

---

## 폴더 / 파일 구조

```
desk_project/
│
├── core/                      # 공통 기반 (외부 프로젝트 의존 없음)
│   ├── config.py              # 전체 설정값 중앙 관리 (.env 로드 포함)
│   └── scoring.py             # 점수 계산 유틸리티 (z-score, minmax 등)
│
├── db/                        # DB 연결 및 쿼리
│   └── db_manager.py          # PostgreSQL 연결 및 products 테이블 관리
│
├── search/                    # 검색 & 추천 로직
│   ├── searcher.py            # Jina CLIP v2 쿼리 임베딩 + pgvector 검색 
│   └── recommender.py         # 셋업 추천 (후보 검색 + 조합 스코어링)
│
├── pipeline/                  # 데이터 수집 & 벡터화
│   ├── collector.py           # 네이버 쇼핑 API 상품 수집
│   └── vectorizer.py          # [LEGACY] KoCLIP 기반 로컬 벡터화 백업 -> 지금은 사용안함
│
├── scripts/
│   ├── export_for_csv.py      # DB → CSV 내보내기 (Colab 업로드용)
│   └── import_embeddings.py   # Colab npy 임베딩 → DB 적재
│
├── docs/
│   ├── PROJECT_FLOW.md        # 전체 파이프라인 상세 설명
│   ├── DB_SETUP.md            # Docker DB 설정 가이드
│   ├── RECOMMENDATION_LOGIC.md # 추천 로직 설명
│ 
│
├── cli.py                     # CLI 입력 파싱 및 결과 출력
├── test.py                    # 대화형 메뉴 진입점 (검색 / 셋업 추천)
├── main.py                    # 전체 파이프라인 CLI (collect / vectorize / search)
│
├── docker-compose.yml         # PostgreSQL + pgvector 컨테이너 정의
├── setup_db.sql               # DB 초기 설정 SQL (참고용)
├── .env.example               # 환경변수 템플릿
├── requirements.txt           # Python 패키지 목록
│
└── data/                      # ⚠️ .gitignore 처리 
    ├── raw/                   # products.csv, 이미지 파일
    └── embeddings/            # npy 임베딩 파일
```

---

## clone한 뒤 따라야 할 순서

```
1. 레포 클론
2. .env 설정
3. Docker DB 실행
4. Python 가상환경 생성 + 패키지 설치
5. 상품 수집 실행
6. Colab에서 임베딩 생성 → DB 적재
7. 검색 / 추천 실행
```

---

## 실행 전 준비사항

- Python 3.11 이상
- Docker Desktop 설치 및 실행
- 네이버 개발자 계정 (쇼핑 API 키)
- HuggingFace 계정 (토큰)
- Google Colab 접근 가능

---

## 1. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 실제 값으로 채운다:

```env
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=여기에_비밀번호_입력

NAVER_CLIENT_ID=네이버_클라이언트_ID
NAVER_CLIENT_SECRET=네이버_클라이언트_시크릿

HF_TOKEN=허깅페이스_토큰
```

> 네이버 API 키: [developers.naver.com](https://developers.naver.com)  
> HuggingFace 토큰: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 2. Docker PostgreSQL 실행

```bash
docker compose up -d
```

컨테이너 상태 확인:

```bash
docker ps
# desk-postgres 컨테이너가 Up 상태인지 확인
```

접속 확인:

```bash
docker exec -it desk-postgres psql -U postgres -d postgres
# \dt 로 테이블 목록 확인
# \q 로 종료
```

> 자세한 DB 설정은 [docs/DB_SETUP.md](docs/DB_SETUP.md) 참고

---

## 3. Python 가상환경 생성 및 패키지 설치

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> PyTorch는 CPU / GPU 환경에 따라 설치 명령이 다를 수 있습니다.  
> GPU 환경: [pytorch.org](https://pytorch.org/get-started/locally/) 에서 CUDA 버전에 맞는 명령 확인

---

## 4. 상품 수집

```bash
python main.py collect
```

실행하면 카테고리별로 수집할 상품 수를 입력하는 프롬프트가 표시됩니다 (엔터 = 기본 10개).

수집 카테고리: DESK, MONITOR, KEYBOARD, MOUSE, MONITOR_ARM, LAPTOP_STAND, MOUSEPAD, DESK_SHELF, LIGHTING, SPEAKER, CLOCK, DESK_LAMP, DECO

---

## 5. 임베딩 생성 및 DB 적재

**Colab 권장 (GPU 필요)**

```bash
# 1) DB 데이터를 CSV로 내보내기
python scripts/export_for_csv.py
# → data/raw/products.csv 생성

# 2) Colab에 products.csv 업로드 후 Jina CLIP v2로 임베딩 생성
#    (Colab 코드는 docs/COLAB_EMBEDDING.md 참고)
#    결과물: product_ids.npy, image_embeds.npy, text_embeds.npy

# 3) npy 파일을 data/embeddings/ 에 배치 후 DB 적재
python scripts/import_embeddings.py
```

> 자세한 흐름: [docs/COLAB_EMBEDDING.md](docs/COLAB_EMBEDDING.md)

---

## 6. 검색 / 셋업 추천 실행

```bash
python test.py
```

```
1: 단일 상품 검색
2: 셋업 추천
모드 선택 (1/2, 종료=q):
```

- **1번** — 검색어를 입력하면 유사한 상품 Top 5 출력
- **2번** — 색감, 테마, 용도, 예산, 카테고리를 입력하면 데스크 셋업 조합 추천

---

## 기타 실행 명령

```bash
# 전체 파이프라인 (수집 → 벡터화 → 검색 테스트)
python main.py

# 테이블 초기화 후 재수집 (지금 쓸 필요 없음)
python main.py clean-collect

# 로컬 이미지 zip으로 벡터화 (얘도 지금 사용 X)
python main.py vectorize-local data/raw/processed_images.zip

# 검색 모드 지정 테스트
python main.py search "화이트 미니멀 책상" --mode equal_zsum

# 임베딩 전체 초기화
python main.py reset-embeddings
```

---

## DB 데이터 공유 방법

```bash
# 덤프 추출 (데이터 제공자)
docker exec desk-postgres pg_dump -U postgres postgres > dump.sql

# 덤프 적용 (팀원)
docker compose up -d
docker exec -i desk-postgres psql -U postgres postgres < dump.sql
```

---


