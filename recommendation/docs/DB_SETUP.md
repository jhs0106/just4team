# DB 설정 가이드

Docker 기반 PostgreSQL + pgvector 환경 설정 방법을 설명합니다.

---

## 사전 준비

- Docker Desktop 설치 및 실행 중인 상태
- `.env` 파일에 DB 연결 정보 설정 완료 (`.env.example` 참고)

---

## Docker 컨테이너 실행

```bash
docker compose up -d
```

`docker-compose.yml` 구성:

- 이미지: `pgvector/pgvector:pg16` (PostgreSQL 16 + pgvector 확장 내장)
- 컨테이너명: `desk-postgres`
- 포트: 로컬 5432 → 컨테이너 5432
- 데이터 볼륨: `pgdata` (컨테이너 중지 후에도 데이터 유지)

---

## 컨테이너 상태 확인

```bash
docker ps
```

`desk-postgres` 컨테이너가 `Up` 상태인지 확인합니다.

---

## DB 접속

```bash
# psql로 직접 접속
docker exec -it desk-postgres psql -U postgres -d postgres

# 유용한 psql 명령어
\dt            -- 테이블 목록
\d products    -- products 테이블 컬럼 구조
SELECT COUNT(*) FROM products;
\q             -- 종료
```

---

## products 테이블 구조

테이블은 `deskterior/database/manager.py`의 `create_table()`이 자동으로 생성합니다.
별도로 SQL 파일을 실행할 필요 없습니다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | SERIAL | 자동 증가 PK |
| title | TEXT | 상품명 (HTML 태그 제거됨) |
| link | TEXT | 상품 상세 링크 |
| image | TEXT | 상품 이미지 URL |
| lprice | INTEGER | 최저가 (원) |
| mall_name | TEXT | 쇼핑몰명 |
| product_id | TEXT (UNIQUE) | 네이버 상품 고유 ID |
| brand | TEXT | 브랜드명 |
| category | TEXT | 카테고리 코드 (예: `KEYBOARD`, `MONITOR`) |
| embedding_img | vector(1024) | Jina CLIP v2 이미지 임베딩 |
| embedding_txt | vector(1024) | Jina CLIP v2 텍스트 임베딩 |
| metadata | JSONB | 파싱된 사이즈 정보 (모니터 인치, 키보드 배열 등) |

> `metadata` 예시:
> - MONITOR: `{"inch": 27, "width_mm": 614, "depth_mm": 180, "source": "title"}`
> - KEYBOARD: `{"layout": "TKL", "width_mm": 360, "depth_mm": 130, "source": "title"}`
>
> `width_mm` / `depth_mm`는 가용공간 필터(`space_constraints`) 적용 시 사용됩니다. 없는 경우 해당 상품은 크기 제약 없이 통과합니다.

---

## pgvector 설치 확인

```sql
-- psql 접속 후 실행
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

`pgvector/pgvector:pg16` 이미지는 pgvector가 내장되어 있습니다.

---

## 컨테이너 관리

```bash
# 중지 (데이터 보존)
docker compose stop

# 재시작
docker compose up -d

# 컨테이너만 삭제 (볼륨/데이터 보존)
docker compose down

# 완전 삭제 (데이터 포함 — 주의)
docker compose down -v
```

---

## DB 데이터 공유

상품 데이터, 임베딩 등은 Git에 포함되지 않으므로 덤프 파일로 공유합니다.

```bash
# 덤프 추출 (데이터 제공자)
docker exec desk-postgres pg_dump -U postgres postgres > dump.sql

# 덤프 복원 (팀원)
docker compose up -d
docker exec -i desk-postgres psql -U postgres postgres < dump.sql
```

> `dump.sql`은 `.gitignore`에 등록되어 있으므로 Google Drive 등 별도 채널로 공유하세요.

---

## 주의사항

- `.env`는 절대 커밋 금지 (DB 비밀번호 포함)
- `docker compose down -v`는 볼륨까지 삭제하므로 데이터가 완전히 사라집니다
- 임베딩(`embedding_img`, `embedding_txt`)은 1024차원 벡터로, 덤프 파일 크기가 클 수 있습니다
