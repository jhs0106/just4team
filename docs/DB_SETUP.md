# DB 설정 가이드

이 문서는 Docker 기반 PostgreSQL + pgvector 환경 설정 방법을 설명합니다.

---

## 사전 준비

- Docker Desktop 설치 및 실행 중인 상태
- `.env` 파일에 DB 비밀번호 설정 완료

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

## DB 접속 방법

```bash
# psql로 직접 접속
docker exec -it desk-postgres psql -U postgres -d postgres

# 유용한 psql 명령어
\dt          -- 테이블 목록
\d products  -- products 테이블 컬럼 구조
\q           -- 종료
```

---

## 테이블 자동 생성

테이블은 `db_manager.py`가 자동으로 생성합니다. 별도로 SQL을 실행할 필요 없습니다.

```bash
# main.py collect 실행 시 자동으로 create_table() 호출됨
python main.py collect
```

수동으로 테이블만 생성하려면:
```bash
python -c "from db_manager import DBManager; db = DBManager(); db.create_table(); db.close()"
```

---

## products 테이블 구조

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | SERIAL | 자동 증가 PK |
| title | TEXT | 상품명 (HTML 태그 제거됨) |
| link | TEXT | 상품 링크 |
| image | TEXT | 이미지 URL |
| lprice | INTEGER | 최저가 (원) |
| mall_name | TEXT | 쇼핑몰명 |
| product_id | TEXT (UNIQUE) | 네이버 상품 고유 ID |
| brand | TEXT | 브랜드명 |
| category | TEXT | 카테고리 코드 (예: KEYBOARD) |
| embedding_img | vector(1024) | Jina CLIP v2 이미지 임베딩 |
| embedding_txt | vector(1024) | Jina CLIP v2 텍스트 임베딩 |
| embedding | vector(1024) | 하이브리드 임베딩 (이미지 0.6 + 텍스트 0.4) |
| metadata | JSONB | 파싱된 사이즈 정보 (책상/모니터 등) |

---

## pgvector 사용 여부 확인

```sql
-- psql 접속 후
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

`vector` 확장이 설치되어 있으면 pgvector가 활성화된 상태입니다. `pgvector/pgvector:pg16` 이미지를 사용하면 자동으로 내장되어 있습니다.

---

## 컨테이너 관리 명령어

```bash
# 중지 (데이터 보존)
docker compose stop

# 재시작
docker compose up -d

# 완전 삭제 (데이터 포함)
docker compose down -v
```

> `docker compose down` (v 옵션 없음)은 컨테이너만 삭제하고 볼륨(데이터)은 보존합니다.  
> `-v` 옵션을 붙이면 볼륨까지 삭제되므로 데이터가 완전히 사라집니다. 주의하세요.

---

## DB 데이터 공유

데이터를 공유할 때는 dump 파일을 사용합니다.

```bash
# 덤프 추출 (제공자)
docker exec desk-postgres pg_dump -U postgres postgres > dump.sql

# 덤프 적용 (팀원)
docker compose up -d
docker exec -i desk-postgres psql -U postgres postgres < dump.sql
```

---

## 주의사항

- `dump.sql`은 .gitignore에 등록되어 있으므로 커밋되지 않습니다. 별도 채널로 공유하세요.
- DB 비밀번호는 `.env`에서 관리하며, `.env`는 절대 커밋하지 않습니다.
- `setup_db.sql`은 초기 스키마 참고용 파일이며, 현재는 `db_manager.create_table()`이 자동 처리합니다.
