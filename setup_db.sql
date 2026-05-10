-- =============================================
-- pgvector 테스트용 DB 초기 설정 스크립트
-- psql -U postgres -d deskterior -f setup_db.sql
-- =============================================

-- 1. 연결 후 pgvector 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 상품 테이블 생성
--    embedding: KoCLIP koclip-base-pt 모델 기준 512차원
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,           -- 상품명 (한글)
    category    TEXT,                    -- 카테고리 (모니터, 키보드 등)
    price       INTEGER,                 -- 가격 (원)
    image_url   TEXT,                    -- 상품 이미지 URL
    embedding   vector(512)              -- KoCLIP 임베딩 벡터
);

-- 3. 벡터 유사도 검색용 인덱스 생성
--    ivfflat: 대용량에 적합한 근사 최근접 이웃(ANN) 인덱스
--    lists=100: 클러스터 수 (데이터가 적을 때는 낮게 설정)
CREATE INDEX IF NOT EXISTS products_embedding_idx
    ON products
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

-- 확인
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
