# just4team — Deskterior

사용자 책상 사진 + 추천 제품 5종을 받아 책상에 배치된 합성 이미지를 생성하는 Visual-RAG 시스템.

## 시스템 구조

```
브라우저 (https)
   ↓
backend-server  (Spring Boot 4, 8443, JSP)
   ↓
ai-server       (FastAPI, 8000)  ──→  PostgreSQL+pgvector
   ↓                                      ↑
recommendation  (FastAPI, 8001)  ─────────┘
```

## 디렉토리

```
just4team/
├── ai-server/                  FastAPI :8000 — SD 생성, 객체 제거, 배치
│   ├── api/
│   │   ├── main.py                    엔드포인트 (/generate, /recommend-and-generate, /jobs)
│   │   ├── models.py                  Pydantic 요청/응답
│   │   ├── config.py                  카테고리·치수·각도 상수
│   │   ├── placement.py               available-space 기반 bbox 계산
│   │   ├── space_analysis.py          top-view 가용 영역 분석
│   │   ├── controlnet_inpaint_processor.py  SD1.5+ControlNet+IP-Adapter+LoRA per-product 생성
│   │   ├── product_inpaint_processor.py     (대체 경로)
│   │   ├── harmonization_processor.py       (폐기, 파일만 보존)
│   │   ├── object_removal_processor.py      DINO+SAM2+LaMa 통합
│   │   ├── dino_processor.py                Grounding DINO
│   │   ├── sam2_processor.py                SAM-2
│   │   ├── lama_processor.py                LaMa inpainting
│   │   ├── composite.py                     CV 합성, 3-zone blend
│   │   ├── mask_utils.py                    마스크 후처리
│   │   ├── utils.py                         find_product_image 등
│   │   └── adapters/
│   │       ├── recommendation_bridge.py     setup dict → GenerateRequest 변환
│   │       └── style_mapper.py              한글 → StyleName enum
│   ├── configs/config.yaml
│   ├── docs/ARCHITECTURE.md       전체 시스템 상세
│   ├── scripts/                   demo, batch test, DB 연결 테스트
│   ├── outputs/
│   │   ├── debug/                 generation 디버그 산출물 (gitignored)
│   │   └── models/lora_external/  JU_DeskStyle LoRA (tracked)
│   ├── data/test/processed_images/  제품 cutout PNG 1711개 (tracked)
│   ├── logs/
│   ├── venv/
│   └── .env                        DB_HOST 등
│
├── recommendation/             FastAPI :8001 — 테마+예산+사용자 사진 → 5종 셋업
│   ├── main.py                        CLI 진입점
│   ├── deskterior/
│   │   ├── api/server.py              POST /recommend
│   │   ├── core/config.py             DB 설정, MODEL_NAME
│   │   ├── database/manager.py
│   │   ├── recommender/
│   │   │   ├── config.py              테마 프리셋, 카테고리, BLEND_WEIGHT
│   │   │   └── engine.py              Beam Search, ItemScore, SetupScore
│   │   ├── retrieval/searcher.py      Jina CLIP v2 (text/image embedding)
│   │   └── cli/recommend.py
│   ├── docker-compose.yml         PostgreSQL+pgvector (옵션, 네이티브 PG 권장)
│   ├── docs/                      DB_SETUP, PROJECT_FLOW, RECOMMENDATION_LOGIC
│   ├── scripts/                   import_embeddings, update_metadata
│   ├── tests/
│   ├── venv/
│   └── .env                        POSTGRES_*, NAVER_*, HF_TOKEN
│
├── backend-server/             Spring Boot 4 :8443 — JSP 프론트 + ai-server 호출
│   ├── src/main/
│   │   ├── java/com/smu/just4team/backendserver/
│   │   │   ├── BackendServerApplication.java
│   │   │   ├── ServletInitializer.java
│   │   │   ├── config/SecurityConfig.java
│   │   │   ├── controller/
│   │   │   │   ├── HomeController.java
│   │   │   │   ├── CustomizeController.java   /customize, POST /api/customize
│   │   │   │   ├── ResultController.java      /result
│   │   │   │   ├── JobStatusController.java   /api/jobs/{id} 프록시
│   │   │   │   └── WebConfig.java
│   │   │   └── service/AiServerClient.java    ai-server HTTP 호출
│   │   ├── resources/
│   │   │   ├── application.properties         포트, SSL, ai.server.base-url
│   │   │   ├── keystore.p12                   HTTPS 자체 서명 (tracked)
│   │   │   └── static/
│   │   └── webapp/WEB-INF/jsp/
│   │       ├── home.jsp, index.jsp, customize.jsp, result.jsp
│   │       └── layout/
│   ├── build.gradle, settings.gradle, gradlew.bat
│   └── bootstrap-5.3.8-dist/      Bootstrap 정적 자원
│
└── desk_db.dump                  PostgreSQL 덤프 (백업용)
```

## 실행 (PC, 각 cmd 창)

```cmd
:: recommendation
cd recommendation && venv\Scripts\activate.bat
uvicorn deskterior.api.server:app --host 0.0.0.0 --port 8001 --reload > logs\rec.log 2>&1

:: ai-server
cd ai-server && venv\Scripts\activate.bat
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir api > logs\ai.log 2>&1

:: backend-server
cd backend-server
gradlew.bat bootRun > logs\spring.log 2>&1
```

웹: https://localhost:8443/customize

## 환경

- Windows 11, Python 3.12.1, Java 17+, PostgreSQL 18 + pgvector
- GPU: RTX 4070 (CUDA 12.4, PyTorch 2.6)
- HuggingFace cache: `C:\Users\JMS\.cache\huggingface\hub` (~96GB)
- 브랜치: `test2` (활성), `main`, `feature/recommendation`, `test`

## 주요 모델

| 용도 | 모델 |
|---|---|
| 객체 검출 | Grounding DINO tiny |
| Segmentation | SAM-2.1 hiera-large |
| Inpainting (제거) | LaMa |
| Depth | DPT-Large |
| 생성 backbone | SD1.5 Inpainting |
| ControlNet | depth, canny |
| Reference | IP-Adapter Plus |
| 스타일 | LoRA JU_DeskStyle |
| 배치 점수 | LightGBM Ranker (AUC 0.9557) |
| 임베딩 | Jina CLIP v2 (text + image) |
