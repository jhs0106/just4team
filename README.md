# just4team — Deskterior

Visual-RAG 기반 데스크테리어 시뮬레이션. 사용자 책상 사진 + 추천 제품 5종을 받아 책상 위에 배치된 합성 이미지를 생성한다.

- 팀: just4team
- 활성 브랜치: `test2`
- 관련 브랜치: `feature/recommendation`, `main`, `test`

---

## 1. 시스템 구조

```
브라우저 (https)
   ↓
backend-server  (Spring Boot 4, JSP)
   ↓
ai-server       (FastAPI)
   ↓
recommendation  (FastAPI)  →  PostgreSQL+pgvector
                               ↑
                          Jina CLIP v2 (text + image embedding)
```

각 서버 포트는 `application.properties` / `.env` / uvicorn 실행 시 인자로 설정.

역할 분리:
- **backend-server**: JSP 프론트 + 사용자 입력 수집 + ai-server HTTP 호출
- **recommendation**: 테마 + 예산 + 사용자 정면 사진 → 5종 셋업 (Visual-RAG의 Retrieval)
- **ai-server**: 책상 분석 + 제품 배치 + SD 합성 (Visual-RAG의 Augmentation + Generation)

---

## 2. 전체 데이터 흐름

```
사용자 입력 (JSP customize.jsp)
  · 책상 정면 사진 (필수)
  · 책상 top-view 사진 (필수)
  · 책상 width/depth cm (필수)
  · style (화이트/블랙/게이밍/우드)
  · budget (원)
  · deskMode (own_desk/add/replace/empty_desk)
  · (옵션) deskClickX, deskClickY  ← 빈 책상 모드
        ↓
backend-server CustomizeController
  · STYLE_MAP: 한글 → enum (white/black/gaming/wood)
  · cm → mm 변환
  · base64 인코딩
  · AiServerClient.submitRecommendAndGenerate()
        ↓ POST /recommend-and-generate
ai-server /recommend-and-generate
  · job_id 즉시 발급, 백그라운드 처리
  · POST /recommend { theme, budget, image_base64 }
        ↓
recommendation /recommend
  · Jina CLIP v2 image encoder → 사용자 정면 사진 임베딩
  · 카테고리별 텍스트 쿼리 + 사용자 사진 임베딩 가중평균 (W=0.4)
  · pgvector 검색 → 카테고리별 후보 50개
  · ItemScore 계산 → 카테고리별 정렬
  · Beam Search (beam_size=50, top_k=1)
  · ThemeGate 통과 검사
  · setup 반환 (5종: MONITOR, KEYBOARD, MOUSE, MOUSEPAD, SPEAKER)
        ↓
ai-server: setup → GenerateRequest 변환 (recommendation_bridge)
  · 카테고리 매핑 (10개 1:1 + LIGHTING / DESK/MONITOR_ARM skip)
  · find_product_image(id): processed_images/{id}.png 존재 검증
  · ProductItem(category, name, image_id, width_mm, depth_mm)
        ↓
ai-server run_generate (Stage 1 → 2 → 3)
        ↓
job_store[job_id]: { result_image, products, num_placed, ... }
        ↓ Spring 폴링 (JobStatusController)
JSP result.jsp: 합성 이미지 + 제품 정보 표시
```

---

## 3. 디렉토리 구조

```
just4team/
├── ai-server/                          FastAPI 서버
│   ├── api/
│   │   ├── main.py                            엔드포인트 + run_generate 오케스트레이션 + _run_cn per-product 루프
│   │   ├── models.py                          Pydantic 요청/응답 스키마
│   │   ├── config.py                          카테고리 상수 (mm, 종횡비, 배치 순서, perspective tilt)
│   │   ├── utils.py                           b64 변환, 카테고리 정규화, CSV 카탈로그 로드, find_product_image
│   │   ├── placement.py                       Stage 2: available-space 기반 bbox 계산, ranker 통합
│   │   ├── space_analysis.py                  Stage 2: top-view 가용 영역 분석
│   │   ├── object_removal_processor.py        Stage 1: DINO+SAM2+LaMa 오케스트레이션
│   │   ├── dino_processor.py                  Grounding DINO 텍스트 검출
│   │   ├── sam2_processor.py                  SAM-2.1 세그멘테이션
│   │   ├── lama_processor.py                  LaMa inpainting
│   │   ├── controlnet_inpaint_processor.py    Stage 3: SD1.5+ControlNet[depth,canny]+IP-Adapter+LoRA
│   │   ├── composite.py                       CV alpha composite, silhouette, contact/cast shadow
│   │   ├── mask_utils.py                      마스크 후처리
│   │   ├── product_inpaint_processor.py       legacy (/product_place 전용)
│   │   ├── harmonization_processor.py         폐기 (보존만)
│   │   └── adapters/
│   │       ├── recommendation_bridge.py       setup dict → GenerateRequest
│   │       └── style_mapper.py                한글 → StyleName enum
│   ├── configs/config.yaml                    런타임 설정
│   ├── docs/ARCHITECTURE.md                   상세 아키텍처
│   ├── scripts/                               run_demo, end_to_end_demo, styled_test, test_db_connection
│   ├── data/test/
│   │   ├── products.csv                       제품 카탈로그
│   │   └── processed_images/{id}.png          제품 cutout PNG 1711개 (tracked)
│   ├── outputs/
│   │   ├── debug/<timestamp>/                 generation 디버그 (gitignored)
│   │   └── models/
│   │       ├── layout_ranker.pkl              LightGBM AUC 0.9557 (tracked)
│   │       └── lora_external/JU_DeskStyle/    PEFT LoRA (tracked)
│   ├── logs/                                  서버 stdout/stderr
│   ├── venv/                                  Python 3.12 (gitignored)
│   └── .env                                   DB_HOST, DB_PORT, ...
│
├── recommendation/                     FastAPI 서버
│   ├── main.py                                CLI 진입점
│   ├── deskterior/
│   │   ├── api/server.py                      POST /recommend (image_base64 옵션)
│   │   ├── core/config.py                     DB 설정, MODEL_NAME=jinaai/jina-clip-v2
│   │   ├── database/manager.py                psycopg2 connection
│   │   ├── recommender/
│   │   │   ├── config.py                      THEME_PRESETS, MANDATORY/OPTIONAL_CATEGORIES, ROLE_PAIR_WEIGHTS, OPTIONAL_GAIN_THRESHOLD, BLEND_WEIGHT
│   │   │   └── engine.py                      retrieve_candidates, compute_item_score, recommend_setup (Beam Search), pareto_filter, diversify
│   │   ├── retrieval/searcher.py              embed_text_query, embed_image_query (Jina CLIP v2 + xattn patch)
│   │   └── cli/recommend.py                   CLI 진입
│   ├── docker-compose.yml                     pgvector/pgvector:pg16 (옵션, 네이티브 PG 권장)
│   ├── docs/                                  DB_SETUP, PROJECT_FLOW, RECOMMENDATION_LOGIC
│   ├── scripts/                               import_embeddings, update_metadata, _fix_monitor_metadata
│   ├── tests/                                 pytest
│   ├── venv/                                  (gitignored)
│   └── .env                                   POSTGRES_*, NAVER_*, HF_TOKEN
│
├── backend-server/                     Spring Boot 4 (HTTPS)
│   ├── src/main/
│   │   ├── java/com/smu/just4team/backendserver/
│   │   │   ├── BackendServerApplication.java
│   │   │   ├── ServletInitializer.java
│   │   │   ├── config/SecurityConfig.java
│   │   │   ├── controller/
│   │   │   │   ├── HomeController.java                /, /home
│   │   │   │   ├── CustomizeController.java           GET /customize, POST /api/customize
│   │   │   │   ├── ResultController.java              GET /result
│   │   │   │   ├── JobStatusController.java           GET /api/jobs/{id} → ai-server 프록시
│   │   │   │   └── WebConfig.java
│   │   │   └── service/AiServerClient.java            submitRecommendAndGenerate, fetchJobStatusRaw
│   │   ├── resources/
│   │   │   ├── application.properties                 server.port, ai.server.base-url, SSL keystore 경로
│   │   │   ├── keystore.p12                           HTTPS 자체 서명 (tracked)
│   │   │   └── static/
│   │   └── webapp/WEB-INF/jsp/
│   │       ├── home.jsp, index.jsp, customize.jsp, result.jsp
│   │       └── layout/
│   ├── build.gradle, settings.gradle, gradlew.bat
│   └── bootstrap-5.3.8-dist/                  Bootstrap 정적 자원
│
└── desk_db.dump                        PostgreSQL pg_dump 백업
```

---

## 4. recommendation 서버 알고리즘

### 4.1 입력
- `theme`: white | black | gaming | wood
- `budget`: 원 (양수)
- `image_base64`: 사용자 책상 정면 사진 (옵션, 주어지면 사진 임베딩이 검색 쿼리에 반영됨)

### 4.2 카테고리
- **MANDATORY**: MONITOR, KEYBOARD, MOUSE, MOUSEPAD, SPEAKER (5종 모두 필수)
- **OPTIONAL**: (현재 비어 있음) — 향후 DESK_LAMP, HEADSET 등 확장 여지

### 4.3 후보 검색
1. `THEME_CATEGORY_QUERIES[theme][category]` → 카테고리별 검색 텍스트
2. `embed_text_query(query)` → 1024-dim 임베딩
3. 사용자 사진 있으면 `embed_image_query(image_bytes)` → 가중평균:
   ```
   query_emb = (1 - W) × text_emb + W × user_image_emb
   W = USER_IMAGE_BLEND_WEIGHT = 0.4
   query_emb = L2 normalize(query_emb)
   ```
4. pgvector 검색 (SQL):
   ```sql
   ORDER BY (0.60 × (embedding_img <=> query_vec)
           + 0.40 × (embedding_txt <=> query_vec)) ASC
   LIMIT 50
   ```
5. `CATEGORY_EXCLUDE_KEYWORDS` 필터 (예: MONITOR에서 "TV", "사이니지" 제거)

### 4.4 ItemScore
```
ItemScore = 0.30 × ImageSim
          + 0.25 × TextSim
          + 0.35 × ThemeEvidence
          + 0.10 × ValueScore
```

ThemeEvidence:
```
= min(1.0, positive_hits / 2.0)
- min(0.6, negative_hits × 0.20)
+ CategoryThemeBonus
```
positive/negative_keywords는 `THEME_PRESETS[theme]`. CategoryThemeBonus는 카테고리+테마 조합에 추가 보너스 (예: gaming 모니터에 "144hz" 포함 시 +0.20).

ValueScore: 카테고리 내 min-max 정규화된 `(0.30×ImageSim + 0.25×TextSim + 0.35×ThemeEvidence) / log(1 + price)`.

### 4.5 Beam Search (recommend_setup)
- **Phase 1**: MANDATORY 카테고리 순차 추가, beam_size=50 유지, 예산 초과 가지 제거, quick_score 기준 정렬
- **Phase 2**: OPTIONAL 카테고리 (현재 비어 있음) — OptionalGain ≥ threshold일 때만 포함
- 각 beam state는 누적 가격, 카테고리 커버리지, items, quick_score 보유

### 4.6 SetupScore
```
SetupScore = 0.35 × SetupThemeEvidence
           + 0.20 × AvgItemScore
           + 0.20 × RoleAwareCompatibility
           + 0.15 × MandatoryCoverage
           + 0.05 × OptionalUsefulness
           + 0.05 × ValueEfficiency
```

RoleAwareCompatibility: `ROLE_PAIR_WEIGHTS` 가중 평균
| 카테고리 쌍 | 가중치 |
|---|---|
| KEYBOARD ↔ MOUSE | 1.00 |
| KEYBOARD ↔ MOUSEPAD | 0.95 |
| MOUSE ↔ MOUSEPAD | 0.95 |
| MONITOR ↔ SPEAKER | 0.60 |
| MONITOR ↔ DESK_LAMP | 0.50 |
| 기타 쌍 | 0.30 |

### 4.7 ThemeGate
| 테마 | 필수 카테고리 매칭 최소 | 평균 ThemeEvidence 최소 | 충돌 키워드 허용 |
|---|---|---|---|
| white | 1개 이상 | 0.58 | ≤ 1 |
| black | 2개 이상 | 0.60 | ≤ 1 |
| gaming | 2개 이상 | 0.65 | (제한 없음) |
| wood | 1개 이상 | 0.53 (optional 있을 때) / 0.60 (없을 때) | ≤ 1 |

기준 미달 시 `ALLOW_THEME_GATE_FALLBACK=True`이면 final_score에 ×0.75 패널티 후 통과.

### 4.8 Pareto Filter + Diversify
- 동등하거나 우월한 번들이 있는 가지 제거
- 상위 K개 번들 중 ID Jaccard 유사도 > 0.6면 제외 (다양성 확보)

---

## 5. ai-server 파이프라인 (POST /generate)

```
입력
  desk_image (front-view, base64)
  top_view_image (필수)
  style (StyleName enum)
  products[] (category, image_id, width_mm, depth_mm)
  desk_width_mm, desk_depth_mm
  mode (own_desk|add|replace|empty_desk)
  generation_mode (controlnet|cv_composite|placement_only)
        ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Object Removal                                     │
│ object_removal_processor.py                                 │
│                                                             │
│ 1.1 DINO 검출 — _REMOVAL_PROMPT 텍스트 매칭                 │
│     "laptop. monitor. keyboard. mouse. mousepad. ..."       │
│ 1.2 SAM-2 세그멘테이션 — bbox → 정확한 mask                 │
│ 1.3 max_area_ratio 필터 (0.40) — 책상 자체 잡힘 방지        │
│ 1.4 LaMa inpainting — mask 영역 채움                        │
│                                                             │
│ 출력: cleaned_front (배치 가능한 빈 책상)                   │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Placement                                          │
│ placement.calc_placements_from_available_space              │
│                                                             │
│ 2.1 top-view 공간 분석 (space_analysis)                     │
│   · DINO로 top-view 점유 영역 검출 → occupied_mask          │
│   · SAM-2로 책상 영역 segmentation → desk_mask              │
│   · mode 따라 remove_mask 결정 (add: 유지 / own_desk: 전부) │
│   · available_mask = desk_mask − keep_occupied              │
│   · connectedComponentsWithStats로 가용 region 분할         │
│                                                             │
│ 2.2 후보 위치 샘플링                                        │
│   · 각 가용 region 안에 7×7 grid 후보점                     │
│   · ry pre-filter: _CAT_RY_RANGE 밖 후보 즉시 제외          │
│                                                             │
│ 2.3 점수 계산 (score_region_for_product)                    │
│   rule_score: 카테고리 선호 위치까지 거리 + 페널티 (영역    │
│               위반, 근접, 가장자리)                         │
│   learned_score: LightGBM ranker (22 feature)               │
│               MONITOR/MOUSEPAD/LIGHTING은 학습 샘플 부족 →  │
│               rule_score만 사용                             │
│   final_score = rule × 0.7 + learned × 0.3                  │
│                                                             │
│ 2.4 최고점 → front-view bbox 변환                           │
│   _front_bbox_for_anchor:                                   │
│   · ry를 _CAT_RY_RANGE로 clamp                              │
│   · ps = 0.60 + 0.40 × ry (perspective scale, 가까울수록 큼)│
│   · px_per_mm = fv_dw / desk_width_mm                       │
│   · fv_pw = w_mm × px_per_mm × ps                           │
│   · fv_ph = d_mm × px_per_mm × ps                           │
│   · 카테고리별 cap (KEYBOARD/MONITOR/MOUSE 등) 적용         │
│   · 카테고리간 관계 강제 (relation_state):                  │
│       monitor_rx → keyboard 가로 정렬                       │
│       keyboard_y2 → mouse 같은 y                            │
│       mousepad_region → mouse 안쪽 강제 배치                │
│   · KEYBOARD 강제 후보: scoring 실패 시 monitor 아래 강제   │
│                                                             │
│ 출력: placement_items = [{product, region(x1,y1,x2,y2),     │
│                          score, score_meta, ...}]           │
└─────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Per-product SD Generation                           │
│ controlnet_inpaint_processor.generate_product                │
│ (placement 순서대로 카테고리당 1회 SD 호출)                  │
│                                                              │
│ 3.1 Context crop (bbox + padding)                            │
│ 3.2 SD 호환 해상도 resize (긴 변 512, 8의 배수)              │
│ 3.3 perspective_warp (flat 카테고리) — _CAT_TILT_DEG         │
│     MOUSEPAD 20°, KEYBOARD 30°, MOUSE 45°, LAPTOP_STAND 50°  │
│ 3.4 CV pre-composite — 제품을 cleaned crop 위에 합성         │
│     (canny edge + 3-zone blend 입력)                         │
│ 3.5 ControlNet 입력 생성                                     │
│   · depth_sd: DPT-Large(cleaned crop)                        │
│   · canny_sd: Canny(composite_sd) ← 제품 외곽선              │
│ 3.6 IP-Adapter 입력: 제품 이미지 letterbox 512×512           │
│ 3.7 silhouette mask: 제품 alpha → inpaint 영역               │
│ 3.8 SD ControlNet Inpaint 호출 (1 pass)                      │
│   · image=cleaned crop, mask=silhouette                      │
│   · control=[depth, canny], ip_adapter=product               │
│   · prompt="{cat_desc}, {style} style, ..."                  │
│   · LoRA=JU_DeskStyle (scale 0.65)                           │
│   · IP scale: MONITOR 0.40 / MOUSE/MOUSEPAD 0.60 /           │
│               SPEAKER 0.40 / DEFAULT 0.55 / DESK_LAMP 0.40 / │
│               DESK_SHELF 0.30 / brightness<40: 0.0           │
│ 3.9 3-zone blend (composite ↔ SD output)                     │
│   · inner 65% composite / 35% SD                             │
│   · edge 25% composite / 75% SD                              │
│   · bg 0% composite / 100% SD                                │
│ 3.10 paste-back: SD 결과 → 원본 해상도 LANCZOS resize → paste│
│ 3.11 _add_shadows (composite.py)                             │
│   · contact shadow (제품 base 접지)                          │
│   · cast shadow (상단 좌측 광원 가정, 우측 아래 그림자)      │
│                                                              │
│ KEYBOARD aspect mismatch (자연 비율 < 2.0) → CV fallback     │
│ (SD 안 거치고 단순 paste)                                    │
└──────────────────────────────────────────────────────────────┘
        ↓
출력 result_image (모든 제품 배치된 책상 사진)
```

---

## 6. 카테고리 정책 (ai-server/api/config.py)

### 6.1 지원 카테고리
```
MONITOR, KEYBOARD, MOUSE, MOUSEPAD, SPEAKER,
DESK_LAMP, DESK_SHELF, LAPTOP_STAND, DECO, CLOCK, LIGHTING
```

### 6.2 주요 dict
| 상수 | 용도 |
|---|---|
| `_CATEGORY_DIMS_MM` | 카테고리 default (width_mm, depth_mm). 제품 메타데이터 없을 때 폴백 |
| `_PLACEMENT_ORDER` | 배치 순서 (낮을수록 먼저). MONITOR → KEYBOARD → MOUSEPAD → MOUSE → SPEAKER … |
| `_PREFERRED_POS` | rule_score 계산용 선호 (rx, ry) |
| `_FRONT_HEIGHT_RATIO` | 픽셀 높이/너비 비율 (upright 카테고리용) |
| `_FRONT_CATS / _BACK_CATS` | front line / back line 카테고리 분류 |
| `_CAT_ASPECT_VALID` | SD 적합 종횡비 범위. range 밖이면 CV fallback |
| `_CAT_RY_RANGE` | ry 안전 범위 (예: MONITOR 0.18~0.35 = 책상 뒤 1/3) |
| `_MIN_FRONT_SIZE` | 카테고리별 최소 픽셀 크기 |
| `_CONTACT_Y_OFFSET` | 시각적 접지 보정 |
| `_PRODUCT_FORM_TIER` | flat / semi_flat / upright 분류 |
| `_CAT_TILT_DEG` | perspective_warp depression angle (flat 카테고리) |
| `_CAT_DEPTH_SHADING` | post-blend AO 강도 |
| `_REMOVAL_PROMPT` | Stage 1 DINO 텍스트 프롬프트 |
| `_CV_ONLY_CATS` | SD 생략하고 CV composite만 사용할 카테고리 (현재 비어 있음) |

### 6.3 ry 안전 범위 (책상 깊이 정규화 좌표, 0=뒤 1=앞)
| 카테고리 | ry_min | ry_max | 비고 |
|---|---|---|---|
| MONITOR | 0.18 | 0.35 | 책상 뒤 1/3 |
| DESK_SHELF | 0.15 | 0.32 | 모니터 깊이 비슷 |
| LIGHTING | 0.05 | 0.20 | 모니터 위 (벽 가까이) |
| DESK_LAMP | 0.20 | 0.50 | 책상 뒤~중간 |
| SPEAKER | 0.18 | 0.45 | 책상 뒤~중간 |
| LAPTOP_STAND | 0.30 | 0.55 | 책상 중간 |
| DECO | 0.20 | 0.55 | 자유 |
| (기타) | 0.10 | 0.85 | default |

---

## 7. 데이터 흐름 상세

### 7.1 카테고리 정규화
```
사용자 입력 "MONITOR" 또는 "monitor"
    ↓
utils.normalize_category → "MONITOR"
    ↓
config._CATEGORY_DIMS_MM[cat] → (width, depth) 기본값
config._PLACEMENT_ORDER[cat]  → 배치 순서
config._PREFERRED_POS[cat]    → rule_score 선호 위치
config._CAT_ASPECT_VALID[cat] → SD 적합 비율
config._CAT_RY_RANGE[cat]     → ry clamp 범위
```

### 7.2 image_id 흐름
```
recommendation DB.products.id  (PostgreSQL SERIAL)
    ↓ setup["items"][cat]["id"]
recommendation_bridge.setup_to_generate_request
    ↓ ProductItem.image_id
ai-server _run_cn
    ↓ find_product_image(image_id) → Path
ai-server/data/test/processed_images/{id}.png
    ↓ Image.open → prod_alpha (RGBA)
controlnet_inpaint_processor.generate_product
    ↓ IP-Adapter reference + canny edge + silhouette mask
SD generation
```

### 7.3 mm 치수 흐름
```
recommendation DB.products.metadata (JSONB)
    {width_mm: 440, depth_mm: 150, source: "parsed"}
    ↓
recommendation_bridge._extract_size(item)
    ↓ ProductItem.width_mm / depth_mm
placement.py
    ↓ w_mm × px_per_mm × ps = fv_pw
    ↓ d_mm × px_per_mm × ps = fv_ph
front-view bbox (pixels)
```

---

## 8. API 엔드포인트

### 8.1 backend-server (Spring Boot, HTTPS)
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` `/home` | 메인 |
| GET | `/customize` | 입력 폼 |
| POST | `/api/customize` | 폼 처리 → ai-server 호출 → redirect /result?jobId=X |
| GET | `/result?jobId=X` | 결과 페이지 (JSP가 폴링) |
| GET | `/api/jobs/{id}` | ai-server `/jobs/{id}` 프록시 |

### 8.2 ai-server (FastAPI)
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 헬스 체크 |
| GET | `/styles` | 지원 StyleName enum |
| GET | `/jobs/{job_id}` | 비동기 작업 폴링 |
| POST | `/generate` | Stage 1 → 2 → 3 풀 파이프라인 |
| POST | `/recommend-and-generate` | recommendation 호출 + /generate 통합 |
| POST | `/remove` | Stage 1만 (책상 위 물체 제거) |
| POST | `/product_place` | 단일 마스크 inpainting (legacy) |
| POST | `/segment` | SAM-2 포인트 세그멘테이션 |

generation_mode:
- `controlnet` (기본): per-product SD + IP-Adapter + ControlNet + LoRA
- `cv_composite`: SD 미사용. CV alpha composite + shadows
- `placement_only`: 배치 bbox만 시각화 (디버그)
- `harmonize`: 폐기됨

### 8.3 recommendation (FastAPI)
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 헬스 체크 |
| GET | `/themes` | 지원 테마 목록 |
| POST | `/recommend` | { theme, budget, image_base64? } → setup |

---

## 9. 모델

| 모델 | HuggingFace ID / 경로 | 용도 | Stage |
|---|---|---|---|
| Grounding DINO | `IDEA-Research/grounding-dino-tiny` | 텍스트 기반 객체 검출 | 1, 2 |
| SAM-2 | `facebook/sam2.1-hiera-large` | 세그멘테이션 | 1, 2 |
| LaMa | `simple-lama-inpainting` (내장) | 제거 영역 inpainting | 1 |
| DPT-Large | `Intel/dpt-large` | depth 추출 (ControlNet 입력) | 3 |
| SD1.5 Inpaint | `runwayml/stable-diffusion-v1-5` | 제품 생성 backbone | 3 |
| ControlNet depth | `lllyasviel/sd-controlnet-depth` | 책상 구조 락 | 3 |
| ControlNet canny | `lllyasviel/sd-controlnet-canny` | 제품 외곽선 락 | 3 |
| IP-Adapter Plus | `h94/IP-Adapter` (`ip-adapter-plus_sd15.bin`) | 제품 이미지 reference | 3 |
| Style LoRA | `outputs/models/lora_external/JU_DeskStyle` (PEFT) | 책상 스타일 | 3 |
| Layout Ranker | `outputs/models/layout_ranker.pkl` (LightGBM 22 feature, AUC 0.9557) | 배치 점수 | 2 |
| Jina CLIP v2 | `jinaai/jina-clip-v2` | 텍스트 + 이미지 임베딩 | recommendation |

---

## 10. 외부 의존성

| 라이브러리 | 용도 | 버전 |
|---|---|---|
| diffusers | SD 파이프라인 | 0.37.1 |
| transformers | DPT-Large, Jina CLIP | — |
| peft | LoRA 로드 | — |
| simple-lama-inpainting | LaMa | — |
| segment-anything-2 | SAM-2 | — |
| lightgbm | learned ranker | — |
| opencv-python | mask, canny, contact shadow | — |
| FastAPI | 서버 | 0.135.3 |
| psycopg2-binary | PostgreSQL | — |
| pgvector | 벡터 검색 | — |

---

## 11. 실행

각 서버 포트는 `application.properties` / 환경 변수 / uvicorn `--port` 인자로 설정. 아래는 셸 형식 예시.

```
# recommendation
cd recommendation && (venv activate)
uvicorn deskterior.api.server:app --host 0.0.0.0 --port <PORT> --reload

# ai-server  (--reload-dir api 필수: outputs/debug 변경에 reload 트리거 방지)
cd ai-server && (venv activate)
uvicorn api.main:app --host 0.0.0.0 --port <PORT> --reload --reload-dir api

# backend-server (Gradle)
cd backend-server
gradlew bootRun
```

웹: backend-server의 HTTPS 포트로 `/customize` 접속.

스크립트 실행:
```
python ai-server/scripts/styled_test.py --style white --budget 500000 --seed 42
python ai-server/scripts/end_to_end_demo.py --mock
```

---

## 12. 환경

- **OS**: Windows
- **Python**: 3.12.x
- **Java**: 17+
- **PostgreSQL**: 16+ + pgvector
- **GPU**: CUDA 12.x + PyTorch 2.6
- **HuggingFace cache**: 사용자 home의 `.cache/huggingface/hub` (필요 시 `HF_HOME` 환경 변수로 redirect)
- **xformers**: 호환 wheel 없는 환경에서는 Jina CLIP v2 xattn 비활성화 monkey-patch 적용 ([recommendation/.../searcher.py](recommendation/deskterior/retrieval/searcher.py))

---

## 13. 폐기된 시도

| 항목 | 폐기 사유 |
|---|---|
| `harmonization_processor.py` | 환경 영역 hallucination (유령 받침대, 글로우 덩어리). 파일 보존, 호출 X |
| `sd35_inpaint_processor.py` | IP-Adapter/ControlNet 미지원으로 품질 미달 (브랜치 history) |
| `calc_products.py` | MILP 예산 최적화 프로토타입 (미사용) |
| img2img ControlNet+LoRA 책상 스타일 변환 | 사용자 책상이 낯선 책상으로 변형됨 |
| seamlessClone `/place` | 색상 왜곡, 경계 번짐 |
| Visual-RAG "100% faithfulness" framing | 사용자 의도는 reference 활용. 100% 보존 불필요 |
| GET /recommend (theme+budget only) | 사용자 사진 미반영 → POST + image_base64로 변경 |
| placement perspective foreshortening 식 (fv_ph = d_mm/desk_depth × fv_dh) | 정면 각도 사진에서 fv_dh 작아 fv_ph 거의 0. mm×px_per_mm로 단순화 |

---

## 14. 알려진 제약 / 미해결 TODO

### 14.1 SD generation 단계 magic angle (현재 미해결)
- `_CAT_TILT_DEG` (MOUSEPAD 20°, KEYBOARD 30°, MOUSE 45°, LAPTOP_STAND 50°)와 `_DEFAULT_DESK_TILT_DEG=30°` 모두 hardcoded
- 사용자 책상 카메라 각도와 무관하게 적용 → 정면 사진(depression≈14°)에 20~45° warp 적용하면 제품 과압축
- 해결안:
  - A. perspective_warp 호출 제거, 제품 cutout을 placement bbox에 fit
  - B. depression angle을 책상 detection에서 자동 계산 (sin(d) = fv_dh / (desk_depth × px_per_mm))

### 14.2 KEYBOARD CV fallback
- `_CAT_ASPECT_VALID["KEYBOARD"] = (2.0, 99.0)` 범위 기준
- DB의 cutout 자연 비율 < 2.0이면 SD 안 거치고 단순 paste → "합성 붙인 수준" 결과
- 해결안:
  - A. range를 (1.2, 99.0)으로 완화
  - B. KEYBOARD를 CV fallback 분기에서 제외
  - C. prod_alpha를 mm 비율에 맞춰 사전 stretch

### 14.3 해상도 hardcode
- SD 작업 해상도 512 고정 → 작은 모니터 영역 흐릿
- 옵션: 512→768 (VRAM 1.5배, 시간 2배) 또는 Real-ESRGAN 후처리

### 14.4 LightGBM 미설치 환경
- `pip install lightgbm` 없으면 ranker "no_model" → rule_score만 사용
- Fix #1로 ry가 정상 반영되므로 큰 문제 없음

---

## 15. 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-05-19 | 초기 ControlNet + IP-Adapter + LoRA 파이프라인 (test 브랜치) |
| 2026-05-20 | top-view 공간 분석 + placement 스코어링 도입 |
| 2026-05-21 | 3-zone blend, contact shadow 분리 |
| 2026-05-22 | LightGBM ranker 학습 (AUC 0.9557), score 블렌드 |
| 2026-05-23 | 모듈 분리 (config/utils/composite/placement), test2 브랜치 분기 |
| 2026-05-23 | SD3.5 실험 → 폐기 |
| 2026-05-23~24 | harmonization 시도 → 환경 hallucination으로 폐기 |
| 2026-05-24 | per-product SD generation 복원, IP scale 상향, negative prompt 정리 |
| 2026-05-24 | recommendation 어댑터 + LIGHTING 카테고리 추가 |
| 2026-05-24 | 전체 audit + 10가지 disconnect 점검/수정 |
| 2026-05-26 | recommendation 이미지 임베딩 (Jina CLIP v2 image encoder + xattn patch), POST /recommend |
| 2026-05-26 | placement fv_ph 식 mm×px_per_mm로 단순화 (perspective foreshortening 폐기) |
| 2026-05-27 | 원격 작업 환경 전환 |
