# Deskterior AI Server

**프로젝트 주제: Visual-RAG 기반 데스크테리어 시뮬레이션 모델 설계 및 구현**

사용자의 실제 책상 사진(정면)과 예산·스타일 조건으로 추천된 제품 목록을 받아, 그 제품들이 책상 위에 자연스럽게 배치된 데스크테리어 시뮬레이션 이미지를 생성하는 AI 서버.

---

## ★ Architecture: Visual-RAG

본 시스템은 텍스트 도메인의 **RAG(Retrieval-Augmented Generation)** 아키텍처 패턴을 **visual modality**로 확장한 **Visual-RAG compound AI system**이다.

```
Text-RAG (LLM):
  query → [Retrieval] Vector DB → [Augmentation] context 주입 → [Generation] LLM 생성
                                                                  ↑
                                              retrieved fact가 hallucination 억제

Visual-RAG (본 시스템):
  style + budget → [Retrieval] Jina CLIP + product DB
                   → [Augmentation] 실제 제품 PNG 픽셀을 CV로 책상에 합성
                                                ↑
                                       retrieved fact를 generation context에 강제 주입
                   → [Generation] SD1.5 ControlNet Inpaint
                                                ↑
                                       제품 픽셀 주변(seam/그림자/조명)만 conditional generation
```

### RAG 구성요소별 매핑

| RAG 단계 | Text-RAG | Visual-RAG (본 시스템) | 구현 위치 |
|---|---|---|---|
| **R**etrieval | Vector DB 유사도 검색 | Jina CLIP + product DB 검색 | Spring Boot 서버 |
| **A**ugmentation | retrieved doc를 prompt context에 삽입 | 제품 PNG를 CV alpha composite으로 책상에 주입 | `composite.composite_one_with_silhouette()` |
| **G**eneration | LLM이 context 기반 응답 생성 | SD가 주입된 제품 주변(배경/seam/그림자)만 생성 | `harmonization_processor.harmonize()` |
| **Faithfulness guarantee** | retrieved doc 인용 (원본 보존) | per-pixel strength_map의 제품 영역 = 0.0 (픽셀 불변) | `_build_strength_map()` |

### 설계 원칙

**1. Retrieval-augmented = 제품 identity hallucination 제거**
- 순수 생성 모델(Gemini Nano Banana, ChatGPT image gen 등)은 제품을 텍스트로 그려내므로 hallucination 발생 — 키 배열·로고·베젤 두께 어긋남
- Visual-RAG는 **DB의 실제 제품 PNG 픽셀을 그대로 retrieval하여 generation context에 inject** → 사용자가 보는 제품 = 실제 구매 가능한 제품 (픽셀 단위 일치)
- 이는 RAG에서 retrieved document가 변형 없이 인용되는 것과 동일한 architectural guarantee

**2. SD는 G(generation) 컴포넌트, conditional generation 담당**
- SD는 제품을 *생성*하지 않음. 대신 retrieved fact 주변의 환경(seam, 그림자, 조명)을 *조건부 생성*
- per-pixel strength_map: 제품 내부 = 0.00 (불변 보장), seam ring = 0.42, 그림자 = 0.36
- differential blend로 SD output을 픽셀 단위 가중치 적용 → retrieval된 사실 보존 + 환경 자연스럽게 합성

**3. 사용자의 실제 책상 보존**
- LaMa로 기존 물체만 제거 → 책상 자체의 색감·질감·원근 유지
- 책상 스타일 변환(img2img) 단계 없음

### Hallucination 통제 범위

| 영역 | hallucination 가능성 | 통제 방식 |
|---|---|---|
| 제품 외형·로고·디테일 | **0 (불가능)** | strength_map 제품 영역 = 0.0 |
| 책상 자체 | **0 (불가능)** | strength_map 책상 영역 = 0.0 |
| 제품 경계 seam 5~15px | 미세함 | ControlNet depth+canny로 구조 락 + strength 0.42 제한 |
| 그림자·접지 영역 36px | 제한적 | ControlNet + strength 0.36 + gradient fade |

순수 SD 생성 대비 **hallucination 영역을 화면의 ~10% 이하로 축소**하고, 그 안에서도 ControlNet으로 구조를 lock한다.

---

## 4-Stage 파이프라인

```
[사용자 입력] desk_image + style + product_list
        │
        │  ※ Retrieval(R) 단계는 Spring Boot + Jina CLIP에서 선행 수행 →
        │     product_list[image_id]가 retrieved fact (실제 제품 PNG 참조)
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1 — Object Removal (전처리)                         │
│   Grounding DINO 검출 → SAM-2 마스크 → LaMa inpainting   │
│   결과: 빈 책상 이미지 (cleaned_desk)                     │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2 — [Augmentation] Retrieval Injection             │
│   placement.py: top-view 공간 분석 + LightGBM ranker      │
│     → 제품별 bbox 위치 결정                                │
│   composite_one_with_silhouette:                          │
│     - retrieved 제품 PNG를 cleaned_desk에 alpha composite │
│     - 모든 제품 한 번에 합성                                │
│     - 각 제품 silhouette mask 추출 (Stage 3 입력용)         │
│   결과: composite_full (RAG의 augmented context) +         │
│         silhouettes[] (faithfulness mask)                 │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3 — [Generation] Conditional Generation            │
│   harmonization_processor.py (단일 SD 호출)               │
│                                                           │
│   1. strength_map 생성 (faithfulness 보장):                │
│        제품 내부      → 0.00 (retrieved fact 불변)         │
│        seam ring 14px → 0.42 (환경 경계 생성)              │
│        그림자 영역 36px → 0.36 gradient (그림자 생성)      │
│        그 외 책상     → 0.00 (책상 자체 불변)              │
│                                                           │
│   2. SD1.5 ControlNet Inpaint (1회만):                     │
│        image=composite_full       (= augmented context)   │
│        mask=binary(strength>0)    (= generation 영역 한정) │
│        control=[depth, canny]     (= 구조 제약)            │
│        LoRA=JU_DeskStyle 0.30                              │
│                                                           │
│   3. Per-pixel differential blend (faithfulness 강제):     │
│        final = composite × (1-s) + sd_output × s          │
│        s=0 → retrieved fact 100% 보존                      │
│        s>0 → SD output 적용 (환경만)                       │
│                                                           │
│   결과: harmonized_image                                  │
└──────────────────────────────────────────────────────────┘
        │
        ▼
[출력] result_image (base64)
```

### 이전 방식 대비 이점

| 항목 | 옛 방식 (per-product SD 생성) | 새 방식 (harmonize) |
|---|---|---|
| 제품 픽셀 정확도 | ~60% (IP-Adapter 참조 생성) | **100%** (CV composite) |
| SD 호출 횟수 | N개 제품 × 1회 (5~10회) | **1회** (제품 수 무관) |
| 제품 간 조명 일관성 | 제품마다 따로 → 일관성 깨짐 | **하나의 SD pass로 통일** |
| seam 후처리 | 3-zone blend, rim darkening 등 | **strength_map 자체가 해결** |
| Inference 시간 (5개 기준) | ~50초 | **~12초 (4×)** |

---

## 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET  | `/health` | 서버 상태 |
| GET  | `/styles` | 지원 스타일 목록 |
| GET  | `/jobs/{job_id}` | 비동기 작업 결과 폴링 |
| POST | `/generate` | **권장: 4-stage 풀 파이프라인** |
| POST | `/remove` | Stage 1만 (책상 위 물체 제거) |
| POST | `/product_place` | 단일 영역 inpainting (legacy, 호환용) |
| POST | `/segment` | SAM-2 포인트 세그멘테이션 |

비동기 호출: POST 요청 시 즉시 `job_id` 반환 → `GET /jobs/{job_id}`로 폴링.

---

### POST `/generate` — 4-stage 풀 파이프라인 (권장)

```json
{
  "image_base64":          "<책상 정면 사진>",
  "style":                 "white",
  "products": [
    { "category": "MONITOR",   "name": "LG 27인치", "image_id": 236, "width_mm": 600, "depth_mm": 200 },
    { "category": "KEYBOARD",  "name": "MX Keys",  "image_id": 441, "width_mm": 295, "depth_mm": 132 },
    { "category": "MOUSE",     "name": "MX Master","image_id": 557, "width_mm": 124, "depth_mm": 84 },
    { "category": "SPEAKER",   "name": "...",      "image_id": 1472, "width_mm": 100, "depth_mm": 120 },
    { "category": "DESK_LAMP", "name": "...",      "image_id": 1881, "width_mm": 80,  "depth_mm": 80 }
  ],
  "desk_width_mm":         1400,
  "desk_depth_mm":         700,
  "top_view_image_base64": "<탑뷰 사진 — 선택>",
  "mode":                  "own_desk",
  "generation_mode":       "harmonize",
  "removal_strategy":      "combined"
}
```

| 필드 | 기본값 | 설명 |
|---|---|---|
| `style` | 필수 | `white` `black` `modern` `gaming` `cozy` `nordic` `retro` `industrial` `general` |
| `products[].image_id` | — | `data/test/processed_images/{id}.png` 파일 번호 |
| `products[].width_mm`/`depth_mm` | — | 실측치. 없으면 CSV 카탈로그 기본값 사용 |
| `desk_width_mm`/`desk_depth_mm` | — | 책상 실측치. 픽셀 스케일 변환에 사용 |
| `mode` | `own_desk` | `add` (유지+추가) / `own_desk` (전체 제거+추가) / `replace` / `empty_desk` |
| `generation_mode` | `harmonize` | **`harmonize`** (기본, 신규) / `cv_composite` (SD 미사용) / `placement_only` (배치만) |
| `removal_strategy` | `combined` | `combined` (마스크 합쳐 1회 LaMa) / `sequential` (제품별) / `none` |

**완료 응답:**
```json
{
  "status":          "done",
  "cleaned_image":   "<Stage 1 결과>",
  "result_image":    "<Stage 3 최종 결과>",
  "num_removed":     3,
  "num_placed":      5
}
```

**디버그 출력 (`outputs/debug/<timestamp>/`):**
- `cleaned_front.png` — Stage 1 결과
- `composite_full.png` — Stage 2 결과 (제품 전부 합성)
- `harmonize_strength_map.png` — Stage 3 per-pixel strength
- `harmonize_binary_mask.png` — SD inpaint 마스크
- `harmonize_depth.png` / `harmonize_canny.png` — ControlNet 입력
- `harmonize_sd_raw.png` — SD 출력 (blend 전)
- `harmonize_final.png` — differential blend 후 최종
- `harmonize_meta.json` — 모든 하이퍼파라미터·프롬프트 기록
- `products_list.json` — 제품별 배치 정보·점수
- `placement_debug.png` — 배치 시각화 (bbox + 점수)

---

### POST `/remove`, `/product_place`, `/segment`

레거시·서브 엔드포인트. 자세한 설명은 [api/main.py](api/main.py) 참고.

---

## 기술 스택

| 역할 | 모델 |
|---|---|
| 텍스트 기반 물체 검출 | Grounding DINO `IDEA-Research/grounding-dino-tiny` |
| 세그멘테이션 | SAM-2 `facebook/sam2.1-hiera-large` |
| 물체 제거 inpainting | LaMa `simple-lama-inpainting` |
| Depth 추출 | DPT-Large `Intel/dpt-large` |
| Harmonization SD backbone | `runwayml/stable-diffusion-v1-5` (Inpaint) |
| ControlNet | depth `lllyasviel/sd-controlnet-depth` + canny `lllyasviel/sd-controlnet-canny` |
| 배치 위치 ranker | LightGBM (22 features, `outputs/models/layout_ranker.txt`) |
| Style LoRA (선택) | `outputs/models/lora_external/JU_DeskStyle` |

---

## 실행

```powershell
# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 서버 실행
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 헬스체크
curl http://localhost:8000/health
```

**테스트 (golden path):**
```powershell
python test_pipeline.py
# 또는 멀티 스타일
python test_pipeline.py multi white,black 3
```

---

## 프로젝트 구조

```
ai-server/
├── api/
│   ├── main.py                     # FastAPI 앱 + 엔드포인트 + _run_generate
│   ├── models.py                   # Pydantic 요청/응답
│   ├── config.py                   # 카테고리 상수 (mm, 종횡비, 배치 순서)
│   ├── utils.py                    # b64, 카테고리 normalize, CSV 카탈로그
│   ├── composite.py                # CV composite + silhouette + _add_shadows
│   ├── placement.py                # 공간 분석 + 학습된 layout ranker (22 feat)
│   ├── space_analysis.py           # top-view 가용 공간 분석
│   ├── harmonization_processor.py  # ★ Stage 3 핵심: SD1.5 ControlNet harmonize
│   ├── object_removal_processor.py # Stage 1 오케스트레이터
│   ├── dino_processor.py           # Grounding DINO
│   ├── sam2_processor.py           # SAM-2
│   ├── lama_processor.py           # LaMa
│   ├── product_inpaint_processor.py # /product_place 엔드포인트 전용 (legacy)
│   └── mask_utils.py
├── configs/
│   └── config.yaml
├── data/test/
│   ├── desk_image.jpg              # 테스트용 책상 사진 (git 미포함)
│   ├── products.csv                # 제품 카탈로그
│   └── processed_images/<id>.png   # 제품 이미지 (alpha 채널 권장)
├── outputs/
│   ├── debug/<timestamp>/          # 실행별 디버그 산출물
│   ├── test_results/               # 테스트 파이프라인 결과
│   └── models/
│       ├── layout_ranker.txt       # LightGBM 가중치
│       └── lora_external/          # JU_DeskStyle LoRA (PEFT)
├── logs/
│   └── server_errors.log
├── requirements.txt
├── test_pipeline.py                # 통합 테스트 스크립트
├── README.md                       # ★ 이 파일
└── CLAUDE.md                       # 개발 컨텍스트 (Claude Code용)
```

---

## 환경

- Python 3.12 권장
- diffusers 0.37.1, transformers, peft (LoRA), lightgbm (ranker)
- CUDA GPU 권장 (12GB+ VRAM). harmonize는 단일 SD pass라 7~8GB로도 동작 가능.

---

## Recommendation 시스템 통합 (2026-05-24)

`feature/recommendation` 브랜치(Jina CLIP v2 + PostgreSQL + pgvector 기반)와의 통합 어댑터를 추가함.

```
[사용자 입력]               책상 사진 + 색감 + 테마 + 예산 + 카테고리
        ↓
[Retrieval]                 SetupRecommender.recommend_setup()
                            → setup dict (top-k 제품 조합)
        ↓
[Bridge]                    adapters.setup_to_generate_request()
                            → GenerateRequest 변환
        ↓
[AI 서버 Visual-RAG]        POST /generate
                            → harmonize 파이프라인 (Stage 1→2→3)
        ↓
[출력]                      최종 합성 이미지
```

**카테고리 매핑** (`api/adapters/recommendation_bridge.py`):
- 10개 카테고리 1:1 매핑 (MONITOR/KEYBOARD/MOUSE/MOUSEPAD/SPEAKER/DESK_LAMP/DESK_SHELF/LAPTOP_STAND/CLOCK/DECO)
- LIGHTING (모니터 위 라이트바): AI 서버에 정식 추가
- DESK / MONITOR_ARM: skip (DESK는 사용자 사진 사용, MONITOR_ARM은 front-view에서 안 보임)

**Style 매핑** (`api/adapters/style_mapper.py`):
- 자유 한글 텍스트(`color_text`, `theme_text`) → StyleName enum
- theme 우선, 매칭 없으면 color, 둘 다 없으면 general

**End-to-end 데모**:
```powershell
# mock setup으로 어댑터 단독 테스트
python ai-server/scripts/end_to_end_demo.py --mock

# 실제 recommendation 모듈 사용 (feature/recommendation merge 후)
python ai-server/scripts/end_to_end_demo.py `
    --color-text 화이트 `
    --theme-text 미니멀 `
    --budget 300000 `
    --categories MONITOR,KEYBOARD,MOUSE,DESK_LAMP,SPEAKER
```

**아직 미합의 사항**:
- recommendation DB의 product.id와 AI 서버의 `processed_images/<id>.png` ID 체계 일치 여부
- 책상 사진/크기 입력 UI 위치 (확인됨: 제품 추천 전 가장 초반 단계)
- 제품 이미지 동기화 시점 (옵션 A 채택: recommendation이 사전 배경 제거)
