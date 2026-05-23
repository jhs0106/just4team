# Deskterior AI Server

사용자의 실제 책상 사진(정면)과 예산·스타일 조건으로 추천된 제품 목록을 받아, 그 제품들이 책상 위에 자연스럽게 배치된 데스크테리어 시뮬레이션 이미지를 생성하는 AI 서버.

---

## 핵심 설계 원칙

**1. 제품 픽셀은 절대 변형하지 않는다 (anti-hallucination)**
- 생성형 LLM 이미지 모델(Gemini, ChatGPT image gen 등)은 제품을 텍스트로 그려내므로 hallucination 발생
- 본 시스템은 **DB의 실제 제품 이미지 픽셀을 그대로 사용** → 사용자가 보는 제품 = 실제 구매 가능한 제품
- SD(Stable Diffusion)는 제품을 *생성*하는 도구가 아니라 *조화시키는(harmonize)* 도구로만 사용

**2. SD의 역할은 seam/그림자/조명 매칭에 한정**
- CV로 합성한 책상 이미지를 SD가 받아, 제품 경계선·그림자·조명만 자연스럽게 다듬음
- per-pixel strength map으로 제품 내부는 strength=0 → SD가 절대 손대지 못함
- seam ring(경계 띠) 0.42, 그림자 영역 0.36 강도로만 SD diffusion 적용

**3. 사용자의 실제 책상을 보존**
- 책상 사진 입력 → LaMa로 기존 물체만 제거 → 책상 자체의 색감·질감·원근 유지
- 책상 스타일 변환(img2img) 단계 없음

---

## 4-Stage 파이프라인

```
[사용자 입력] desk_image + style + product_list
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1 — Object Removal                                 │
│   Grounding DINO 검출 → SAM-2 마스크 → LaMa inpainting   │
│   결과: 빈 책상 이미지 (cleaned_desk)                     │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2 — Layout & Multi-Product CV Composite            │
│   placement.py: top-view 공간 분석 + 학습된 ranker        │
│     → 제품별 bbox 위치 산출                                │
│   composite_one_with_silhouette: 모든 제품을 한 번에       │
│     cleaned_desk 위에 alpha composite                     │
│     + 각 제품의 silhouette 마스크 추출                     │
│   결과: composite_full + silhouettes[]                    │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3 — Global Harmonization Pass (단일 SD 호출)        │
│   harmonization_processor.py                              │
│                                                           │
│   1. strength_map 생성 (per-pixel float):                 │
│        제품 내부      → 0.00 (절대 보존)                   │
│        seam ring 14px → 0.42 (경계 자연스럽게)             │
│        그림자 영역 36px → 0.36 (gradient, 그림자 생성)     │
│        그 외 책상     → 0.00                              │
│                                                           │
│   2. SD1.5 ControlNet Inpaint (1회만 호출):                │
│        image=composite_full                               │
│        mask=binary(strength>0)                            │
│        control=[depth, canny] from composite_full         │
│        LoRA=JU_DeskStyle 0.30 (배경 스타일에만)            │
│                                                           │
│   3. Per-pixel differential blend:                        │
│        final = composite × (1-s) + sd_output × s          │
│        → 제품 내부 픽셀 100% 보존                          │
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

## Spring Boot 연동 시 합의 필요 사항

- `products[].category` 값이 Spring Boot DB 카테고리명과 일치하는지 확인
- `image_id`로 AI 서버 `processed_images/<id>.png`를 찾는데, Spring Boot가 이 ID를 어떻게 넘겨줄지 협의
- `top_view_image_base64`는 선택. 없으면 front-view 기반 fallback 배치로 동작
