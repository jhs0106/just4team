# Deskterior AI Server — 시스템 아키텍처

> 이 문서는 시스템 전체 구조, 각 컴포넌트의 책임, 데이터 흐름, 알려진 제약을 명시한다.
> 코드 변경 시 이 문서도 함께 갱신할 것.

---

## 0. 시스템 개요

**입력**: 사용자 책상 정면 사진 + 스타일 키워드 + 추천 제품 리스트(id, mm 치수)
**출력**: 그 책상 위에 추천 제품들이 배치된 합성 이미지

**역할 분리**:
- **외부 (Spring Boot / Jina CLIP / PostgreSQL+pgvector)**: 제품 검색 = Visual-RAG의 R(Retrieval)
- **본 AI 서버**: 책상 분석 + 제품 배치 + 합성 = Visual-RAG의 A(Augmentation) + G(Generation)

---

## 1. 전체 파이프라인 (POST /generate)

```
[입력]
  desk_image (front-view, jpg/png)
  top_view_image (선택)
  style (white|black|gaming|...)
  products[] (category, image_id, width_mm, depth_mm)
  desk_width_mm, desk_depth_mm
  mode (own_desk|add|replace|empty_desk)
  generation_mode (controlnet|cv_composite|placement_only)
                                                ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 1: Object Removal                                  │
│   object_removal_processor.py                            │
│                                                          │
│   1.1 DINO 검출: _REMOVAL_PROMPT 텍스트 매칭으로 책상 위  │
│       기존 물체 bbox 추출                                 │
│   1.2 SAM-2 세그멘테이션: bbox → 정확한 mask              │
│   1.3 max_area_ratio 필터: 너무 큰 mask 제외             │
│       (책상 자체 잡히는 것 방지)                          │
│   1.4 LaMa inpainting: mask 영역 채움 → 빈 책상           │
│                                                          │
│   출력: cleaned_front (제품 배치 가능한 빈 책상)          │
└──────────────────────────────────────────────────────────┘
                                                ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 2: Placement (위치 결정)                            │
│   placement.py: calc_placements_from_available_space()   │
│                                                          │
│   2.1 탑뷰 공간 분석 (space_analysis.py)                 │
│     - DINO로 탑뷰에서 책상 위 점유 영역 검출             │
│     - SAM-2로 책상 영역 segmentation                     │
│     - mode에 따라 remove_mask 결정                       │
│     - available_space_mask = desk_mask − keep_occupied   │
│     - connectedComponentsWithStats로 가용 region 분할     │
│                                                          │
│   2.2 후보 위치 샘플링                                    │
│     - 각 가용 region 안에 7×7 grid 후보점                │
│                                                          │
│   2.3 점수 계산 (score_region_for_product)               │
│     rule_score: 카테고리 선호 위치까지 거리 + 페널티      │
│                 (영역 위반, 근접, 가장자리)               │
│     learned_score: LightGBM ranker (22 feature)          │
│                    카테고리별 제외 — MONITOR/MOUSEPAD/    │
│                    LIGHTING는 학습 샘플 부족              │
│     final_score = rule × 0.7 + learned × 0.3             │
│                                                          │
│   2.4 최고점 → front-view bbox 변환                       │
│     _front_bbox_for_anchor: 카테고리별 anchor type        │
│     - 이전 카테고리 위치(monitor_rx, keyboard_y2 등) 참조 │
│     - 픽셀 크기: mm 치수 × px/mm × perspective scale      │
│                                                          │
│   출력: placement_items = [{product, region, score, ...}]│
└──────────────────────────────────────────────────────────┘
                                                ↓
┌──────────────────────────────────────────────────────────┐
│ Stage 3: Per-product SD Generation                       │
│   controlnet_inpaint_processor.py                        │
│   (각 제품마다 1회 SD 호출)                               │
│                                                          │
│   3.1 Context crop (bbox + padding)                      │
│   3.2 SD 호환 해상도 resize (긴 변 512, 8의 배수)         │
│   3.3 CV pre-composite: 제품을 cleaned crop 위에 합성    │
│       (canny edge 추출 + 3-zone blend 입력 용도)         │
│   3.4 ControlNet 입력 생성:                              │
│     - depth_sd: DPT-Large(cleaned crop)                  │
│     - canny_sd: Canny(composite_sd) ← 제품 외곽선         │
│   3.5 IP-Adapter 입력: 제품 이미지 letterbox 512×512      │
│   3.6 silhouette mask: 제품 alpha → inpaint 영역         │
│   3.7 SD ControlNet Inpaint 호출 (1 pass):               │
│       image=cleaned crop, mask=silhouette,               │
│       control=[depth, canny], ip_adapter=product,        │
│       prompt="{cat_desc}, {style} style, ..."            │
│       LoRA=JU_DeskStyle 0.65                             │
│   3.8 3-zone blend (composite ↔ SD output):              │
│       inner 65% / edge 25% / bg 0%                       │
│       → 제품 외형 어느 정도 보존 + SD 조명/디테일 반영    │
│   3.9 paste-back: SD 결과 → 원본 해상도 LANCZOS resize    │
│       → cleaned 이미지에 paste                            │
│                                                          │
│   3.10 _add_shadows (composite.py):                      │
│        contact shadow + cast shadow CV로 추가             │
│                                                          │
│   출력: 각 제품이 추가된 current 이미지                   │
└──────────────────────────────────────────────────────────┘
                                                ↓
[출력] result_image (모든 제품 배치된 책상 사진)
```

---

## 2. 파일 구조 및 책임

### 2.1 핵심 파이프라인 파일

| 파일 | 라인 | 책임 |
|---|---|---|
| `api/main.py` | 785 | FastAPI 앱, 엔드포인트, `_run_generate` 오케스트레이션, `_run_cn` per-product 루프 |
| `api/models.py` | 114 | Pydantic 요청/응답 스키마 |
| `api/config.py` | 184 | 카테고리 상수 (mm 치수, 종횡비, 배치 순서, 매핑) |
| `api/utils.py` | 108 | b64 변환, 카테고리 정규화, CSV 카탈로그 로드 |

### 2.2 처리 모듈

| 파일 | 라인 | 단계 | 책임 |
|---|---|---|---|
| `api/object_removal_processor.py` | 397 | Stage 1 | DINO+SAM-2+LaMa 오케스트레이션 |
| `api/dino_processor.py` | 273 | Stage 1 | Grounding DINO 텍스트 검출 |
| `api/sam2_processor.py` | 87 | Stage 1 | SAM-2 세그멘테이션 |
| `api/lama_processor.py` | 43 | Stage 1 | LaMa inpainting |
| `api/space_analysis.py` | 174 | Stage 2 | top-view 공간 분석, available_mask 빌드 |
| `api/placement.py` | 666 | Stage 2 | 후보 샘플링, 스코어링, LightGBM ranker, front-view 변환 |
| `api/composite.py` | 360 | Stage 2/3 | CV alpha composite, silhouette, contact shadow |
| `api/controlnet_inpaint_processor.py` | 518 | Stage 3 | SD1.5 + ControlNet + IP-Adapter + LoRA 제품 생성 |
| `api/mask_utils.py` | 36 | 공통 | 마스크 후처리 (small region 제거 등) |

### 2.3 보조 / 폐기

| 파일 | 상태 | 비고 |
|---|---|---|
| `api/product_inpaint_processor.py` | legacy | `/product_place` 엔드포인트 전용. `/generate` 경로엔 미사용 |
| `api/harmonization_processor.py` | **폐기** | 2026-05-23 실험. 환경 영역 hallucination 문제로 main.py에서 호출 안 함. 파일은 보존 |
| `api/adapters/` | 통합 준비 | recommendation 시스템 연결 어댑터 (style_mapper, recommendation_bridge) |

---

## 3. 주요 데이터 흐름

### 3.1 카테고리 처리 흐름

```
사용자가 요청한 카테고리 (예: "MONITOR" 또는 "monitor")
    ↓
config._CATEGORY_ALIASES → 정규화된 카테고리 코드 ("MONITOR")
    ↓
config._CATEGORY_DIMS_MM[cat] → (width_mm, depth_mm) 기본값
config._PLACEMENT_ORDER[cat]  → 배치 순서 (낮을수록 먼저)
config._PREFERRED_POS[cat]    → rule_score 계산용 (rx, ry)
config._FRONT_HEIGHT_RATIO[cat] → 픽셀 높이/너비 비율
config._FRONT_CATS / _BACK_CATS → 카테고리 분류
config._CAT_ASPECT_VALID[cat] → SD 적합 종횡비 범위
    ↓
placement.score_region_for_product → 점수
placement._front_bbox_for_anchor   → front-view bbox
    ↓
controlnet_inpaint_processor.generate_product → SD 호출
composite._add_shadows                          → 그림자
```

### 3.2 ID 흐름

```
recommendation DB
    ↓ id (PostgreSQL SERIAL)
recommendation SetupRecommender
    ↓ items[cat]["id"]
adapters.setup_to_generate_request
    ↓ ProductItem.image_id
main._run_cn
    ↓ find_product_image(image_id) → Path
processed_images/{id}.png 로드
```

---

## 4. 알려진 disconnect / 한계 — 2026-05-24 audit 결과

> 모든 항목 fix 완료. 각 fix는 commit history와 코드 코멘트 참조.

### 4.1 [FIX 완료] placement ranker ry → front-view 반영

**문제**: LightGBM ranker가 22 feature로 최적 (rx, ry) 골라도, MONITOR/KEYBOARD/MOUSE/DESK_LAMP/DESK_SHELF의 y 좌표는 `_front_bbox_for_anchor` 안에서 하드코딩 값(28%/62%/45%/35%)으로 덮어쓰임.

**Fix**: `_front_bbox_for_anchor`를 ry-based로 재설계.
- 모든 카테고리가 ranker가 산출한 ry를 base로 사용
- `_CAT_RY_RANGE` (config.py) 상수로 카테고리별 안전 범위 clamp
- 카테고리간 관계(MONITOR 위치 → KEYBOARD가 그 아래로, KEYBOARD 위치 → MOUSE가 옆에) 강제 유지

**관련 파일**: `placement.py:96-200` (_front_bbox_for_anchor), `config.py:_CAT_RY_RANGE`

### 4.2 [FIX 완료] LIGHTING 카테고리 front-view 분기 추가

**문제**: config.py 14개 dict에 LIGHTING 추가됐지만 `_front_bbox_for_anchor`에 분기 없음 → fallback 경로로 빠짐.

**Fix**: `_front_bbox_for_anchor`에 LIGHTING 전용 분기 추가.
- 모니터 가로폭 비슷 (fv_dw × 0.45)
- 매우 얇은 높이 (라이트바)
- monitor_rx 따라 x 정렬
- monitor_contact_y 위쪽으로 강제 (라이트바 하단이 모니터 상단 살짝 가림)

**관련 파일**: `placement.py:_front_bbox_for_anchor` LIGHTING 분기, `config.py:_CAT_RY_RANGE["LIGHTING"]`

### 4.3 [FIX 완료] harmonization_processor 폐기 처리

**문제**: 2026-05-23~24 실험 → 폐기됐는데 파일·import 남아 있음.

**Fix**:
- 파일 상단에 큰 경고 코멘트 박스 (DEPRECATED 명시)
- `main.py`에서 import 제거
- `models.py`의 generation_mode 설명에서 harmonize 옵션 제거

**관련 파일**: `harmonization_processor.py` 헤더, `main.py:62-66`, `models.py:103`

### 4.4 [FIX 완료] recommendation 어댑터 ID 검증 추가

**문제**: image_id가 실제 파일에 있는지 어댑터 단계에서 검증 안 함.

**Fix**: `setup_to_generate_request`에 `verify_images=True, on_missing="skip"` 옵션 추가.
- `find_product_image(id)`로 파일 존재 확인
- 누락 시 skip 또는 raise 선택 가능
- 누락 ID 목록 콘솔 출력

**관련 파일**: `adapters/recommendation_bridge.py`

### 4.5 [FIX 완료] negative prompt 정리 + 의도 명시

**문제**: 카테고리별 negative prompt 누적, "screen content"가 모니터 화면 다 죽이는 부작용.

**Fix**:
- `_NEGATIVE_PROMPT`에 정책 코멘트 추가 (의도 + 변경 이력)
- `_CAT_NEGATIVE` 각 카테고리에 차단 패턴 의도 명시
- 화면 콘텐츠 차단 키워드 제거 (모니터 자연 표시 허용)

**관련 파일**: `controlnet_inpaint_processor.py:38-82`

### 4.6 [FIX 완료] IP-Adapter scale 근거 명시

**문제**: 카테고리별 ip_scale 값의 근거 없음.

**Fix**: `_run_cn`에 카테고리별 scale 결정 표 코멘트.
| Category | Scale | Reason |
|---|---|---|
| MONITOR | 0.60 | QR/JG79 hallucination 방지 (2026-05-24 사태 후 0.35→0.60) |
| MOUSE/MOUSEPAD | 0.60 | 작은 디테일 보존 |
| SPEAKER | 0.55 | 형태 다양성 |
| DEFAULT | 0.55 | 안전 기본값 |
| DESK_LAMP | 0.40 | 음영 어색 방지 |
| DESK_SHELF | 0.35 | 단순 구조 |
| brightness<40 | 0.0 | lifestyle 컷 |

**관련 파일**: `main.py:_run_cn` ip_scale 결정 블록

### 4.7 [FIX 완료] 3-zone blend 가중치 정책 명시

**문제**: inner 65% / edge 25% / bg 0% — 근거·의도 코드에 없음.

**Fix**: `controlnet_inpaint_processor.py`에 zone별 가중치 의도 표 코멘트.
| Zone | Composite 비중 | SD 비중 | 의도 |
|---|---|---|---|
| inner | 0.65 | 0.35 | DB 픽셀 보존 (얼굴/로고 인식) |
| edge | 0.25 | 0.75 | 책상 톤 융합 |
| bg | 0.00 | 1.00 | SD 자유 |

**관련 파일**: `controlnet_inpaint_processor.py:_blend_w 블록`

### 4.8 [FIX 완료] _add_shadows 광원 방향 명시

**문제**: cast shadow가 우측 아래 고정. 실제 책상 사진 광원과 무관.

**Fix**: `_add_shadows` 함수 헤더에 광원 방향 정책 코멘트.
- 가정: 상단 좌측 광원 (실내 형광등/창문 자연스러운 위치)
- 한계: 사진별 실제 광원과 불일치 가능. 추후 자동 추정 작업

**관련 파일**: `composite.py:_add_shadows`

### 4.9 [FIX 완료] LaMa removal prompt 위험 단어 정리

**문제**: "light bar. screen bar. monitor light." 같은 합성어가 DINO에서 "light" 부분 매칭으로 책상 밝은 영역 false-positive 위험.

**Fix**:
- 위 3개 단어 제거 (LIGHTING은 입력 시에만 생성, 제거 대상 아님)
- `_REMOVAL_PROMPT`에 정책 코멘트 추가 (단어 추가 시 검증 필수)

**관련 파일**: `config.py:_REMOVAL_PROMPT`

### 4.10 [별도 TODO] 해상도 hardcode

**문제**: SD 작업 해상도 512 고정 → 작은 모니터 영역 흐릿.

**상태**: 별도 작업으로 진행 (현재 audit 범위 외). 옵션:
- SD 작업 해상도 512→768 (VRAM 1.5배, 시간 2배)
- Real-ESRGAN 후처리 (추가 의존성)

### 4.11 [별도 TODO] LightGBM 미설치

**문제**: 현재 venv에 lightgbm 패키지 없음 → ranker는 항상 "no_model"로 skip되고 rule_score만 사용됨.

**상태**: 별도 작업. `pip install lightgbm` 또는 ranker 자체 폐기 결정 필요.
영향: 현재 시스템은 rule_score만으로 동작 중. Fix #1로 ry가 정상 반영되므로 큰 문제 없음.

---

## 5. 외부 의존성

| 라이브러리 | 용도 | 버전 |
|---|---|---|
| diffusers | SD 파이프라인 | 0.37.1 |
| transformers | DPT-Large depth | — |
| peft | LoRA 로드 | — |
| simple-lama-inpainting | LaMa | — |
| segment-anything-2 | SAM-2 | — |
| lightgbm | learned ranker | — |
| opencv-python | mask 처리, canny, contact shadow | — |
| FastAPI | 서버 | 0.135.3 |

| 모델 | HuggingFace ID / 경로 |
|---|---|
| Grounding DINO | `IDEA-Research/grounding-dino-tiny` |
| SAM-2 | `facebook/sam2.1-hiera-large` |
| LaMa | `simple-lama-inpainting` (내장 모델) |
| DPT-Large | `Intel/dpt-large` |
| SD1.5 Inpaint | `runwayml/stable-diffusion-v1-5` |
| ControlNet depth | `lllyasviel/sd-controlnet-depth` |
| ControlNet canny | `lllyasviel/sd-controlnet-canny` |
| IP-Adapter Plus | `h94/IP-Adapter` (`ip-adapter-plus_sd15.bin`) |
| Style LoRA | `outputs/models/lora_external/JU_DeskStyle` (PEFT) |
| Layout Ranker | `outputs/models/layout_ranker.pkl` (LightGBM, AUC 0.9552) |

---

## 6. API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 서버 상태 |
| GET | `/styles` | 지원 스타일 enum |
| GET | `/jobs/{job_id}` | 비동기 작업 폴링 |
| POST | `/generate` | **메인: Stage 1→2→3 풀 파이프라인** |
| POST | `/remove` | Stage 1만 (책상 위 물체 제거) |
| POST | `/product_place` | 단일 마스크 inpainting (legacy 호환용) |
| POST | `/segment` | SAM-2 포인트 세그멘테이션 |

`/generate`의 `generation_mode`:
- `controlnet` (기본): per-product SD 생성 (IP-Adapter + ControlNet + LoRA)
- `cv_composite`: SD 미사용. CV alpha composite + `_add_shadows`만
- `placement_only`: 배치 bbox만 시각화 (디버그용)
- `harmonize`: **폐기됨.** 환경 영역 hallucination 발생. 호출 권장 안 함

---

## 7. 디렉토리 구조

```
ai-server/
├── api/                              # FastAPI 서버 코드
│   ├── main.py                       # 엔드포인트 + 오케스트레이션
│   ├── models.py                     # Pydantic 스키마
│   ├── config.py                     # 카테고리 상수
│   ├── utils.py                      # 공통 유틸
│   ├── object_removal_processor.py   # Stage 1
│   ├── dino_processor.py
│   ├── sam2_processor.py
│   ├── lama_processor.py
│   ├── space_analysis.py             # Stage 2 공간 분석
│   ├── placement.py                  # Stage 2 스코어링/배치
│   ├── composite.py                  # Stage 2/3 CV ops
│   ├── controlnet_inpaint_processor.py  # Stage 3 SD
│   ├── product_inpaint_processor.py  # legacy
│   ├── harmonization_processor.py    # 폐기 (보존)
│   ├── mask_utils.py
│   └── adapters/
│       ├── __init__.py
│       ├── style_mapper.py           # 한글 텍스트 → StyleName
│       └── recommendation_bridge.py  # setup → GenerateRequest
├── configs/
│   └── config.yaml                   # 서버 런타임 설정
├── data/
│   └── test/
│       ├── desk_image.jpg            # 테스트 책상 1
│       ├── desk_image2.jpg           # 테스트 책상 2
│       ├── desk_top_image.jpg
│       ├── desk_top_image2.jpg
│       ├── products.csv              # 제품 카탈로그
│       └── processed_images/{id}.png # 배경 제거된 제품 PNG
├── outputs/
│   ├── debug/<timestamp>/            # 실행별 디버그
│   ├── test_results/<timestamp>/     # 테스트 결과
│   ├── styled_test/                  # styled_test 스크립트 결과
│   └── models/
│       ├── layout_ranker.pkl
│       └── lora_external/JU_DeskStyle/
├── scripts/
│   ├── end_to_end_demo.py            # recommendation 통합 데모
│   └── styled_test.py                # style+budget 임시 테스트
├── docs/
│   └── ARCHITECTURE.md               # ★ 이 문서
├── logs/
│   └── server_errors.log
├── test_pipeline.py                  # 통합 테스트
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## 8. 핵심 개념 / 용어

| 용어 | 정의 |
|---|---|
| **front-view** | 사용자가 촬영한 책상 정면 사진 |
| **top-view** | 책상을 위에서 본 사진 (선택. 공간 분석에 사용) |
| **available_space_mask** | 책상 영역에서 점유 영역(occupied)을 뺀 마스크 |
| **rx, ry** | 책상 bbox 기준으로 정규화된 좌표 (0~1) |
| **rule_score** | 휴리스틱 점수 (선호 위치까지 거리 + 페널티) |
| **learned_score** | LightGBM ranker가 산출한 점수 |
| **final_score** | `rule × 0.7 + learned × 0.3` |
| **relation_state** | 카테고리간 의존성을 누적하는 dict (monitor_rx, keyboard_y2 등) |
| **3-zone blend** | SD 결과와 CV composite를 inner(65%)/edge(25%)/bg(0%) 가중 평균 |
| **anchor type** | 카테고리별 배치 기준점 (base_contact, back_contact 등) |

---

## 9. 변경 이력

| 날짜 | 변경 | 비고 |
|---|---|---|
| 2026-05-19 | 초기 파이프라인 (ControlNet + IP-Adapter + LoRA) | test 브랜치 |
| 2026-05-20 | placement 스코어링, top-view 공간 분석 도입 | |
| 2026-05-21 | 3-zone blend, contact shadow 분리, phase 1/2/3 테스트 | |
| 2026-05-22 | LightGBM ranker 학습 + 통합 (AUC 0.9552) | |
| 2026-05-23 | 모듈 분리 (config/utils/composite/placement) | test2 브랜치 |
| 2026-05-23 | SD3.5 backbone 실험 → 폐기 | conditioning 미흡 |
| 2026-05-23 | harmonization 접근 시도 → 폐기 | 환경 hallucination |
| 2026-05-24 | per-product SD generation 복원, IP scale 상향, negative prompt 정리 | |
| 2026-05-24 | recommendation 어댑터, LIGHTING 카테고리 추가 | |
| 2026-05-24 | **전체 audit + 10가지 disconnect 점검/수정** | 본 작업 |

---

> 다음 섹션은 fix 진행 후 갱신 (Section 4의 각 항목 [FIX 완료]로 변경).
