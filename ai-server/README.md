# Deskterior AI Server

**Visual-RAG 기반 데스크테리어 시뮬레이션** — 사용자의 책상 정면 사진과 추천 제품 목록을 받아, 그 제품들이 책상 위에 자연스럽게 배치된 합성 이미지를 생성하는 AI 서버.

> 시스템 전체 구조·각 컴포넌트 책임·기술 결정·알려진 한계는 **`docs/ARCHITECTURE.md`** 참조.

---

## 한눈에 보는 파이프라인

```
[입력]  desk_image + style + product_list (image_id, 카테고리, mm 치수)
   │
   ▼
[Stage 1] Object Removal
   Grounding DINO 검출 → SAM-2 마스크 → LaMa inpainting
   결과: 빈 책상 이미지
   │
   ▼
[Stage 2] Placement
   top-view 공간 분석 → 7×7 grid 후보 → rule_score (휴리스틱) +
   learned_score (LightGBM 22 feature ranker) 블렌드 (7:3)
   → 카테고리별 안전 범위(_CAT_RY_RANGE)로 ry clamp → front-view bbox
   결과: 제품별 (x1,y1,x2,y2) bbox
   │
   ▼
[Stage 3] Per-product SD Generation
   각 제품마다 controlnet_inpaint_processor.generate_product() 호출:
   - SD1.5 + ControlNet[depth, canny] + IP-Adapter Plus + LoRA
   - 3-zone blend로 CV pre-composite와 SD output 가중 평균
   - _add_shadows로 contact/cast shadow 추가
   │
   ▼
[출력]  최종 합성 이미지 (base64)
```

---

## 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET  | `/health` | 서버 상태 |
| GET  | `/styles` | 지원 스타일 enum |
| GET  | `/jobs/{job_id}` | 비동기 작업 폴링 |
| POST | `/generate` | **메인: Stage 1→2→3 풀 파이프라인** |
| POST | `/remove` | Stage 1만 (책상 위 물체 제거) |
| POST | `/product_place` | 단일 마스크 inpainting (legacy 호환) |
| POST | `/segment` | SAM-2 포인트 세그멘테이션 |

비동기 호출: POST 요청 시 즉시 `job_id` 반환 → `GET /jobs/{job_id}`로 폴링.

### POST `/generate` 요청 형식

```json
{
  "image_base64":          "<책상 정면 사진>",
  "style":                 "white",
  "products": [
    { "category": "MONITOR",   "name": "...", "image_id": 230, "width_mm": 600, "depth_mm": 200 },
    { "category": "KEYBOARD",  "name": "...", "image_id": 344, "width_mm": 295, "depth_mm": 132 },
    { "category": "MOUSE",     "name": "...", "image_id": 613, "width_mm": 124, "depth_mm": 84 }
  ],
  "desk_width_mm":         1400,
  "desk_depth_mm":         700,
  "top_view_image_base64": "<탑뷰 사진 (선택)>",
  "mode":                  "own_desk",
  "generation_mode":       "controlnet",
  "removal_strategy":      "combined"
}
```

| 필드 | 기본값 | 설명 |
|---|---|---|
| `style` | 필수 | `white` `black` `modern` `gaming` `cozy` `nordic` `retro` `industrial` `general` |
| `products[].image_id` | — | `data/test/processed_images/{id}.png` 파일 번호 |
| `desk_width_mm` / `desk_depth_mm` | — | 책상 실측치. mm→px 변환 기준 |
| `mode` | `own_desk` | `add` (유지+추가) / `own_desk` (전체 제거 후 추가) / `replace` / `empty_desk` |
| `generation_mode` | `controlnet` | `controlnet` (기본, per-product SD) / `cv_composite` (SD 미사용) / `placement_only` (배치만) |
| `removal_strategy` | `combined` | `combined` (마스크 합쳐 LaMa 1회) / `sequential` (제품별) / `none` |

---

## 모델

| 역할 | 모델 |
|---|---|
| 텍스트 기반 물체 검출 | Grounding DINO `IDEA-Research/grounding-dino-tiny` |
| 세그멘테이션 | SAM-2 `facebook/sam2.1-hiera-large` |
| 물체 제거 inpainting | LaMa `simple-lama-inpainting` |
| Depth 추출 | DPT-Large `Intel/dpt-large` |
| SD backbone | `runwayml/stable-diffusion-v1-5` (Inpaint) |
| ControlNet | depth `lllyasviel/sd-controlnet-depth` + canny `lllyasviel/sd-controlnet-canny` |
| IP-Adapter | Plus `h94/IP-Adapter` (`ip-adapter-plus_sd15.bin`) |
| Style LoRA | `outputs/models/lora_external/JU_DeskStyle` (PEFT) |
| Layout Ranker | `outputs/models/layout_ranker.pkl` (LightGBM, 22 feature, AUC 0.9552) |

---

## 실행

```powershell
# venv 활성화
.\venv\Scripts\Activate.ps1

# 서버 실행
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 헬스체크
curl http://localhost:8000/health
```

---

## 테스트

```powershell
# style+budget 입력 기반 임시 테스트 (recommendation 없이 CSV에서 선택)
python ai-server/scripts/styled_test.py --style white --budget 500000 --seed 42
python ai-server/scripts/styled_test.py --style black --budget 600000 --seed 42
python ai-server/scripts/styled_test.py --style gaming --budget 700000 --seed 42

# end-to-end demo (recommendation 통합 — feature/recommendation merge 후)
python ai-server/scripts/end_to_end_demo.py --mock
python ai-server/scripts/end_to_end_demo.py --color-text 화이트 --theme-text 미니멀 --budget 300000

# 통합 phase 테스트
PYTHONIOENCODING=utf-8 python ai-server/test_pipeline.py generate 3
```

---

## Recommendation 시스템 통합

`feature/recommendation` 브랜치(Python + Jina CLIP v2 + PostgreSQL+pgvector)와의 어댑터 제공.

```
[사용자 입력] 책상 사진 + 색감 + 테마 + 예산 + 카테고리
        ↓
[Retrieval]  SetupRecommender.recommend_setup()
        ↓
[Bridge]     adapters.setup_to_generate_request()
             ID 검증, style 매핑, 카테고리 매핑
        ↓
[AI 서버]    POST /generate → Stage 1→2→3
        ↓
[출력]       최종 합성 이미지
```

자세한 통합 결정사항·매핑 정책은 `docs/ARCHITECTURE.md` §2, §3 참조.

---

## 디렉토리 구조

```
ai-server/
├── api/                              # FastAPI 서버
│   ├── main.py                       # 엔드포인트 + 오케스트레이션
│   ├── models.py                     # Pydantic 스키마
│   ├── config.py                     # 카테고리 상수, _CAT_RY_RANGE 등
│   ├── utils.py                      # 공통 유틸
│   ├── object_removal_processor.py   # Stage 1 오케스트레이터
│   ├── dino_processor.py             # Grounding DINO
│   ├── sam2_processor.py             # SAM-2
│   ├── lama_processor.py             # LaMa
│   ├── space_analysis.py             # Stage 2 공간 분석
│   ├── placement.py                  # Stage 2 스코어링/배치
│   ├── composite.py                  # CV composite + _add_shadows
│   ├── controlnet_inpaint_processor.py  # ★ Stage 3 (메인 생성)
│   ├── product_inpaint_processor.py  # legacy (/product_place 전용)
│   ├── harmonization_processor.py    # DEPRECATED (보존, 호출 안 함)
│   ├── mask_utils.py
│   └── adapters/
│       ├── __init__.py
│       ├── style_mapper.py           # 한글 텍스트 → StyleName
│       └── recommendation_bridge.py  # setup → GenerateRequest
├── configs/config.yaml
├── data/test/
│   ├── desk_image.jpg                # 테스트 책상 1
│   ├── desk_image2.jpg               # 테스트 책상 2
│   ├── desk_top_image*.jpg           # 탑뷰
│   ├── products.csv
│   └── processed_images/{id}.png     # 배경 제거된 제품 PNG
├── outputs/
│   ├── debug/<timestamp>/            # 실행별 디버그
│   ├── test_results/<timestamp>/     # 통합 테스트 결과
│   ├── styled_test/                  # styled_test 결과
│   └── models/
│       ├── layout_ranker.pkl
│       └── lora_external/JU_DeskStyle/
├── scripts/
│   ├── end_to_end_demo.py            # recommendation 통합 데모
│   └── styled_test.py                # style+budget 임시 테스트
├── docs/
│   └── ARCHITECTURE.md               # ★ 전체 시스템 문서
├── logs/server_errors.log
├── test_pipeline.py
├── requirements.txt
├── README.md                         # ← 이 파일
└── CLAUDE.md                         # 개발 컨텍스트
```

---

## 환경

- Python 3.12 권장
- diffusers 0.37.1, transformers, peft (LoRA), opencv-python
- CUDA GPU 권장 (12GB+ VRAM). per-product SD 호출 시 5제품 기준 ~40~60초

---

## 핵심 결정사항 (요약)

- **per-product SD generation 채택** (controlnet_inpaint_processor)
  - 제품마다 SD 1회 호출. 3-zone blend로 CV composite와 가중 평균
  - 직전 harmonization 시도(2026-05-23) → 환경 hallucination으로 폐기

- **제품 픽셀 100% 보존 ≠ 목표**
  - DB 제품을 reference로 활용한 자연스러운 스타일 셋업이 목적
  - IP-Adapter scale 0.55~0.60으로 제품 외형 강하게 참조하되 SD가 환경에 맞게 조정

- **placement scoring 시스템**
  - top-view 공간 분석 → 7×7 grid 후보 → rule + learned score 블렌드
  - **2026-05-24 fix**: ranker ry가 front-view bbox에 실제 반영되도록 재설계

자세한 결정 이력은 `docs/ARCHITECTURE.md` §9 참조.

---

## 미해결 사항 (별도 작업)

- LightGBM 미설치 → ranker 미작동 (rule_score로 폴백 중)
- SD 작업 해상도 512 hardcode → 작은 모니터 영역 흐림 (768 상향 또는 ESRGAN 검토)
- recommendation DB id ↔ AI 서버 image_id 일치 여부 (스펙 합의 필요)
- 자세한 내용: `docs/ARCHITECTURE.md` §4.10, §4.11
