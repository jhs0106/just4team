# Deskterior AI Server

사용자의 실제 책상 사진을 받아 원하는 스타일로 변환하고, 선택한 상품을 자연스럽게 합성하는 AI 서버.

---

## 파이프라인

```
입력: 책상 사진 + 스타일 + 상품 이미지

Step 1  /remove    책상 위 물체 제거    Grounding DINO + SAM-2 + LaMa
Step 2  /img2img   스타일 변환          SD v1.5 img2img + LoRA (JU_DeskStyle)
Step 3  /place     상품 합성            cv2.seamlessClone (픽셀 정확도 보존, GPU 불필요)

보조  /analyze    탑뷰 공간 분석       SAM-2 + 가용 영역 계산
```

---

## 기술 스택

| 역할 | 모델 |
|---|---|
| 물체 감지 (텍스트 기반) | Grounding DINO `IDEA-Research/grounding-dino-tiny` |
| 물체 세그멘테이션 | SAM-2 `facebook/sam2.1-hiera-large` |
| 물체 제거 | LaMa `simple-lama-inpainting` |
| 스타일 변환 | SD v1.5 img2img + LoRA (rank=8, alpha=32) |
| 상품 합성 | OpenCV `cv2.seamlessClone` (GPU 불필요) |

---

## 실행

```bash
docker compose up --build -d
curl http://localhost:8000/health
```

> 첫 실행 시 HuggingFace 모델 자동 다운로드 (10GB+, 수십 분 소요).
> 이후 실행부터는 캐시(`huggingface_cache` 볼륨) 사용으로 빠름.

**LoRA 모델 배치 필요** (git 미포함):
```
outputs/models/lora_final/   ← 이 경로에 수동 배치
```
없으면 LoRA 없이 실행됨 (에러 없음, 스타일 반영 약해짐).

---

## API

모든 POST 요청은 비동기. `job_id`를 즉시 반환하고 `GET /jobs/{job_id}`로 결과를 폴링.

### `GET /health`
서버 상태 확인.

### `GET /styles`
사용 가능한 스타일 목록 반환.
`white` `black` `modern` `gaming` `cozy` `nordic` `retro` `industrial`

### `GET /jobs/{job_id}`
작업 상태 조회. `status`: `pending` → `running` → `done` / `failed`

---

### `POST /remove` — Step 1: 물체 제거

감지 방식은 요청 내용에 따라 자동 선택됨.

| 조건 | 사용 방식 |
|---|---|
| `prompt` 있음 | Grounding DINO → SAM-2 (텍스트로 지정한 물체만) |
| `top_view_image_base64` 있음 | 탑뷰 SAM-2 Auto |
| 둘 다 없음 | SAM-2 Auto (전체 자동 감지) |

**Request**
```json
{
  "image_base64": "...",
  "prompt": "keyboard. mouse. pen holder.",
  "max_area_ratio": 0.20
}
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `prompt` | `null` | 제거할 물체 텍스트. `"물체1. 물체2."` 형식 |
| `top_view_image_base64` | `null` | 탑뷰 이미지 (prompt 없을 때 사용) |
| `max_area_ratio` | `0.20` | 이 비율 초과 마스크는 책상/배경으로 간주해 제외 |

**완료 후 (`GET /jobs/{job_id}`)**
```json
{
  "status": "done",
  "cleaned_image": "...",
  "mask_image": "...",
  "detection_overlay": "...",
  "num_objects": 3
}
```

---

### `POST /img2img` — Step 2: 스타일 변환

**Request**
```json
{
  "image_base64": "...",
  "prompt": "clean minimal white desk setup, white peripherals, soft lighting",
  "style": "white",
  "strength": 0.7
}
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `strength` | `0.7` | 0=원본 유지, 1=완전 재생성. 0.6~0.8 권장 |
| `num_inference_steps` | `30` | |
| `guidance_scale` | `7.5` | |

**완료 후 (`GET /jobs/{job_id}`)**
```json
{
  "status": "done",
  "result_image": "..."
}
```

---

### `POST /place` — Step 3: 상품 합성

**Request**
```json
{
  "image_base64": "...",
  "mask_base64": "...",
  "product_image_base64": "..."
}
```

마스크 흰색 영역에 상품을 리사이즈 후 `cv2.seamlessClone`으로 합성. 상품 픽셀을 그대로 보존하면서 경계 조명/색상만 자연스럽게 보정. GPU 불필요.

**완료 후 (`GET /jobs/{job_id}`)**
```json
{
  "status": "done",
  "result_image": "..."
}
```

---

### `POST /analyze` — 탑뷰 공간 분석

탑뷰 이미지에서 책상 위 점유/가용 영역을 분석하고 상품 배치 추천 좌표를 반환.

**Request**
```json
{
  "image_base64": "...",
  "desk_width_cm": 120.0,
  "desk_depth_cm": 60.0
}
```

**완료 후 (`GET /jobs/{job_id}`)**
```json
{
  "status": "done",
  "metrics": {
    "desk_area_cm2": 7200.0,
    "occupied_area_cm2": 2100.0,
    "available_area_cm2": 5100.0,
    "occupancy_ratio": 0.29,
    "available_ratio": 0.71,
    "connected_available_regions_cm2": [3200.0, 1400.0, 500.0]
  },
  "available_mask": "...",
  "occupied_mask": "...",
  "placement_center": [320, 210],
  "num_objects": 5
}
```

---

## 호출 예시 (Python)

```python
import requests, time

BASE = "http://localhost:8000"

def poll(job_id):
    while True:
        res = requests.get(f"{BASE}/jobs/{job_id}").json()
        if res["status"] in ("done", "failed"):
            return res
        time.sleep(5)

# Step 1
r = requests.post(f"{BASE}/remove", json={"image_base64": desk_b64}).json()
step1 = poll(r["job_id"])
cleaned = step1["cleaned_image"]

# Step 2
r = requests.post(f"{BASE}/img2img", json={
    "image_base64": cleaned,
    "prompt": "clean minimal white desk setup, white peripherals",
    "style": "white",
}).json()
step2 = poll(r["job_id"])
styled = step2["result_image"]
```

---

## 프로젝트 구조

```
ai-server/
├── api/
│   ├── main.py                     # FastAPI 앱, 엔드포인트
│   ├── models.py                   # Pydantic 모델
│   ├── object_removal_processor.py # Step 1 오케스트레이터
│   ├── dino_processor.py           # Grounding DINO 텍스트 기반 감지
│   ├── sam2_processor.py           # SAM-2 세그멘테이션
│   ├── lama_processor.py           # LaMa 인페인팅
│   ├── img2img_processor.py        # Step 2: SD img2img + LoRA
│   ├── place_processor.py          # Step 3: cv2.seamlessClone
│   ├── space_processor.py          # /analyze: 탑뷰 공간 분석
│   └── inpaint_processor.py        # 레거시 (IP-Adapter-Plus)
├── configs/
│   └── config.yaml
├── data/
│   └── test/                       # 테스트 이미지 (git 미포함, 직접 배치)
├── outputs/
│   └── models/lora_final/          # LoRA 가중치 (git 미포함, 별도 배포)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── test_pipeline.py                # 전체 파이프라인 통합 테스트
```
