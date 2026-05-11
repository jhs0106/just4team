# Deskterior AI Server

사용자의 실제 책상 사진을 받아 원하는 스타일로 변환하고, 선택한 상품을 자연스럽게 합성하는 AI 서버.

---

## 파이프라인

```
입력: 책상 사진 + 스타일 + 상품 이미지

Step 1  /remove    책상 위 물체 제거    SAM-2 AutomaticMaskGenerator + LaMa
Step 2  /img2img   스타일 변환          SD v1.5 img2img + LoRA (JU_DeskStyle)
Step 3  /place     상품 합성            cv2.seamlessClone (픽셀 정확도 보존, GPU 불필요)
```

---

## 기술 스택

| 역할 | 모델 |
|---|---|
| 물체 감지 | SAM-2 `facebook/sam2.1-hiera-large` |
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

**Request**
```json
{
  "image_base64": "...",
  "max_area_ratio": 0.20
}
```

**완료 후 (`GET /jobs/{job_id}`)**
```json
{
  "status": "done",
  "cleaned_image": "...",
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
│   ├── img2img_processor.py        # Step 2
│   ├── place_processor.py          # Step 3
│   ├── inpaint_processor.py        # 레거시 (IP-Adapter-Plus)
│   ├── object_removal_processor.py # Step 1 오케스트레이터
│   ├── sam2_processor.py           # SAM-2
│   └── lama_processor.py           # LaMa
├── configs/
│   └── config.yaml
├── outputs/
│   └── models/lora_final/          # LoRA 가중치 (별도 배포)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
