# Deskterior AI Server

사용자의 실제 책상 사진(정면)을 받아, 예산·스타일 조건에 맞게 추천된 제품들이 책상 위에 배치된 데스크테리어 이미지를 생성하는 AI 서버.

---

## 서비스 흐름

```
사용자 입력: 책상 사진(정면) + 예산 + 스타일
    ↓
Spring Boot + Jina CLIP: 예산·스타일에 맞는 제품 추천 → 제품 목록 AI 서버로 전달
    ↓
AI 서버: 책상 위 기존 물체 제거 → 추천 제품 배치 이미지 생성
    ↓
사용자: 생성 이미지(레이아웃 시뮬레이션) + 실제 제품 카드(구매 링크) 나란히 표시
```

> 이미지는 "이 책상에 이 카테고리 제품들을 올려두면 이런 셋업이 됨"을 보여주는 레이아웃 시뮬레이션입니다.
> 이미지 속 제품이 실제 DB 제품과 픽셀 단위로 동일할 필요는 없습니다. 실제 제품 정보는 별도 카드로 표시됩니다.

---

## 파이프라인

```
Step 1  POST /remove         책상 위 기존 물체 제거    Grounding DINO + SAM-2 + LaMa
Step 2  POST /product_place  제품 inpainting 배치      SD Inpainting (제품 수만큼 반복)

보조    POST /segment        마우스패드 세그멘테이션    SAM-2 (px/mm 스케일 계산용)
```

---

## 기술 스택

| 역할 | 모델 |
|---|---|
| 물체 감지 (텍스트 기반) | Grounding DINO `IDEA-Research/grounding-dino-tiny` |
| 물체 세그멘테이션 | SAM-2 `facebook/sam2.1-hiera-large` |
| 물체 제거 | LaMa `simple-lama-inpainting` |
| 제품 배치 inpainting | SD Inpainting `runwayml/stable-diffusion-inpainting` |

---

## 실행

```bash
# venv 활성화
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# 또는
.\venv\Scripts\activate.bat   # Windows CMD

# 서버 실행 (--reload: 코드 변경 시 자동 반영)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API

모든 POST 요청은 비동기입니다. `job_id`를 즉시 반환하고 `GET /jobs/{job_id}`로 결과를 폴링합니다.

### `GET /health`
서버 상태 확인.

### `GET /styles`
지원 스타일 목록: `white` `black` `modern` `gaming` `cozy` `nordic` `retro` `industrial`

### `GET /jobs/{job_id}`
작업 상태 조회. `status`: `pending` → `running` → `done` / `failed`

---

### `POST /remove` — Step 1: 물체 제거

감지 방식은 요청 내용에 따라 자동 선택됩니다.

| 조건 | 사용 방식 |
|---|---|
| `prompt` 있음 | Grounding DINO → SAM-2 (지정한 물체만) |
| `top_view_image_base64` 있음 | 탑뷰 SAM-2 Auto |
| 둘 다 없음 | SAM-2 Auto (전체 자동 감지) |

**Request**
```json
{
  "image_base64": "...",
  "prompt": "keyboard. mouse. monitor. headset.",
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

### `POST /product_place` — Step 2: 제품 배치

마스크 영역에 텍스트 프롬프트 기반으로 제품을 생성합니다. 제품 수만큼 반복 호출합니다.

**Request**
```json
{
  "image_base64": "...",
  "mask_base64": "...",
  "prompt": "white mechanical keyboard on desk mat"
}
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `prompt` | 필수 | 제품 설명. 스타일 포함 권장 (예: `"white wireless mouse"`) |
| `num_inference_steps` | `30` | |
| `guidance_scale` | `12.0` | 높을수록 프롬프트 충실도 증가 |

**완료 후 (`GET /jobs/{job_id}`)**
```json
{
  "status": "done",
  "result_image": "..."
}
```

---

### `POST /segment` — SAM-2 포인트 세그멘테이션

정규화 좌표(0~1)로 클릭 포인트를 지정하면 해당 물체의 마스크를 반환합니다.
마우스패드를 클릭해 px/mm 스케일 계산에 사용합니다.

**Request**
```json
{
  "image_base64": "...",
  "point_x": 0.30,
  "point_y": 0.75,
  "label": 1
}
```

**완료 후 (`GET /jobs/{job_id}`)**
```json
{
  "status": "done",
  "mask_base64": "...",
  "overlay_base64": "..."
}
```

---

### `POST /generate` — 전체 파이프라인 단일 호출 ⚠️ 임시 스펙

> **Spring Boot 연동 형식이 아직 팀 내 합의되지 않았습니다.**
> 아래 스펙은 파이프라인 테스트를 위해 임시로 정한 것으로, 실제 연동 시 변경될 수 있습니다.

Step 1(물체 제거) → Step 2(제품 배치)를 한 번의 호출로 처리합니다.
Spring Boot는 이 엔드포인트만 호출하면 됩니다.

**Request**
```json
{
  "image_base64": "...",
  "style": "white",
  "products": [
    { "category": "KEYBOARD", "name": "로지텍 MX Keys Mini" },
    { "category": "MOUSE",    "name": "로지텍 MX Master 3" },
    { "category": "MONITOR",  "name": "LG 27인치 4K 모니터" }
  ]
}
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `style` | 필수 | 사용자 선택 스타일. `white` `black` `gaming` `cozy` `modern` `nordic` `retro` `industrial` |
| `products` | 필수 | Spring Boot가 추천한 제품 목록. `category` + `name` |
| `max_area_ratio` | `0.20` | 이 비율 초과 마스크는 책상/배경으로 간주해 제외 |

**지원 카테고리 (`category` 값)**
```
KEYBOARD / MOUSE / MONITOR / SPEAKER / DESK_LAMP / DESK_SHELF / LAPTOP_STAND / DECO / CLOCK
```

**완료 후 (`GET /jobs/{job_id}`)**
```json
{
  "status": "done",
  "result_image": "..."
}
```

> **미합의 사항 (팀 협의 필요)**
> - `category` 값이 Spring Boot DB 카테고리명과 일치하는지 확인 필요
> - `name`을 한국어로 받을지 영어로 받을지 결정 필요

---

## 호출 예시 (Python)

```python
import requests, time, base64
from pathlib import Path

BASE = "http://localhost:8000"

def poll(job_id):
    while True:
        res = requests.get(f"{BASE}/jobs/{job_id}").json()
        if res["status"] == "done":
            return res
        if res["status"] == "failed":
            raise RuntimeError(res.get("error"))
        time.sleep(5)

desk_b64 = base64.b64encode(Path("desk.jpg").read_bytes()).decode()

# Step 1: 물체 제거
r = requests.post(f"{BASE}/remove", json={
    "image_base64": desk_b64,
    "prompt": "keyboard. mouse. monitor. headset.",
}).json()
cleaned = poll(r["job_id"])["cleaned_image"]

# Step 2: 제품 배치 (제품 수만큼 반복)
current = cleaned
for prompt, mask_b64 in products:  # Spring Boot에서 받은 제품 목록
    r = requests.post(f"{BASE}/product_place", json={
        "image_base64": current,
        "mask_base64": mask_b64,
        "prompt": prompt,
    }).json()
    current = poll(r["job_id"])["result_image"]

# current = 최종 이미지 (base64)
```

---

## 프로젝트 구조

```
ai-server/
├── api/
│   ├── main.py                     # FastAPI 앱, 엔드포인트
│   ├── models.py                   # Pydantic 요청/응답 모델
│   ├── object_removal_processor.py # Step 1: DINO + SAM-2 + LaMa 오케스트레이터
│   ├── dino_processor.py           # Grounding DINO 텍스트 기반 물체 검출
│   ├── sam2_processor.py           # SAM-2 세그멘테이션
│   ├── lama_processor.py           # LaMa 물체 제거 inpainting
│   └── product_inpaint_processor.py # Step 2: SD Inpainting 제품 배치
├── configs/
│   └── config.yaml                 # 서버 설정
├── data/
│   └── test/                       # 테스트 데이터 (git 미포함, 직접 배치)
├── outputs/
│   └── test_results/               # 파이프라인 테스트 결과
├── logs/
│   └── server_errors.log           # 서버 에러 로그
├── requirements.txt
└── test_pipeline.py                # 전체 파이프라인 통합 테스트
```
