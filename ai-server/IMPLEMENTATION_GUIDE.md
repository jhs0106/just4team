# 구현 인수인계 문서

## 현재 상태 요약

| 단계 | 엔드포인트 | 상태 | 비고 |
|---|---|---|---|
| Step 1: 물체 제거 | `POST /remove` | **완료** | SAM-2 + LaMa |
| Step 2: 스타일 변환 | `POST /img2img` | **완료** | SD img2img + LoRA, strength=0.7 |
| Step 3: 상품 배치 | `POST /place` | **완료** | cv2.seamlessClone, 외형 보존, GPU 불필요 |
| (레거시) | `POST /inpaint` | 유지 | IP-Adapter-Plus, 외형 변형 있음 — 비교용 |

---

## 완료된 것 (건드리지 않아도 됨)

### Step 1 — `/remove`
- SAM-2 `AutomaticMaskGenerator`로 책상 위 물체 자동 감지
- `max_area_ratio=0.20`: 이미지 20% 초과 마스크 제외 (책상/배경 보호)
- `y_min_ratio=0.25`: 이미지 상단 25% 내 물체 제외 (모니터/벽)
- LaMa로 감지된 물체 순차 제거 → 빈 책상 반환

### Step 2 — `/img2img`
- `StableDiffusionImg2ImgPipeline` + LoRA (`JU_DeskStyle` 트리거워드)
- `strength=0.7`: 원본 구조 보존 + 스타일 변환 균형점 (테스트로 확정)
- IP-Adapter 미사용: img2img latent에 원본 정보가 이미 포함되어 있어 추가 시 신호 충돌
- `DPMSolverMultistepScheduler` (Karras sigmas): 부분 디노이징 궤적에 적합

---

## 완료된 것 (Step 3)

### Step 3 — `/place`

**구현**: `api/place_processor.py` — `cv2.seamlessClone(MIXED_CLONE)`

**동작 방식**:
1. 마스크에서 배치 영역 bounding box 계산
2. 상품 이미지를 bounding box 크기로 리사이즈
3. `cv2.seamlessClone`으로 경계 조명/색상만 자연스럽게 보정, 상품 픽셀은 그대로 유지

**핵심 제약**:
- GPU 불필요 — 순수 CPU OpenCV, `to("cuda")` / `to("cpu")` 패턴 없음
- `center` 좌표가 배경 경계를 넘으면 OpenCV 에러 → place_processor.py에서 clamp 처리함
- `/inpaint` (IP-Adapter-Plus)는 레거시로 유지 — 비교 목적

---

## 코드 작성 규칙

```
모든 GPU 작업     run_in_executor로 분리 (main.py 패턴 참고)
모델 로드         싱글톤 (get_xxx_processor 함수)
GPU 해제          pipe.to("cpu") + torch.cuda.empty_cache()
LoRA 로드         PeftModel.from_pretrained(pipe.unet, lora_path)  ← pipe.load_lora_weights() 금지
IP-Adapter        attention_slicing과 동시 사용 금지
이미지 해상도     512×512 고정
이미지 형식       base64 PNG
```

---

## Docker 실행

```bash
docker compose up --build -d
curl http://localhost:8000/health

# LoRA 없이 실행될 경우 (정상이지만 스타일 약함)
# outputs/models/lora_final/ 폴더 수동 배치 필요
```

첫 실행 시 HuggingFace 모델 자동 다운로드 (SAM-2, SD, IP-Adapter 등, 10GB+).
