"""
전체 파이프라인 통합 테스트 스크립트
  Step 1: /remove   — 책상 위 물체 제거
  Step 2: /img2img  — 스타일 변환
  Step 3: /analyze  — 탑뷰 가용공간 분석

결과물: outputs/test_results/<timestamp>/
"""

import base64
import json
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = "http://localhost:8000"
OUT = Path("outputs/test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)

DESK_IMAGE     = Path("data/test/desk_image.jpg")
TOP_VIEW_IMAGE = Path("data/test/desk_top_image.jpg")

STYLE          = "white"
PROMPT         = "clean minimal white desk setup, white peripherals, soft lighting"
STRENGTH       = 0.7
DESK_WIDTH_CM  = 120.0
DESK_DEPTH_CM  = 60.0


def to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def save_b64(b64_str: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(b64_str))


def poll(job_id: str, label: str, interval: float = 5.0) -> dict:
    print(f"  [{label}] 처리 중", end="", flush=True)
    while True:
        res = requests.get(f"{BASE}/jobs/{job_id}").json()
        if res["status"] == "done":
            print(" 완료")
            return res
        if res["status"] == "failed":
            print(f" 실패: {res.get('error')}")
            raise RuntimeError(f"{label} 실패: {res.get('error')}")
        print(".", end="", flush=True)
        time.sleep(interval)


def step1_remove() -> str:
    print("\n[Step 1] /remove — 책상 위 물체 제거")
    t = time.time()

    r = requests.post(f"{BASE}/remove", json={
        "image_base64": to_b64(DESK_IMAGE),
        "prompt": "pen holder. small fan. rubik's cube. usb cable. book. notebook. headphone stand.",
    }).json()

    result = poll(r["job_id"], "remove")
    cleaned_b64 = result["cleaned_image"]

    save_b64(cleaned_b64,              OUT / "step1_cleaned.png")
    save_b64(result["detection_overlay"], OUT / "step1_overlay.png")
    save_b64(result["mask_image"],        OUT / "step1_mask.png")

    print(f"  감지 물체: {result['num_objects']}개  ({time.time()-t:.1f}s)")
    print(f"  저장: step1_cleaned.png / step1_overlay.png / step1_mask.png")
    return cleaned_b64


def step2_img2img(cleaned_b64: str) -> str:
    print("\n[Step 2] /img2img — 스타일 변환")
    t = time.time()

    r = requests.post(f"{BASE}/img2img", json={
        "image_base64": cleaned_b64,
        "prompt":        PROMPT,
        "style":         STYLE,
        "strength":      STRENGTH,
    }).json()

    result = poll(r["job_id"], "img2img")
    styled_b64 = result["result_image"]

    save_b64(styled_b64, OUT / "step2_styled.png")

    print(f"  style={STYLE}  strength={STRENGTH}  ({time.time()-t:.1f}s)")
    print(f"  저장: step2_styled.png")
    return styled_b64


def step3_analyze() -> None:
    print("\n[Step 3] /analyze — 탑뷰 가용공간 분석")
    t = time.time()

    r = requests.post(f"{BASE}/analyze", json={
        "image_base64": to_b64(TOP_VIEW_IMAGE),
        "desk_width_cm": DESK_WIDTH_CM,
        "desk_depth_cm": DESK_DEPTH_CM,
    }).json()

    result = poll(r["job_id"], "analyze")

    save_b64(result["available_mask"], OUT / "step3_available_mask.png")
    save_b64(result["occupied_mask"],  OUT / "step3_occupied_mask.png")

    metrics = result["metrics"]
    center  = result["placement_center"]

    summary = {
        "desk_area_cm2":      metrics["desk_area_cm2"],
        "occupied_area_cm2":  metrics["occupied_area_cm2"],
        "available_area_cm2": metrics["available_area_cm2"],
        "occupancy_ratio":    f"{metrics['occupancy_ratio']*100:.1f}%",
        "available_ratio":    f"{metrics['available_ratio']*100:.1f}%",
        "num_objects":        result["num_objects"],
        "placement_center_px": center,
        "top_regions_cm2":    metrics["connected_available_regions_cm2"][:3],
    }
    (OUT / "step3_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  감지 물체: {result['num_objects']}개  ({time.time()-t:.1f}s)")
    print(f"  책상 면적:  {metrics['desk_area_cm2']:.0f} cm²")
    print(f"  점유 면적:  {metrics['occupied_area_cm2']:.0f} cm²  ({metrics['occupancy_ratio']*100:.1f}%)")
    print(f"  가용 면적:  {metrics['available_area_cm2']:.0f} cm²  ({metrics['available_ratio']*100:.1f}%)")
    print(f"  배치 추천 좌표: {center}")
    print(f"  저장: step3_available_mask.png / step3_occupied_mask.png / step3_metrics.json")


def main():
    print(f"=== Deskterior 파이프라인 테스트 ===")
    print(f"결과 저장 위치: {OUT.resolve()}")

    if not DESK_IMAGE.exists():
        raise FileNotFoundError(f"이미지 없음: {DESK_IMAGE}")
    if not TOP_VIEW_IMAGE.exists():
        raise FileNotFoundError(f"이미지 없음: {TOP_VIEW_IMAGE}")

    r = requests.get(f"{BASE}/health")
    if r.status_code != 200:
        raise ConnectionError("서버 응답 없음. Docker 컨테이너 상태를 확인하세요.")
    print("서버 상태: OK\n")

    t_total = time.time()
    cleaned = step1_remove()
    step2_img2img(cleaned)
    step3_analyze()

    print(f"\n=== 완료 (총 {time.time()-t_total:.1f}s) ===")
    print(f"결과물 위치: {OUT.resolve()}")


if __name__ == "__main__":
    main()
