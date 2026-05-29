"""
전체 파이프라인 테스트 스크립트
  Step 1  /remove   DINO + SAM-2로 기존 물체 제거 → 빈 책상
  Step 2  /img2img  스타일 적용
  Step 3a /segment  SAM-2 포인트 세그멘테이션으로 배치 구역 마스크 생성
  Step 3b /inpaint  IP-Adapter로 상품 합성
  - 첫 번째 상품: ip_adapter_scale 4단계 비교 (0.5 / 0.7 / 0.9 / 1.0)
  - 추가 샘플 상품: scale=0.8 기본값

실행 위치: ai-server/ 디렉터리
  python test_place.py --ids 330                특정 상품 ID
  python test_place.py --ids 330 522            여러 ID
  python test_place.py --style modern           스타일 지정 (기본: white)
  python test_place.py --skip-remove            Step 1 건너뜀 (이미 빈 책상이면)
  python test_place.py --skip-img2img           Step 2 건너뜀 (스타일 변환 생략)
  python test_place.py --point 0.30 0.75        SAM-2 클릭 포인트 (기본: 마우스패드 중앙)
  python test_place.py --mousepad-size 900 400  마우스패드 실제 크기 mm (기본: 900×400)

필요 파일:
  data/test/desk_image.jpg        배경 책상 사진
  data/test/products.csv          상품 목록
  data/test/processed_images/     {id}.png 상품 이미지 폴더
"""

import argparse
import ast
import base64
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from io import BytesIO
from PIL import Image

BASE = "http://localhost:8000"
OUT  = Path("outputs/test_pipeline") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)

BG_IMAGE = Path("data/test/desk_image.jpg")
CSV_PATH = Path("data/test/products.csv")
IMG_DIR  = Path("data/test/processed_images")

DEFAULT_PROMPT = "clean desk, natural lighting, photorealistic"
DEFAULT_STYLE  = "white"

# 키보드 배열별 표준 크기 (width_mm, depth_mm)
KEYBOARD_SIZES = [
    # (pattern, width_mm, depth_mm, height_mm)
    (r"10[48]키?|풀\s*배열|full.?size",        440, 130, 35),
    (r"87키?|88키?|tkl|텐키\s*리스",            360, 130, 35),
    (r"8[2-5]키?|75\s*%",                       320, 130, 35),
    (r"6[6-8]키?|65\s*%",                       310, 100, 30),
    (r"61키?|60\s*%",                           290, 100, 30),
]


# 유틸리티
def to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def save_b64(b64_str: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(b64_str))

def poll(job_id: str, label: str, interval: float = 2.0) -> dict:
    print(f"  [{label}] 처리 중", end="", flush=True)
    while True:
        resp = requests.get(f"{BASE}/jobs/{job_id}")
        if resp.status_code != 200:
            print(f"\n  [{label}] HTTP {resp.status_code}: {resp.text}")
            raise RuntimeError(f"{label} 폴링 실패 — 서버 로그 확인 필요")
        res = resp.json()
        if "status" not in res:
            print(f"\n  [{label}] 응답 이상: {res}")
            raise RuntimeError(f"{label} 응답에 status 없음")
        if res["status"] == "done":
            print(" 완료")
            return res
        if res["status"] == "failed":
            print(f" 실패: {res.get('error')}")
            raise RuntimeError(f"{label} 실패: {res.get('error')}")
        print(".", end="", flush=True)
        time.sleep(interval)

def infer_keyboard_size(title: str, category: str = "") -> tuple[float, float, float] | None:
    """타이틀/카테고리에서 키보드 배열 감지 → (width_mm, depth_mm, height_mm) 반환.
    키보드로 판단되지 않으면 None 반환."""
    text = (title + " " + category).lower()
    if "키보드" not in text and "keyboard" not in text and "키캡" not in text:
        return None
    for pattern, w, d, h in KEYBOARD_SIZES:
        if re.search(pattern, text, re.IGNORECASE):
            return float(w), float(d), float(h)
    # 키보드로 판단되지만 배열 불명 → TKL 기본값
    return 360.0, 130.0, 35.0

def make_product_mask(segment_mask_b64: str,
                      width_mm: float | None,
                      depth_mm: float | None,
                      height_mm: float | None,
                      mousepad_width_mm: float,
                      mousepad_depth_mm: float) -> tuple[str, str]:
    """
    SAM-2 마스크 안에 제품 실제 크기 기반 사각형 마스크 생성.
    - x축 스케일: 마우스패드 너비 픽셀 / 실제 mm (원근 영향 없음)
    - y축 스케일: 마우스패드 높이 픽셀 / 실제 mm (원근 압축 자동 반영)
    - 마스크 = 탑면(depth×px_per_mm_y) + 앞면(height×px_per_mm_x)
    - 마우스패드 앞 테두리에 앵커링 → 키보드가 앞에 놓인 것처럼 배치
    Returns: (product_mask_b64, size_label)
    """
    mask_img = Image.open(BytesIO(base64.b64decode(segment_mask_b64))).convert("L")
    arr = np.array(mask_img)
    ys, xs = np.where(arr > 128)
    if len(xs) == 0 or width_mm is None or depth_mm is None:
        return segment_mask_b64, "전체 마스크 (크기 미지정)"
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad_px_w = x2 - x1
    pad_px_h = y2 - y1
    px_per_mm_x = pad_px_w / mousepad_width_mm
    px_per_mm_y = pad_px_h / mousepad_depth_mm
    prod_px_w = min(int(width_mm * px_per_mm_x), pad_px_w)
    # 탑면: 원근 압축된 깊이 / 앞면: 수직 높이 (수평 스케일 기준)
    top_h  = int(depth_mm  * px_per_mm_y)
    face_h = int((height_mm or 0) * px_per_mm_x)
    prod_px_h = min(top_h + face_h, pad_px_h)
    # 마우스패드 앞 테두리(y2)에 앵커링, 가로 중앙 배치
    cx  = (x1 + x2) // 2
    ry2 = y2
    ry1 = max(y1, ry2 - prod_px_h)
    rx1 = max(x1, cx - prod_px_w // 2)
    rx2 = min(x2, rx1 + prod_px_w)
    new_arr = np.zeros_like(arr)
    new_arr[ry1:ry2, rx1:rx2] = 255
    buf = BytesIO()
    Image.fromarray(new_arr, mode="L").save(buf, format="PNG")
    h_str = f"+{int(height_mm)}mm높이" if height_mm else ""
    label = (f"{int(width_mm)}×{int(depth_mm)}{h_str}mm "
             f"→ {prod_px_w}×{prod_px_h}px (탑{top_h}+앞{face_h})")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), label

def load_products(csv_path: Path, img_dir: Path, limit: int,
                  ids: list[str] | None = None) -> list[dict]:
    products = []
    id_set = set(ids) if ids else None
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if id_set and row["id"] not in id_set:
                continue
            img_path = img_dir / f"{row['id']}.png"
            if not img_path.exists():
                continue
            try:
                meta = ast.literal_eval(row.get("metadata", "{}") or "{}")
            except Exception:
                meta = {}
            width_mm  = meta.get("width_mm")
            depth_mm  = meta.get("depth_mm")
            height_mm = meta.get("height_mm")
            title     = row["title"]
            category  = row.get("category", "")
            # 메타데이터에 크기 없으면 타이틀로 추론
            if not width_mm or not depth_mm:
                inferred = infer_keyboard_size(title, category)
                if inferred:
                    width_mm, depth_mm, height_mm = inferred
            products.append({
                "id":        row["id"],
                "title":     title,
                "category":  category,
                "width_mm":  width_mm,
                "depth_mm":  depth_mm,
                "height_mm": height_mm,
                "img_path":  img_path,
            })
            if id_set is None and len(products) >= limit:
                break
    return products

def print_product_info(p: dict) -> None:
    size_str = ""
    if p["width_mm"] and p["depth_mm"]:
        h = f"×{p['height_mm']:.0f}h" if p.get("height_mm") else ""
        size_str = f"  ({p['width_mm']:.0f}×{p['depth_mm']:.0f}{h}mm)"
    print(f"  [#{p['id']}] {p['title']}{size_str}")

# Step 1: 물체 제거

def step1_remove(bg_b64: str, prompt: str | None = None) -> str:
    if prompt:
        print(f"[Step 1] 물체 제거 (DINO 프롬프트: {prompt})")
    else:
        print("[Step 1] 물체 제거 (SAM-2 Auto)")
    payload = {"image_base64": bg_b64}
    if prompt:
        payload["prompt"] = prompt
    r = requests.post(f"{BASE}/remove", json=payload).json()
    result = poll(r["job_id"], "remove")
    cleaned = result["cleaned_image"]
    save_b64(cleaned, OUT / "step1_cleaned.png")
    if result.get("detection_overlay"):
        save_b64(result["detection_overlay"], OUT / "step1_overlay.png")
    print(f"  감지 물체: {result.get('num_objects', 0)}개\n")
    return cleaned


# ── Step 2: 스타일 변환 ──────────────────────────────────────────

def step2_img2img(img_b64: str, prompt: str, style: str) -> str:
    print(f"[Step 2] 스타일 변환 (style={style})")
    payload = {
        "image_base64": img_b64,
        "prompt": prompt,
        "style": style,
        "strength": 0.35,
    }
    r = requests.post(f"{BASE}/img2img", json=payload).json()
    result = poll(r["job_id"], "img2img")
    styled = result["result_image"]
    save_b64(styled, OUT / "step2_styled.png")
    print()
    return styled


# ── Step 3a: SAM-2 포인트 세그멘테이션 ──────────────────────────

def step3a_segment(img_b64: str, point_x: float, point_y: float) -> str:
    print(f"[Step 3a] 배치 구역 세그멘테이션 (포인트: x={point_x:.2f}, y={point_y:.2f})")
    payload = {
        "image_base64": img_b64,
        "point_x": point_x,
        "point_y": point_y,
    }
    r = requests.post(f"{BASE}/segment", json=payload).json()
    result = poll(r["job_id"], "segment")
    mask = result["mask_base64"]
    save_b64(mask, OUT / "step3a_segment_mask.png")
    if result.get("overlay_base64"):
        save_b64(result["overlay_base64"], OUT / "step3a_segment_overlay.png")
    print()
    return mask


# ── Step 3b: 상품 인페인팅 (/inpaint) ────────────────────────────

def step3b_inpaint(img_b64: str, mask_b64: str, product_b64: str,
                   prompt: str, style: str, ip_adapter_scale: float,
                   label: str) -> str:
    payload = {
        "image_base64":         img_b64,
        "mask_base64":          mask_b64,
        "product_image_base64": product_b64,
        "prompt":               prompt,
        "style":                style,
        "ip_adapter_scale":     ip_adapter_scale,
    }
    r = requests.post(f"{BASE}/inpaint", json=payload).json()
    result = poll(r["job_id"], label)
    return result["result_image"]


# ── 메인 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--style", default=DEFAULT_STYLE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--remove-prompt", default=None,
                        help="Step 1 DINO 프롬프트 (예: \"keyboard. mouse. fan.\")")
    parser.add_argument("--skip-remove", action="store_true")
    parser.add_argument("--skip-img2img", action="store_true")
    parser.add_argument("--point", nargs=2, type=float, default=[0.30, 0.75],
                        metavar=("X", "Y"),
                        help="SAM-2 클릭 포인트 (기본: 0.30 0.75 = 마우스패드 중앙)")
    parser.add_argument("--mousepad-size", nargs=2, type=float, default=[900.0, 400.0],
                        metavar=("WIDTH_MM", "DEPTH_MM"),
                        help="마우스패드 실제 크기 mm (기본: 900 400)")
    args = parser.parse_args()

    mousepad_w_mm, mousepad_d_mm = args.mousepad_size

    print("=== 전체 파이프라인 테스트 ===")
    print(f"스타일: {args.style} | 마우스패드: {mousepad_w_mm:.0f}×{mousepad_d_mm:.0f}mm")
    print(f"결과 위치: {OUT.resolve()}\n")

    for p in [BG_IMAGE, CSV_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"파일 없음: {p}")

    if requests.get(f"{BASE}/health").status_code != 200:
        raise ConnectionError("서버 응답 없음 — uvicorn api.main:app 실행 확인")
    print("서버 상태: OK\n")

    products = load_products(CSV_PATH, IMG_DIR, limit=args.sample + 1, ids=args.ids)
    if not products:
        raise RuntimeError("사용 가능한 상품 없음")
    print(f"테스트 상품: {len(products)}개")
    for p in products:
        print_product_info(p)
    print()

    (OUT / "00_original.jpg").write_bytes(BG_IMAGE.read_bytes())
    bg_b64 = to_b64(BG_IMAGE)

    # Step 1
    if args.skip_remove:
        print("[Step 1] 건너뜀 — 원본 이미지 사용\n")
        cleaned_b64 = bg_b64
    else:
        cleaned_b64 = step1_remove(bg_b64, prompt=args.remove_prompt)

    # Step 2
    if args.skip_img2img:
        print("[Step 2] 건너뜀\n")
        styled_b64 = cleaned_b64
    else:
        styled_b64 = step2_img2img(cleaned_b64, args.prompt, args.style)

    # Step 3a — SAM-2로 마우스패드 세그멘테이션 (한 번만 실행, 모든 상품 공유)
    point_x, point_y = args.point
    zone_mask_b64 = step3a_segment(cleaned_b64, point_x, point_y)

    # Step 3b — 첫 번째 상품: scale 4단계 비교
    p0 = products[0]
    p0_mask_b64, size_label = make_product_mask(
        zone_mask_b64, p0["width_mm"], p0["depth_mm"], p0.get("height_mm"),
        mousepad_w_mm, mousepad_d_mm
    )
    save_b64(p0_mask_b64, OUT / f"step3a_product_mask_{p0['id']}.png")

    h_mm = p0.get("height_mm") or 35
    inpaint_prompt = (
        f"{args.prompt}, {h_mm:.0f}mm height profile visible from front angle"
    )
    print(f"[Step 3b] 상품 합성 — #{p0['id']} {p0['title']}")
    print(f"  마스크: {size_label}")
    print(f"  프롬프트: {inpaint_prompt}")
    print("  ip_adapter_scale 4단계 비교\n")
    prod_b64 = to_b64(p0["img_path"])
    (OUT / f"product_{p0['id']}.png").write_bytes(p0["img_path"].read_bytes())

    scale_tests = [("0.5", 0.5), ("0.7", 0.7), ("0.9", 0.9), ("1.0", 1.0)]
    for label, scale in scale_tests:
        t = time.time()
        print(f"  ip_adapter_scale = {label}")
        result_b64 = step3b_inpaint(styled_b64, p0_mask_b64, prod_b64,
                                    inpaint_prompt, args.style, scale,
                                    f"#{p0['id']} scale={label}")
        out_name = f"result_p{p0['id']}_scale_{label}.png"
        save_b64(result_b64, OUT / out_name)
        print(f"    저장: {out_name}  ({time.time()-t:.1f}s)")
    print()

    # Step 3b — 추가 상품: scale=0.8
    if len(products) > 1:
        print("--- 추가 상품 (ip_adapter_scale=0.8) ---")
        for p in products[1:]:
            print_product_info(p)
            p_mask_b64, size_label = make_product_mask(
                zone_mask_b64, p["width_mm"], p["depth_mm"], p.get("height_mm"),
                mousepad_w_mm, mousepad_d_mm
            )
            save_b64(p_mask_b64, OUT / f"step3a_product_mask_{p['id']}.png")
            p_h_mm = p.get("height_mm") or 35
            p_prompt = f"{args.prompt}, {p_h_mm:.0f}mm height profile visible from front angle"
            print(f"  마스크: {size_label}")
            t = time.time()
            result_b64 = step3b_inpaint(styled_b64, p_mask_b64,
                                        to_b64(p["img_path"]),
                                        p_prompt, args.style, 0.8,
                                        f"#{p['id']}")
            out_name = f"result_p{p['id']}_scale_0.8.png"
            save_b64(result_b64, OUT / out_name)
            print(f"  저장: {out_name}  ({time.time()-t:.1f}s)\n")

    print("=== 완료 ===")
    print(f"결과물 위치: {OUT.resolve()}")


if __name__ == "__main__":
    main()
