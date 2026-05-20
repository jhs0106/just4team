"""
전체 파이프라인 통합 테스트 스크립트
  Step 1: /remove        — 책상 위 물체 제거
  Step 2: /product_place — 마우스패드 기준 px/mm 계산 → 제품별 inpainting

결과물: outputs/test_results/<timestamp>/
"""

import base64
import csv
import random
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

BASE         = "http://100.90.189.37:8000"
OUT          = Path("outputs/test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)

DESK_IMAGE   = Path("data/test/desk_image2.jpg")
PRODUCTS_CSV = Path("data/test/products.csv")

STYLE = "white"

# ── 고정 테스트 제품 세트 ─────────────────────────────────────
# True: 아래 TEST_FIXED_PRODUCTS 사용 / False: CSV 랜덤 선택
USE_FIXED_PRODUCTS = True

# 직접 검증된 단품 이미지 ID (find_valid_products.py 실행 후 채울 것)
# None이면 해당 카테고리는 해당 phase에서 제외됨
TEST_FIXED_PRODUCTS: dict[str, int | None] = {
    "MONITOR":  233,   # ar=1.37 ✓  (화면 콘텐츠 있음 — 교체 권장)
    "KEYBOARD": None,  # TODO: find_valid_products.py --cat KEYBOARD 로 유효 ID 확인 후 입력
    "MOUSE":    560,   # ar=1.01 ✓  (031709 실행 확인됨)
}

# 단계별 카테고리 — None ID인 카테고리는 자동 제외
FIXED_PHASE_CATS: dict[int, list[str]] = {
    1: ["MONITOR"],
    2: ["MONITOR", "KEYBOARD"],
    3: ["MONITOR", "KEYBOARD", "MOUSE"],
}

# 마우스패드 클릭 포인트 (정규화 좌표) — 실제 마우스패드 위치에 맞게 조정
MOUSEPAD_POINT_X  = 0.30
MOUSEPAD_POINT_Y  = 0.75
MOUSEPAD_WIDTH_MM = 900   # XL 마우스패드 기준
MOUSEPAD_DEPTH_MM = 400

# 카테고리별 표준 치수 (width_mm, depth_mm)
CATEGORY_DIMS_MM = {
    "KEYBOARD":     (440, 130),
    "MOUSE":        (70,  120),
    "MONITOR":      (600, 50),
    "SPEAKER":      (90,  120),
    "DESK_LAMP":    (80,  80),
    "DESK_SHELF":   (600, 120),
    "LAPTOP_STAND": (280, 250),
    "DECO":         (80,  80),
    "CLOCK":        (100, 100),
}

# 카테고리별 inpainting 프롬프트
CATEGORY_PROMPT = {
    "KEYBOARD":     "mechanical keyboard on desk mat, top view",
    "MOUSE":        "wireless mouse on desk, top view",
    "MONITOR":      "monitor on desk, front view",
    "SPEAKER":      "desktop speaker on desk",
    "DESK_LAMP":    "modern desk lamp on desk",
    "DESK_SHELF":   "monitor riser shelf on desk",
    "LAPTOP_STAND": "laptop stand on desk",
    "DECO":         "small desk decoration",
    "CLOCK":        "minimalist desk clock",
}

# 카테고리별 수량 규칙 (min, max)
CATEGORY_COUNTS = {
    "KEYBOARD":     (1, 1),
    "MOUSE":        (1, 1),
    "MONITOR":      (1, 1),
    "DESK_LAMP":    (1, 1),
    "SPEAKER":      (0, 2),
    "DESK_SHELF":   (0, 1),
    "LAPTOP_STAND": (0, 1),
    "DECO":         (0, 1),
}

# generate 1/2/3 phase 제품 구성
_PHASE_COUNTS = {
    1: {"KEYBOARD": (1,1), "MOUSE": (1,1), "MONITOR": (1,1),
        "DESK_LAMP": (0,0), "SPEAKER": (0,0), "DESK_SHELF": (0,0), "LAPTOP_STAND": (0,0), "DECO": (0,0)},
    2: {"KEYBOARD": (1,1), "MOUSE": (1,1), "MONITOR": (1,1),
        "DESK_LAMP": (0,0), "SPEAKER": (0,1), "DESK_SHELF": (0,0), "LAPTOP_STAND": (0,0), "DECO": (0,0)},
    3: {"KEYBOARD": (1,1), "MOUSE": (1,1), "MONITOR": (1,1),
        "DESK_LAMP": (1,1), "SPEAKER": (0,1), "DESK_SHELF": (0,0), "LAPTOP_STAND": (0,0), "DECO": (0,0)},
}

STYLE_KEYWORDS = {
    "white": ["화이트", "white", "흰"],
    "black": ["블랙", "black", "검정", "다크"],
}


# ── 유틸 ──────────────────────────────────────────────────────

def to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_img(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def save_b64(b64_str: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(b64_str))


def poll(job_id: str, label: str, interval: float = 5.0) -> dict:
    print(f"  [{label}] 처리 중", end="", flush=True)
    while True:
        res = requests.get(f"{BASE}/jobs/{job_id}")
        if res.status_code != 200:
            raise RuntimeError(f"{label} polling 실패: HTTP {res.status_code}")
        data = res.json()
        if data["status"] == "failed":
            print(f"\n  실패: {data.get('error')}")
            raise RuntimeError(f"{label} 실패: {data.get('error')}")
        if data["status"] == "done":
            result_key = next((k for k in data if k not in ("job_id", "status", "error") and data[k] is not None), None)
            if result_key is None:
                raise RuntimeError(f"{label}: status=done 이지만 결과값이 None. logs/server_errors.log 확인")
            print(" 완료")
            return data
        print(".", end="", flush=True)
        time.sleep(interval)


# ── 제품 선택 ──────────────────────────────────────────────────

def parse_metadata(meta_str: str) -> dict:
    # CSV metadata 컬럼 파싱 (Python dict 리터럴 형식).
    if not meta_str or not meta_str.strip():
        return {}
    try:
        import ast
        return ast.literal_eval(meta_str)
    except Exception:
        return {}


def load_products_by_style(style: str) -> dict[str, list[dict]]:
    keywords = STYLE_KEYWORDS.get(style, [])
    by_cat: dict[str, list[dict]] = {}
    with open(PRODUCTS_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat in ("DESK", "LIGHTING"):  # LIGHTING은 배치 미지원
                continue
            if any(kw in row["title"] for kw in keywords):
                by_cat.setdefault(cat, []).append(row)
    return by_cat


def select_desk(style: str) -> dict | None:
    # 스타일에 맞는 책상 한 개 선택 (치수 확인 가능한 것 우선)
    keywords = STYLE_KEYWORDS.get(style, [])
    candidates = []
    with open(PRODUCTS_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["category"] != "DESK":
                continue
            if any(kw in row["title"] for kw in keywords):
                m = parse_metadata(row.get("metadata", ""))
                if m.get("width_mm"):
                    candidates.append(row)
    return random.choice(candidates) if candidates else None


def select_products(style: str, counts: dict | None = None) -> list[dict]:
    by_cat = load_products_by_style(style)
    selected = []
    for cat, (mn, mx) in (counts or CATEGORY_COUNTS).items():
        pool  = by_cat.get(cat, [])
        count = random.randint(mn, mx)
        if count == 0 or not pool:
            continue
        selected.extend(random.sample(pool, min(count, len(pool))))
    return selected


# 마스크 계산 

def get_mask_bounds(mask_b64: str) -> tuple[int, int, int, int]:
    mask_np = np.array(b64_to_img(mask_b64).convert("L"))
    ys, xs  = np.where(mask_np > 128)
    if len(xs) == 0:
        raise ValueError("마스크 영역 없음")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def make_rect_mask_b64(img_w: int, img_h: int,
                       x1: int, y1: int, x2: int, y2: int) -> str:
    mask = Image.new("L", (img_w, img_h), 0)
    ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=255)
    return img_to_b64(mask)


def calc_product_regions(
    pad_x1: int, pad_y1: int, pad_x2: int, pad_y2: int,
    img_w: int, img_h: int,
    products: list[dict],
) -> list[dict]:
    pad_px_w = pad_x2 - pad_x1
    pad_px_h = pad_y2 - pad_y1
    px_x     = pad_px_w / MOUSEPAD_WIDTH_MM
    px_y     = pad_px_h / MOUSEPAD_DEPTH_MM
    cx       = (pad_x1 + pad_x2) // 2
    mg       = 15

    def clip(x1, y1, x2, y2):
        return (max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2))

    regions = []
    for p in products:
        cat  = p["category"]
        dims = CATEGORY_DIMS_MM.get(cat)
        if not dims:
            continue
        w_mm, d_mm = dims
        pw = int(w_mm * px_x)
        ph = int(d_mm * px_y)

        if cat == "KEYBOARD":
            x1 = cx - pw // 2
            x2 = x1 + pw
            y2 = pad_y2
            y1 = y2 - ph
        elif cat == "MOUSE":
            x1 = pad_x2 + mg
            x2 = x1 + pw
            y2 = pad_y2
            y1 = y2 - ph
        elif cat == "MONITOR":
            x1 = cx - pw // 2
            x2 = x1 + pw
            y2 = pad_y1 - mg
            y1 = y2 - int(300 * px_y)
        elif cat == "SPEAKER":
            x2 = pad_x1 - mg
            x1 = x2 - pw
            y2 = pad_y1 - mg
            y1 = y2 - ph
        elif cat == "DESK_LAMP":
            x1 = pad_x2 + mg * 3
            x2 = x1 + pw
            y2 = pad_y1 - mg
            y1 = y2 - int(300 * px_y)
        elif cat == "DESK_SHELF":
            x1 = cx - pw // 2
            x2 = x1 + pw
            y2 = pad_y1 - mg
            y1 = y2 - ph
        elif cat == "LAPTOP_STAND":
            x2 = pad_x1 - mg
            x1 = x2 - pw
            y1 = pad_y1
            y2 = y1 + ph
        else:
            x1 = pad_x1
            x2 = x1 + pw
            y1 = pad_y1
            y2 = y1 + ph

        x1, y1, x2, y2 = clip(x1, y1, x2, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        regions.append({"product": p, "region": (x1, y1, x2, y2)})

    return regions


# 파이프라인 스텝

def step1_remove() -> str:
    print("\n[Step 1] /remove — 책상 위 물체 제거")
    t = time.time()

    r = requests.post(f"{BASE}/remove", json={
        "image_base64": to_b64(DESK_IMAGE),
        "prompt": "keyboard. mouse. monitor. headset. pen holder. cable. book. notebook.",
    }).json()

    result = poll(r["job_id"], "remove")
    save_b64(result["cleaned_image"],     OUT / "step1_cleaned.png")
    save_b64(result["detection_overlay"], OUT / "step1_overlay.png")
    save_b64(result["mask_image"],        OUT / "step1_mask.png")

    print(f"  감지 물체: {result['num_objects']}개  ({time.time()-t:.1f}s)")
    return result["cleaned_image"]


def step2_place_products(cleaned_b64: str) -> str:
    print("\n[Step 2] 제품 배치 — 마우스패드 기준 px/mm → inpainting")

    img = b64_to_img(cleaned_b64)
    img_w, img_h = img.size

    # 마우스패드 세그먼트
    r = requests.post(f"{BASE}/segment", json={
        "image_base64": cleaned_b64,
        "point_x":      MOUSEPAD_POINT_X,
        "point_y":      MOUSEPAD_POINT_Y,
        "label":        1,
    }).json()
    seg = poll(r["job_id"], "segment/mousepad")
    pad_x1, pad_y1, pad_x2, pad_y2 = get_mask_bounds(seg["mask_base64"])
    print(f"  마우스패드 바운딩 박스: ({pad_x1},{pad_y1}) ~ ({pad_x2},{pad_y2})")

    # 제품 선택 및 영역 계산
    products = select_products(STYLE)
    regions  = calc_product_regions(
        pad_x1, pad_y1, pad_x2, pad_y2, img_w, img_h, products
    )

    print(f"  배치할 제품 {len(regions)}개:")
    for item in regions:
        p = item["product"]
        r_ = item["region"]
        print(f"    [{p['category']}] {p['title'][:30]}  → 픽셀 {r_}")

    # 제품별 순서대로 inpainting
    current_b64 = cleaned_b64
    for i, item in enumerate(regions):
        p      = item["product"]
        cat    = p["category"]
        x1, y1, x2, y2 = item["region"]
        style_prefix = STYLE + " "
        prompt = style_prefix + CATEGORY_PROMPT.get(cat, cat.lower())
        mask_b64 = make_rect_mask_b64(img_w, img_h, x1, y1, x2, y2)

        resp = requests.post(f"{BASE}/product_place", json={
            "image_base64": current_b64,
            "mask_base64":  mask_b64,
            "prompt":       prompt,
        })
        if resp.status_code != 200:
            print(f"\n  /product_place 요청 실패: HTTP {resp.status_code}")
            print(f"  응답: {resp.text[:300]}")
            break
        r = resp.json()
        result = poll(r["job_id"], f"place/{cat}")
        current_b64 = result["result_image"]

        save_b64(current_b64, OUT / f"step2_{i+1:02d}_{cat}.png")

    save_b64(current_b64, OUT / "final.png")
    print(f"  최종 결과: {OUT / 'final.png'}")
    return current_b64


def main():
    print(f"=== Deskterior 파이프라인 테스트 ===")
    print(f"결과 저장 위치: {OUT.resolve()}")

    if not DESK_IMAGE.exists():
        raise FileNotFoundError(f"이미지 없음: {DESK_IMAGE}")

    r = requests.get(f"{BASE}/health")
    if r.status_code != 200:
        raise ConnectionError("서버 응답 없음.")
    print("서버 상태: OK\n")

    t_total = time.time()
    cleaned = step1_remove()
    step2_place_products(cleaned)

    print(f"\n=== 완료 (총 {time.time()-t_total:.1f}s) ===")
    print(f"결과물 위치: {OUT.resolve()}")


def test_generate(phase: int | None = None):
    # POST /generate 단일 호출 테스트
    counts = _PHASE_COUNTS.get(phase) if phase else None
    print(f"\n=== /generate 엔드포인트 테스트 {'(phase ' + str(phase) + ')' if phase else ''} ===")
    print(f"결과 저장 위치: {OUT.resolve()}")

    if not DESK_IMAGE.exists():
        raise FileNotFoundError(f"이미지 없음: {DESK_IMAGE}")

    r = requests.get(f"{BASE}/health")
    if r.status_code != 200:
        raise ConnectionError("서버 응답 없음.")
    print("서버 상태: OK\n")

    # 책상 선택 (치수 파악)
    desk = select_desk(STYLE)
    desk_width_mm = None
    if desk:
        m = parse_metadata(desk.get("metadata", ""))
        desk_width_mm = m.get("width_mm")
        print(f"책상: {desk['title'][:40]}  ({desk_width_mm}mm)")
    else:
        print("책상: 스타일 매칭 없음 — 치수 미지정")

    # ── 제품 선택: 고정 세트 or CSV 랜덤 ──────────────────────────
    if USE_FIXED_PRODUCTS:
        phase_cats = FIXED_PHASE_CATS.get(phase, list(TEST_FIXED_PRODUCTS.keys()))
        products_payload = []
        for cat in phase_cats:
            img_id = TEST_FIXED_PRODUCTS.get(cat)
            if img_id is None:
                print(f"  [{cat}] image_id=None — 제외 (TEST_FIXED_PRODUCTS 미설정)")
                continue
            products_payload.append({"category": cat, "name": f"{cat} fixed-test", "image_id": img_id})
        print(f"\n[고정 테스트 제품] phase={phase}  {len(products_payload)}개:")
        for p in products_payload:
            print(f"  [{p['category']}] image_id={p['image_id']}")
    else:
        csv_products = select_products(STYLE, counts)
        print(f"\n[랜덤 선택 제품] {len(csv_products)}개:")
        for p in csv_products:
            m = parse_metadata(p.get("metadata", ""))
            dims = f"  {m['width_mm']}x{m['depth_mm']}mm" if m.get("width_mm") else ""
            print(f"  [{p['category']}] {p['title'][:35]}{dims}")

        def _csv_payload(p: dict) -> dict:
            m = parse_metadata(p.get("metadata", ""))
            item = {"category": p["category"], "name": p["title"], "image_id": int(p["id"])}
            if m.get("width_mm"): item["width_mm"] = m["width_mm"]
            if m.get("depth_mm"): item["depth_mm"] = m["depth_mm"]
            return item
        products_payload = [_csv_payload(p) for p in csv_products]

    t = time.time()
    TOP_VIEW = Path("data/test/desk_top_image2.jpg")
    payload = {
        "image_base64":        to_b64(DESK_IMAGE),
        "style":               STYLE,
        "products":            products_payload,
        "fixed_test_products": USE_FIXED_PRODUCTS,
    }
    if desk_width_mm:
        payload["desk_width_mm"] = desk_width_mm
    if TOP_VIEW.exists():
        payload["top_view_image_base64"] = to_b64(TOP_VIEW)
        print("\n탑뷰 이미지 포함 — Homography 기반 배치 사용")
    else:
        print("\n탑뷰 이미지 없음 — 비율 기반 배치 사용")

    resp = requests.post(f"{BASE}/generate", json=payload)

    if resp.status_code != 200:
        print(f"요청 실패: HTTP {resp.status_code}")
        print(resp.text[:300])
        return

    result = poll(resp.json()["job_id"], "generate")

    if result.get("cleaned_image"):
        save_b64(result["cleaned_image"],    OUT / "step1_cleaned.png")
    if result.get("composited_image"):
        save_b64(result["composited_image"], OUT / "step2_composited.png")
    if result.get("result_image"):
        save_b64(result["result_image"],     OUT / "step3_final.png")

    print(f"\n제거 물체: {result['num_removed']}개")
    print(f"배치 제품: {result['num_placed']}개")
    print(f"소요 시간: {time.time()-t:.1f}s")
    print(f"\n[결과 파일 — test_results]")
    print(f"  {OUT}/step1_cleaned.png    — LaMa 물체 제거 후")
    print(f"  {OUT}/step3_final.png      — 최종 결과")
    print(f"\n[디버그 파일 — outputs/debug/<timestamp>/]")
    print(f"  cleaned_front.png          — 제거 후 이미지")
    print(f"  placement_only_result.png  — bbox 배치 확인")
    print(f"  cv_composite_result.png    — CV 합성 확인")
    print(f"  placement_debug.png        — bbox 좌표 시각화")
    print(f"  products_list.json         — 사용된 제품/좌표 목록")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "remove":
            step1_remove()
        elif cmd == "place":
            step2_place_products(to_b64(DESK_IMAGE))
        elif cmd == "generate":
            phase = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
            test_generate(phase)
        else:
            print("사용법: python test_pipeline.py [remove|place|generate [1|2|3]]")
    else:
        main()
