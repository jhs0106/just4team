# 여러 스타일(white/black/gaming)을 한 timestamp 폴더 안에 일괄 테스트.
# 출력 구조:
#   outputs/test_results/<timestamp>/
#     ├── white/  step1_cleaned.png, step3_final.png, meta.json
#     ├── black/  ...
#     └── gaming/ ...
#
# 사용 예:
#   python scripts/batch_test.py
#   python scripts/batch_test.py --desk front_view3 --top top_view3
#   python scripts/batch_test.py --styles white,black --budget 500000

import argparse
import base64
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_SERVER = "http://localhost:8000"

# styled_test.py의 상수·헬퍼 재사용
sys.path.insert(0, str(REPO_ROOT / "ai-server"))
from scripts.styled_test import (
    PRODUCTS_CSV, STYLE_KEYWORDS, PRODUCT_BLACKLIST_IDS,
    PREFERRED_IDS, MOCK_PRICES_KRW, DEFAULT_SIZES_MM,
    to_b64, load_products_by_style, select_setup, poll,
)


_DEFAULT_BUDGETS = {"white": 500000, "black": 600000, "gaming": 700000}


def resolve_desk_paths(desk_arg: str | None, top_arg: str | None) -> tuple[Path, Path | None]:
    # --desk 인자가 "front_view3" 같은 이름이면 ai-server/data/test/<이름>.png/.jpg 자동 매칭
    data_dir = REPO_ROOT / "ai-server" / "data" / "test"

    if desk_arg:
        for ext in (".png", ".jpg", ".jpeg"):
            cand = data_dir / f"{desk_arg}{ext}"
            if cand.exists():
                front_path = cand
                break
        else:
            raise FileNotFoundError(f"front view 이미지 없음: {desk_arg}")
    else:
        front_path = data_dir / "desk_image2.jpg"

    top_path: Path | None = None
    if top_arg:
        for ext in (".png", ".jpg", ".jpeg"):
            cand = data_dir / f"{top_arg}{ext}"
            if cand.exists():
                top_path = cand
                break
    else:
        # front 이름에서 자동 매칭 시도 (front_view3 → top_view3 등)
        _auto = data_dir / front_path.name.replace("front_view", "top_view").replace("desk_image", "desk_top_image")
        if _auto.exists() and _auto != front_path:
            top_path = _auto

    return front_path, top_path


def run_one_style(style: str, budget: int, cats: list[str],
                  desk_path: Path, top_path: Path | None,
                  desk_width_mm: int, desk_depth_mm: int,
                  out_dir: Path, seed: int | None) -> dict:
    if seed is not None:
        random.seed(seed)

    print(f"\n=== {style.upper()} (budget={budget:,}원) ===")
    selected, total = select_setup(style, budget, cats)
    if not selected:
        print(f"  [ERROR] {style}: 제품 선택 실패")
        return {"style": style, "ok": False, "error": "no_products"}

    products = []
    for item in selected:
        w, d = DEFAULT_SIZES_MM.get(item["category"], (None, None))
        products.append({
            "category": item["category"], "name": item["title"][:50],
            "image_id": item["id"], "width_mm": w, "depth_mm": d,
        })

    payload = {
        "image_base64":          to_b64(desk_path),
        "style":                 style,
        "products":              products,
        "desk_width_mm":         desk_width_mm,
        "desk_depth_mm":         desk_depth_mm,
        "top_view_image_base64": to_b64(top_path) if top_path else None,
        "mode":                  "own_desk",
        "generation_mode":       "controlnet",
        "removal_strategy":      "combined",
    }

    t0 = time.time()
    resp = requests.post(f"{AI_SERVER}/generate", json=payload, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"  job_id={job_id}")
    result = poll(job_id)
    elapsed = time.time() - t0

    style_dir = out_dir / style
    style_dir.mkdir(parents=True, exist_ok=True)
    if result.get("cleaned_image"):
        (style_dir / "step1_cleaned.png").write_bytes(base64.b64decode(result["cleaned_image"]))
    (style_dir / "step3_final.png").write_bytes(base64.b64decode(result["result_image"]))
    (style_dir / "meta.json").write_text(json.dumps({
        "style": style, "budget": budget, "categories": cats,
        "desk_image":         str(desk_path.relative_to(REPO_ROOT)),
        "top_view_image":     str(top_path.relative_to(REPO_ROOT)) if top_path else None,
        "desk_width_mm":      desk_width_mm,
        "desk_depth_mm":      desk_depth_mm,
        "selected_products": [
            {"category": p["category"], "image_id": p["id"], "title": p["title"],
             "mock_price": p.get("mock_price")}
            for p in selected
        ],
        "total_price":        total,
        "elapsed_s":          round(elapsed, 1),
        "num_removed":        result.get("num_removed"),
        "num_placed":         result.get("num_placed"),
        "seed":               seed,
        "job_id":             job_id,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  완료: num_placed={result.get('num_placed')}, elapsed={elapsed:.1f}s → {style_dir}")
    return {"style": style, "ok": True, "elapsed": elapsed,
            "num_placed": result.get("num_placed"), "out_dir": str(style_dir)}


def main():
    parser = argparse.ArgumentParser(description="여러 style 한 timestamp 폴더에 일괄 테스트")
    parser.add_argument("--styles", type=str, default="white,black,gaming")
    parser.add_argument("--budget", type=int, default=None,
                        help="모든 style에 동일 budget. 지정 안 하면 style별 기본값 사용")
    parser.add_argument("--categories", type=str,
                        default="MONITOR,KEYBOARD,MOUSE,SPEAKER,DESK_LAMP")
    parser.add_argument("--desk", type=str, default=None,
                        help="data/test/<name>.png|jpg (예: front_view3, desk_image2)")
    parser.add_argument("--top", type=str, default=None,
                        help="data/test/<name>.png|jpg (예: top_view3, desk_top_image2)")
    parser.add_argument("--desk-width-mm",  type=int, default=1400)
    parser.add_argument("--desk-depth-mm",  type=int, default=700)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    cats   = [c.strip() for c in args.categories.split(",") if c.strip()]
    desk_path, top_path = resolve_desk_paths(args.desk, args.top)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "ai-server" / "outputs" / "test_results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Batch Test ===")
    print(f"timestamp:  {ts}")
    print(f"desk:       {desk_path.name}")
    print(f"top_view:   {top_path.name if top_path else '(none)'}")
    print(f"styles:     {styles}")
    print(f"categories: {cats}")
    print(f"out:        {out_dir}")

    results = []
    for style in styles:
        budget = args.budget if args.budget else _DEFAULT_BUDGETS.get(style, 500000)
        try:
            r = run_one_style(
                style=style, budget=budget, cats=cats,
                desk_path=desk_path, top_path=top_path,
                desk_width_mm=args.desk_width_mm, desk_depth_mm=args.desk_depth_mm,
                out_dir=out_dir, seed=args.seed,
            )
        except Exception as e:
            print(f"  [ERROR] {style}: {e}")
            r = {"style": style, "ok": False, "error": str(e)}
        results.append(r)

    # 통합 요약
    (out_dir / "batch_summary.json").write_text(
        json.dumps({
            "timestamp":  ts,
            "desk_image": str(desk_path.relative_to(REPO_ROOT)),
            "top_view":   str(top_path.relative_to(REPO_ROOT)) if top_path else None,
            "results":    results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n=== 완료 ===")
    for r in results:
        if r["ok"]:
            print(f"  [{r['style']:6}] placed={r['num_placed']}  {r['elapsed']:.1f}s")
        else:
            print(f"  [{r['style']:6}] FAILED: {r.get('error')}")
    print(f"\n최종 폴더: {out_dir}")


if __name__ == "__main__":
    main()
