# 임시 통합 테스트 — 실제 recommendation 시스템 없이 사용자 입력만 모사.
#
# 사용자 입력:
#   --style:  black | white | gaming 중 선택
#   --budget: 예산 (원 단위) — CSV에 가격 없으므로 모의 가격으로 계산
#
# 동작:
#   1. data/test/products.csv에서 style 키워드 매칭 제품을 카테고리별 후보로 모음
#   2. 모의 가격(카테고리 평균치)을 부여하고 예산 내에서 조합 가능한 것만 통과
#   3. AI 서버 /generate 호출 → 결과 이미지 저장
#
# style 생성 품질 검증이 목적이므로 책상 사진은 data/test/desk_image.jpg 고정 사용.

import argparse
import base64
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


REPO_ROOT     = Path(__file__).resolve().parents[2]
AI_SERVER     = "http://localhost:8000"
DESK_IMAGE    = REPO_ROOT / "ai-server" / "data" / "test" / "desk_image2.jpg"
DESK_TOP_IMG  = REPO_ROOT / "ai-server" / "data" / "test" / "desk_top_image2.jpg"
PRODUCTS_CSV  = REPO_ROOT / "ai-server" / "data" / "test" / "products.csv"


# style별 CSV 키워드 매칭
STYLE_KEYWORDS: dict[str, list[str]] = {
    "white":  ["화이트", "white", "흰",   "아이보리"],
    "black":  ["블랙",   "black", "검정", "다크",  "dark"],
    "gaming": ["게이밍", "gaming", "RGB", "rgb",  "esports", "리그",
               "기계식", "기계식 키보드"],
}


# 결과 품질이 나쁜 제품 ID — 베젤 제거된 모니터·듀얼 컷 등
PRODUCT_BLACKLIST_IDS: set[int] = {
    196,   # MONITOR: 베젤만 제거된 화면 이미지
    247,   # MONITOR: 동일
    235,   # MONITOR: 듀얼 컷 마케팅 이미지
}


# (style, category) → image_id: 검증된 깔끔한 단일 제품 컷 우선 사용.
# random.choice보다 우선. 워터마크/추가 제품/lifestyle 컷이 섞이지 않도록 명시 지정.
# 추가 검증 후 확장.
PREFERRED_IDS: dict[tuple[str, str], int] = {
    ("white",  "KEYBOARD"): 383,   # 애플 매직 키보드 화이트 (정면 단품)
    ("white",  "MONITOR"):  222,   # white MONITOR — 사용자 지정
    ("black",  "MONITOR"):  266,   # QNIX QX24D (단일 정면, 베젤 정상)
    ("gaming", "MOUSE"):    533,   # Logitech G502 HERO (클래식 게이밍 단품)
}

# PREFERRED ID의 실측 mm (DEFAULT_SIZES_MM 덮어쓰기).
# 카테고리 평균 mm이 제품에 안 맞을 때 정확한 크기로 배치하기 위함.
PREFERRED_PRODUCT_MM: dict[int, tuple[int, int]] = {
    383: (279, 115),    # Apple Magic Keyboard (basic, 텐키리스 무선)
    266: (542, 211),    # QNIX QX24D 24" 모니터
    533: (132, 75),     # Logitech G502 HERO 마우스
}


# 카테고리별 모의 가격 (CSV에 가격 없으므로 평균치로 budget 계산)
MOCK_PRICES_KRW: dict[str, int] = {
    "MONITOR":      350000,
    "KEYBOARD":      80000,
    "MOUSE":         50000,
    "MOUSEPAD":      20000,
    "SPEAKER":       80000,
    "DESK_LAMP":     50000,
    "DESK_SHELF":    40000,
    "LAPTOP_STAND":  30000,
    "CLOCK":         30000,
    "DECO":          20000,
    "LIGHTING":      80000,
}


# 카테고리별 표준 치수 (mm) — AI 서버 CSV 카탈로그에 없을 때 폴백
DEFAULT_SIZES_MM: dict[str, tuple[int, int]] = {
    "MONITOR":      (600, 200),
    "KEYBOARD":     (440, 130),
    "MOUSE":        (70,  120),
    "MOUSEPAD":     (900, 400),
    "SPEAKER":      (90,  120),
    "DESK_LAMP":    (80,  400),
    "DESK_SHELF":   (600, 200),
    "LAPTOP_STAND": (280, 250),
    "CLOCK":        (100, 100),
    "DECO":         (80,  80),
    "LIGHTING":     (500, 50),
}


def to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def load_products_by_style(style: str) -> dict[str, list[dict]]:
    # style 키워드와 title이 매칭되는 제품을 카테고리별로 모음. 블랙리스트 ID 제외.
    # PREFERRED_IDS에 (style, cat)으로 명시된 ID는 키워드 미매칭이어도 강제 포함
    #   (사용자가 특정 제품을 지정한 경우 키워드 필터 우회 — 게이밍 모니터를 white 셋업에 쓰는 등).
    keywords = STYLE_KEYWORDS.get(style, [])
    preferred_ids_for_style = {pid for (s, _c), pid in PREFERRED_IDS.items() if s == style}
    by_cat: dict[str, list[dict]] = {}
    with open(PRODUCTS_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat in ("DESK",):
                continue
            try:
                pid = int(row["id"])
            except (KeyError, ValueError):
                continue
            if pid in PRODUCT_BLACKLIST_IDS:
                continue
            title = row.get("title", "")
            _kw_match  = any(kw in title for kw in keywords)
            _preferred = pid in preferred_ids_for_style
            if _kw_match or _preferred:
                by_cat.setdefault(cat, []).append({"id": pid, "title": title, "category": cat})
    return by_cat


def select_setup(style: str, budget: int,
                 wanted_cats: list[str]) -> tuple[list[dict], int]:
    # style 매칭 제품 후보에서 카테고리별 1개씩 골라 budget 안에서 통과되는 조합 반환.
    # 가격은 모의 (MOCK_PRICES_KRW). budget 내에 못 들어가면 비싼 카테고리부터 drop.
    by_cat = load_products_by_style(style)
    selected: list[dict] = []
    total = 0
    drop_order = sorted(wanted_cats, key=lambda c: MOCK_PRICES_KRW.get(c, 0), reverse=True)

    for cat in wanted_cats:
        pool = by_cat.get(cat, [])
        if not pool:
            print(f"  [skip] {cat} — '{style}' 키워드 매칭 제품 없음")
            continue
        price = MOCK_PRICES_KRW.get(cat, 0)
        if total + price > budget:
            print(f"  [skip] {cat} — 예산 초과 (현재 {total:,} + {price:,} > {budget:,})")
            continue

        # PREFERRED_IDS에 (style, cat) 등록되어 있으면 강제 선택
        pref_id = PREFERRED_IDS.get((style, cat))
        chosen = None
        if pref_id is not None:
            for cand in pool:
                if cand["id"] == pref_id:
                    chosen = cand
                    break
            if chosen is None:
                # PREFERRED ID가 pool에 없으면 (스타일 키워드 매칭 실패 등) random fallback
                print(f"  [warn] {cat} PREFERRED id={pref_id} not in pool → random fallback")
        if chosen is None:
            chosen = random.choice(pool)

        chosen["mock_price"] = price
        selected.append(chosen)
        total += price
        _marker = " ★preferred" if pref_id and chosen["id"] == pref_id else ""
        print(f"  [pick] {cat:10} id={chosen['id']:5} price={price:,}원 → 누적 {total:,}원{_marker}")

    return selected, total


def build_payload(selected: list[dict], style: str,
                  desk_width_mm: int, desk_depth_mm: int,
                  top_view_b64: str | None = None) -> dict:
    products = []
    for item in selected:
        cat = item["category"]
        # PREFERRED_PRODUCT_MM이 등록된 id면 그 실측치 우선, 아니면 카테고리 기본값
        if item["id"] in PREFERRED_PRODUCT_MM:
            w, d = PREFERRED_PRODUCT_MM[item["id"]]
        else:
            w, d = DEFAULT_SIZES_MM.get(cat, (None, None))
        products.append({
            "category":  cat,
            "name":      item["title"][:50],
            "image_id":  item["id"],
            "width_mm":  w,
            "depth_mm":  d,
        })
    return {
        "image_base64":          to_b64(DESK_IMAGE),
        "style":                 style,
        "products":              products,
        "desk_width_mm":         desk_width_mm,
        "desk_depth_mm":         desk_depth_mm,
        "top_view_image_base64": top_view_b64,
        "mode":                  "own_desk",
        "generation_mode":       "harmonize",
        "removal_strategy":      "combined",
    }


def poll(job_id: str, timeout: float = 1200.0) -> dict:
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job {job_id} 타임아웃")
        res = requests.get(f"{AI_SERVER}/jobs/{job_id}", timeout=10).json()
        st  = res.get("status")
        if st == "done":
            return res
        if st == "failed":
            raise RuntimeError(f"job 실패: {res.get('error')}")
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="style+budget 입력 기반 임시 통합 테스트")
    parser.add_argument("--style",  type=str, required=True,
                        choices=["black", "white", "gaming"],
                        help="black / white / gaming 중 선택")
    parser.add_argument("--budget", type=int, required=True,
                        help="예산 (원 단위, 예: 500000)")
    parser.add_argument("--categories", type=str,
                        default="MONITOR,KEYBOARD,MOUSE,SPEAKER,DESK_LAMP",
                        help="포함할 카테고리 (쉼표 구분)")
    parser.add_argument("--desk-width-mm",  type=int, default=1400)
    parser.add_argument("--desk-depth-mm",  type=int, default=700)
    parser.add_argument("--top-view",       type=Path, default=None)
    parser.add_argument("--seed",           type=int, default=None,
                        help="random seed (재현성)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not DESK_IMAGE.exists():
        print(f"[ERROR] 책상 이미지 없음: {DESK_IMAGE}")
        sys.exit(1)

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    print(f"\n=== Styled Test ===")
    print(f"style:      {args.style}")
    print(f"budget:     {args.budget:,}원")
    print(f"categories: {cats}")
    print(f"desk:       {args.desk_width_mm}x{args.desk_depth_mm}mm\n")

    print(f"[1/3] 제품 선택 (style={args.style!r}, budget={args.budget:,}원)")
    selected, total = select_setup(args.style, args.budget, cats)
    if not selected:
        print(f"[ERROR] 선택된 제품 없음")
        sys.exit(1)
    print(f"  → {len(selected)}개 제품, 총 {total:,}원 (예산 {args.budget:,}원의 {total/args.budget*100:.1f}%)\n")

    print(f"[2/3] AI 서버 호출")
    # 명시적 --top-view 없으면 desk_image2와 짝인 desk_top_image2 자동 사용
    if args.top_view:
        top_b64 = to_b64(args.top_view)
    elif DESK_TOP_IMG.exists():
        top_b64 = to_b64(DESK_TOP_IMG)
        print(f"  top-view 자동 사용: {DESK_TOP_IMG.name}")
    else:
        top_b64 = None
    payload = build_payload(selected, args.style, args.desk_width_mm, args.desk_depth_mm, top_b64)
    resp    = requests.post(f"{AI_SERVER}/generate", json=payload, timeout=30)
    resp.raise_for_status()
    job_id  = resp.json()["job_id"]
    print(f"  job_id={job_id}")

    print(f"[3/3] polling...")
    t0 = time.time()
    result = poll(job_id)
    elapsed = time.time() - t0
    print(f"  완료: num_removed={result.get('num_removed')}, "
          f"num_placed={result.get('num_placed')}, elapsed={elapsed:.1f}s")

    # 기존 컨벤션과 동일: outputs/test_results/<timestamp>_<style>/
    # 같은 폴더에 step1_cleaned.png(빈 책상), step3_final.png(최종),
    # meta.json(테스트 입력값) 저장.
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "ai-server" / "outputs" / "test_results" / f"{ts}_{args.style}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.get("cleaned_image"):
        (out_dir / "step1_cleaned.png").write_bytes(base64.b64decode(result["cleaned_image"]))
    (out_dir / "step3_final.png").write_bytes(base64.b64decode(result["result_image"]))

    _meta = {
        "style":              args.style,
        "budget":             args.budget,
        "categories":         cats,
        "desk_width_mm":      args.desk_width_mm,
        "desk_depth_mm":      args.desk_depth_mm,
        "selected_products": [
            {"category": p["category"], "image_id": p["id"], "title": p["title"],
             "mock_price": p.get("mock_price")}
            for p in selected
        ],
        "total_price":        total,
        "elapsed_s":          round(elapsed, 1),
        "num_removed":        result.get("num_removed"),
        "num_placed":         result.get("num_placed"),
        "seed":               args.seed,
        "job_id":             job_id,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[결과 파일 — test_results]")
    print(f"  {out_dir / 'step1_cleaned.png'}  — LaMa 물체 제거 후")
    print(f"  {out_dir / 'step3_final.png'}    — 최종 결과")
    print(f"  {out_dir / 'meta.json'}          — 테스트 입력값")
    print(f"\n[디버그 파일 — outputs/debug/<timestamp>/]")
    print(f"  placement_debug.png  — bbox 시각화")
    print(f"  cv_composite_result.png — CV 합성 결과 (SD 전)")
    print(f"  products_list.json   — 좌표/점수/ranker 정보")
    print(f"  products/<cat>_*.png — 카테고리별 SD 중간 산출물")


if __name__ == "__main__":
    main()
