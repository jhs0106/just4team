"""
유효한 테스트 제품 이미지 ID 탐색 유틸리티

사용법:
    python find_valid_products.py --cat KEYBOARD
    python find_valid_products.py --cat MOUSE
    python find_valid_products.py --cat MONITOR
    python find_valid_products.py          # 전체 카테고리

processed_images/ 에 존재하는 이미지들을 대상으로 tight_crop aspect ratio를 계산해
TEST_FIXED_PRODUCTS에 쓸 수 있는 유효한 ID 목록을 출력한다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from api.main import prepare_product_image_for_composite, _CAT_ASPECT_VALID

PROCESSED_DIR = Path("processed_images")

# 카테고리별 image_id 범위 (products.csv 기준)
CAT_ID_RANGES: dict[str, tuple[int, int]] = {
    "MONITOR":  (177, 325),
    "KEYBOARD": (326, 521),
    "MOUSE":    (522, 720),
    "SPEAKER":  (721, 820),
    "DESK_LAMP":(821, 920),
}


def check_image(image_id: int, cat: str) -> dict | None:
    path = PROCESSED_DIR / f"{image_id}.png"
    if not path.exists():
        return None
    try:
        img   = Image.open(path)
        alpha = prepare_product_image_for_composite(img)
        ar    = alpha.width / max(alpha.height, 1)
        cov   = float((np.array(alpha.getchannel("A")) > 127).mean())
        return {"id": image_id, "ar": round(ar, 3), "alpha_cov": round(cov, 3),
                "size": f"{alpha.width}x{alpha.height}"}
    except Exception as e:
        return {"id": image_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", help="카테고리 (MONITOR/KEYBOARD/MOUSE 등)")
    parser.add_argument("--top", type=int, default=10, help="출력할 유효 ID 최대 개수")
    args = parser.parse_args()

    cats = [args.cat.upper()] if args.cat else list(CAT_ID_RANGES.keys())

    for cat in cats:
        id_range = CAT_ID_RANGES.get(cat)
        if id_range is None:
            print(f"[{cat}] 알 수 없는 카테고리 — CAT_ID_RANGES 확인 필요")
            continue

        ar_min, ar_max = _CAT_ASPECT_VALID.get(cat, (0.0, 99.0))
        valid   = []
        invalid = []
        missing = 0

        for img_id in range(id_range[0], id_range[1] + 1):
            r = check_image(img_id, cat)
            if r is None:
                missing += 1
                continue
            if "error" in r:
                continue
            if r["alpha_cov"] > 0.80:
                continue  # 배경 포함/멀티제품 의심
            if ar_min <= r["ar"] <= ar_max:
                valid.append(r)
            else:
                invalid.append(r)

        print(f"\n=== {cat}  (ar 유효 범위: {ar_min}~{ar_max}) ===")
        print(f"  스캔: {id_range[1]-id_range[0]+1}개  |  파일 없음: {missing}개  |  유효: {len(valid)}개  |  범위 밖: {len(invalid)}개")

        if valid:
            print(f"\n  [유효 ID — TEST_FIXED_PRODUCTS에 사용 가능]")
            for r in valid[:args.top]:
                print(f"    id={r['id']:4d}  ar={r['ar']:.3f}  alpha_cov={r['alpha_cov']:.2f}  size={r['size']}")
        else:
            print(f"  유효한 이미지 없음. 다음 ID 범위 중 ar 가장 가까운 것:")
            by_diff = sorted(invalid, key=lambda x: min(abs(x["ar"]-ar_min), abs(x["ar"]-ar_max)))
            for r in by_diff[:5]:
                print(f"    id={r['id']:4d}  ar={r['ar']:.3f}  (범위: {ar_min}~{ar_max})")


if __name__ == "__main__":
    main()
