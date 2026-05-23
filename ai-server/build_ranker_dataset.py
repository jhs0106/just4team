import json
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

IN_JSONL    = Path("data/layout_dataset.jsonl")
OUT_JSONL   = Path("data/placement_ranker_dataset.jsonl")
NEG_PER_POS = 4   # negative 전략당 최대 1개씩, 총 최대 4개
RANDOM_SEED = 42
DEBUG_DIR   = Path("data/debug/ranker")

_PREFERRED_POS = {
    "MONITOR":      {"rx": 0.50, "ry": 0.20},
    "DESK_SHELF":   {"rx": 0.50, "ry": 0.20},
    "KEYBOARD":     {"rx": 0.50, "ry": 0.62},
    "MOUSEPAD":     {"rx": 0.50, "ry": 0.65},
    "MOUSE":        {"rx": 0.70, "ry": 0.62},
    "SPEAKER":      {"rx": 0.25, "ry": 0.25},
    "DESK_LAMP":    {"rx": 0.12, "ry": 0.30},
    "DECO":         {"rx": 0.75, "ry": 0.35},
    "CLOCK":        {"rx": 0.80, "ry": 0.30},
    "LAPTOP_STAND": {"rx": 0.50, "ry": 0.45},
}

_CAT_ID = {
    "MONITOR": 0, "KEYBOARD": 1, "MOUSE": 2, "MOUSEPAD": 3,
    "SPEAKER": 4, "DESK_LAMP": 5, "DESK_SHELF": 6,
    "LAPTOP_STAND": 7, "DECO": 8, "CLOCK": 9,
}

# 카테고리별 "잘못된 zone" 정의 (negative 샘플링용)
_WRONG_ZONE = {
    "MONITOR":   lambda rx, ry: ry > 0.60,  # 모니터가 책상 앞쪽에 있으면 비정상
    "KEYBOARD":  lambda rx, ry: ry < 0.30,  # 키보드가 책상 뒤쪽에 있으면 비정상
    "MOUSE":     lambda rx, ry: ry < 0.30,
    "MOUSEPAD":  lambda rx, ry: ry < 0.30,
    "SPEAKER":   lambda rx, ry: ry > 0.70,
    "DESK_LAMP": lambda rx, ry: 0.25 <= rx <= 0.75,  # 램프가 중앙에 있으면 비정상
}

FEATURE_NAMES = [
    # 기본 위치/크기
    "cat_id",
    "rx", "ry", "rw", "rh",
    # preferred position 거리
    "dist_to_preferred",
    # 경계 여유
    "edge_margin_x", "edge_margin_y",
    # 모니터 상대
    "monitor_rx", "monitor_ry",
    "dist_to_monitor", "dx_to_monitor", "dy_to_monitor",
    # 키보드 상대
    "keyboard_rx", "keyboard_ry",
    # 책상 중심 거리
    "center_distance",
    # zone 플래그
    "is_left", "is_right", "is_back", "is_front",
    # 기존 객체와 겹침
    "overlap_with_existing",
    # 주변 맥락
    "n_other_objects",
]


def _box(rx, ry, rw, rh):
    return [rx - rw/2, ry - rh/2, rx + rw/2, ry + rh/2]


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter  = iw * ih
    a_area = max(1e-8, (a[2]-a[0]) * (a[3]-a[1]))
    b_area = max(1e-8, (b[2]-b[0]) * (b[3]-b[1]))
    return inter / (a_area + b_area - inter + 1e-8)


def make_feature(cat: str, rx: float, ry: float, rw: float, rh: float, objects: list) -> list:
    pref        = _PREFERRED_POS.get(cat, {"rx": 0.5, "ry": 0.5})
    dist_pref   = ((rx - pref["rx"])**2 + (ry - pref["ry"])**2) ** 0.5
    edge_x      = min(rx, 1.0 - rx)
    edge_y      = min(ry, 1.0 - ry)

    mon = next((o for o in objects if o["category"] == "MONITOR"), None)
    monitor_rx  = mon["rx"] if mon else -1.0
    monitor_ry  = mon["ry"] if mon else -1.0
    if mon:
        dx_mon  = rx - mon["rx"]
        dy_mon  = ry - mon["ry"]
        dist_mon = (dx_mon**2 + dy_mon**2) ** 0.5
    else:
        dx_mon = dy_mon = dist_mon = -1.0

    kb = next((o for o in objects if o["category"] == "KEYBOARD"), None)
    keyboard_rx = kb["rx"] if kb else -1.0
    keyboard_ry = kb["ry"] if kb else -1.0

    center_dist = ((rx - 0.5)**2 + (ry - 0.5)**2) ** 0.5

    # zone 플래그
    is_left  = 1 if rx < 0.35 else 0
    is_right = 1 if rx > 0.65 else 0
    is_back  = 1 if ry < 0.35 else 0
    is_front = 1 if ry > 0.65 else 0

    # 기존 객체와 최대 IoU
    candidate_box = _box(rx, ry, rw, rh)
    overlap = max(
        (_iou(candidate_box, _box(o["rx"], o["ry"], o["rw"], o["rh"]))
         for o in objects if o["category"] != cat),
        default=0.0,
    )

    n_others = len([o for o in objects if o["category"] != cat])

    return [
        _CAT_ID.get(cat, -1),
        round(rx, 4), round(ry, 4), round(rw, 4), round(rh, 4),
        round(dist_pref, 4),
        round(edge_x, 4), round(edge_y, 4),
        round(monitor_rx, 4), round(monitor_ry, 4),
        round(dist_mon, 4), round(dx_mon, 4), round(dy_mon, 4),
        round(keyboard_rx, 4), round(keyboard_ry, 4),
        round(center_dist, 4),
        is_left, is_right, is_back, is_front,
        round(overlap, 4),
        n_others,
    ]


# ── Negative 샘플링 전략 4종 ──────────────────────────────────────────────────

def _neg_random_clean(rw, rh, objects, rng, max_try=30):
    """기존 객체와 겹치지 않는 랜덤 위치."""
    half_w = rw / 2; half_h = rh / 2
    pos_boxes = [_box(o["rx"], o["ry"], o["rw"], o["rh"]) for o in objects]
    for _ in range(max_try):
        rx = rng.uniform(half_w, 1.0 - half_w)
        ry = rng.uniform(half_h, 1.0 - half_h)
        if all(_iou(_box(rx, ry, rw, rh), pb) < 0.20 for pb in pos_boxes):
            return rx, ry
    return None


def _neg_overlap(rw, rh, objects, rng, max_try=20):
    """기존 객체와 의도적으로 겹치는 위치 (명확히 나쁜 배치)."""
    if not objects:
        return None
    target = rng.choice(objects)
    for _ in range(max_try):
        jitter_x = rng.uniform(-rw * 0.5, rw * 0.5)
        jitter_y = rng.uniform(-rh * 0.5, rh * 0.5)
        rx = max(rw/2, min(1.0 - rw/2, target["rx"] + jitter_x))
        ry = max(rh/2, min(1.0 - rh/2, target["ry"] + jitter_y))
        if _iou(_box(rx, ry, rw, rh), _box(target["rx"], target["ry"], target["rw"], target["rh"])) > 0.30:
            return rx, ry
    return None


def _neg_extreme_edge(rw, rh, rng):
    """책상 가장자리 극단 위치."""
    side = rng.choice(["left", "right", "top", "bottom"])
    half_w = rw / 2; half_h = rh / 2
    if side == "left":
        return rng.uniform(half_w, half_w + 0.05), rng.uniform(half_h, 1.0 - half_h)
    if side == "right":
        return rng.uniform(1.0 - half_w - 0.05, 1.0 - half_w), rng.uniform(half_h, 1.0 - half_h)
    if side == "top":
        return rng.uniform(half_w, 1.0 - half_w), rng.uniform(half_h, half_h + 0.05)
    return rng.uniform(half_w, 1.0 - half_w), rng.uniform(1.0 - half_h - 0.05, 1.0 - half_h)


def _neg_wrong_zone(cat, rw, rh, rng, max_try=20):
    """카테고리에 맞지 않는 잘못된 zone."""
    wrong_fn = _WRONG_ZONE.get(cat)
    if wrong_fn is None:
        return None
    half_w = rw / 2; half_h = rh / 2
    for _ in range(max_try):
        rx = rng.uniform(half_w, 1.0 - half_w)
        ry = rng.uniform(half_h, 1.0 - half_h)
        if wrong_fn(rx, ry):
            return rx, ry
    return None


# ── Debug 시각화 ─────────────────────────────────────────────────────────────

def save_debug_ranker(image_id, objects, positives, negatives, out_path: Path):
    size = 400
    img  = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, size-1, size-1], outline=(180, 180, 180), width=2)

    for nx, ny in negatives:
        px, py = int(nx * size), int(ny * size)
        draw.ellipse([px-4, py-4, px+4, py+4], fill=(200, 80, 80))

    for px_r, py_r, cat in positives:
        px, py = int(px_r * size), int(py_r * size)
        draw.ellipse([px-6, py-6, px+6, py+6], fill=(80, 200, 80))
        draw.text((px+7, py-6), cat[:3], fill=(50, 150, 50))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="positive/negative 시각화 저장")
    args = parser.parse_args()

    if not IN_JSONL.exists():
        print(f"[error] {IN_JSONL} 없음 — build_layout_dataset.py 먼저 실행")
        return

    rows = [json.loads(l) for l in IN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[ranker] 입력: {len(rows)}개 이미지 / feature_dim={len(FEATURE_NAMES)}")

    rng   = random.Random(RANDOM_SEED)
    stats = {"pos": 0, "neg_clean": 0, "neg_overlap": 0,
             "neg_edge": 0, "neg_zone": 0, "neg_fail": 0}

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fout:
        for row in rows:
            objects  = row["objects"]
            img_id   = row["image_id"]
            style    = row.get("style", "")
            pos_list = []
            neg_list = []

            for obj in objects:
                cat           = obj["category"]
                rx, ry, rw, rh = obj["rx"], obj["ry"], obj["rw"], obj["rh"]

                # positive
                fout.write(json.dumps({
                    "image_id":      img_id,
                    "style":         style,
                    "category":      cat,
                    "label":         1,
                    "neg_type":      "positive",
                    "features":      make_feature(cat, rx, ry, rw, rh, objects),
                    "feature_names": FEATURE_NAMES,
                }, ensure_ascii=False) + "\n")
                stats["pos"] += 1
                pos_list.append((rx, ry, cat))

                # negative 4종
                neg_strategies = [
                    ("clean",   _neg_random_clean(rw, rh, objects, rng)),
                    ("overlap", _neg_overlap(rw, rh, objects, rng)),
                    ("edge",    _neg_extreme_edge(rw, rh, rng)),
                    ("zone",    _neg_wrong_zone(cat, rw, rh, rng)),
                ]
                for neg_type, result in neg_strategies:
                    if result is None:
                        stats["neg_fail"] += 1
                        continue
                    neg_rx, neg_ry = result
                    fout.write(json.dumps({
                        "image_id":      img_id,
                        "style":         style,
                        "category":      cat,
                        "label":         0,
                        "neg_type":      neg_type,
                        "features":      make_feature(cat, neg_rx, neg_ry, rw, rh, objects),
                        "feature_names": FEATURE_NAMES,
                    }, ensure_ascii=False) + "\n")
                    stats[f"neg_{neg_type}"] += 1
                    neg_list.append((neg_rx, neg_ry))

            if args.debug and stats["pos"] <= 200:
                save_debug_ranker(
                    img_id, objects, pos_list, neg_list,
                    DEBUG_DIR / f"{img_id}_ranker.png",
                )

    total_neg = stats["neg_clean"] + stats["neg_overlap"] + stats["neg_edge"] + stats["neg_zone"]
    total     = stats["pos"] + total_neg
    print(f"[done] positive={stats['pos']} negative={total_neg} "
          f"(clean={stats['neg_clean']} overlap={stats['neg_overlap']} "
          f"edge={stats['neg_edge']} zone={stats['neg_zone']}) "
          f"fail={stats['neg_fail']}")
    print(f"  total={total}  pos_ratio={stats['pos']/max(total,1):.2f}")
    print(f"  feature_dim={len(FEATURE_NAMES)}: {FEATURE_NAMES}")
    print(f"  → {OUT_JSONL}")


if __name__ == "__main__":
    main()
