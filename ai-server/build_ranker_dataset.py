import json
import random
import numpy as np
from pathlib import Path

IN_JSONL    = Path("data/layout_dataset.jsonl")
OUT_JSONL   = Path("data/placement_ranker_dataset.jsonl")
NEG_PER_POS = 3
RANDOM_SEED = 42

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

FEATURE_NAMES = [
    "cat_id",
    "rx", "ry", "rw", "rh",
    "dist_to_preferred",
    "edge_margin_x", "edge_margin_y",
    "monitor_rx", "monitor_ry",
    "keyboard_rx", "keyboard_ry",
    "n_other_objects",
]


def make_feature(cat: str, rx: float, ry: float, rw: float, rh: float, objects: list) -> list:
    pref = _PREFERRED_POS.get(cat, {"rx": 0.5, "ry": 0.5})
    dist_pref  = ((rx - pref["rx"])**2 + (ry - pref["ry"])**2) ** 0.5
    edge_x     = min(rx, 1.0 - rx)
    edge_y     = min(ry, 1.0 - ry)
    monitor_rx  = next((o["rx"] for o in objects if o["category"] == "MONITOR"),  -1.0)
    monitor_ry  = next((o["ry"] for o in objects if o["category"] == "MONITOR"),  -1.0)
    keyboard_rx = next((o["rx"] for o in objects if o["category"] == "KEYBOARD"), -1.0)
    keyboard_ry = next((o["ry"] for o in objects if o["category"] == "KEYBOARD"), -1.0)
    n_others   = len([o for o in objects if o["category"] != cat])
    return [
        _CAT_ID.get(cat, -1),
        round(rx, 4), round(ry, 4), round(rw, 4), round(rh, 4),
        round(dist_pref, 4),
        round(edge_x, 4), round(edge_y, 4),
        round(monitor_rx, 4), round(monitor_ry, 4),
        round(keyboard_rx, 4), round(keyboard_ry, 4),
        n_others,
    ]


def _box(rx, ry, rw, rh):
    return [rx - rw / 2, ry - rh / 2, rx + rw / 2, ry + rh / 2]


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter   = iw * ih
    a_area  = max(1e-8, (a[2]-a[0]) * (a[3]-a[1]))
    b_area  = max(1e-8, (b[2]-b[0]) * (b[3]-b[1]))
    return inter / (a_area + b_area - inter + 1e-8)


def _sample_negative(rw: float, rh: float, all_objects: list, rng: random.Random, max_try: int = 30):
    half_w = rw / 2
    half_h = rh / 2
    pos_boxes = [_box(o["rx"], o["ry"], o["rw"], o["rh"]) for o in all_objects]
    for _ in range(max_try):
        rx = rng.uniform(half_w, 1.0 - half_w)
        ry = rng.uniform(half_h, 1.0 - half_h)
        neg = _box(rx, ry, rw, rh)
        if all(_iou(neg, pb) < 0.20 for pb in pos_boxes):
            return rx, ry
    return None


def main():
    if not IN_JSONL.exists():
        print(f"[error] {IN_JSONL} 없음 — build_layout_dataset.py 먼저 실행")
        return

    rows = [json.loads(l) for l in IN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[ranker] 입력: {len(rows)}개 이미지")

    rng   = random.Random(RANDOM_SEED)
    stats = {"pos": 0, "neg": 0, "neg_fail": 0}

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fout:
        for row in rows:
            objects = row["objects"]
            img_id  = row["image_id"]
            style   = row.get("style", "")

            for obj in objects:
                cat = obj["category"]
                rx, ry, rw, rh = obj["rx"], obj["ry"], obj["rw"], obj["rh"]

                # positive
                fout.write(json.dumps({
                    "image_id":      img_id,
                    "style":         style,
                    "category":      cat,
                    "label":         1,
                    "features":      make_feature(cat, rx, ry, rw, rh, objects),
                    "feature_names": FEATURE_NAMES,
                }, ensure_ascii=False) + "\n")
                stats["pos"] += 1

                # negative samples
                neg_count = 0
                for _ in range(NEG_PER_POS * 4):
                    if neg_count >= NEG_PER_POS:
                        break
                    result = _sample_negative(rw, rh, objects, rng)
                    if result is None:
                        stats["neg_fail"] += 1
                        continue
                    neg_rx, neg_ry = result
                    fout.write(json.dumps({
                        "image_id":      img_id,
                        "style":         style,
                        "category":      cat,
                        "label":         0,
                        "features":      make_feature(cat, neg_rx, neg_ry, rw, rh, objects),
                        "feature_names": FEATURE_NAMES,
                    }, ensure_ascii=False) + "\n")
                    stats["neg"] += 1
                    neg_count += 1

    total = stats["pos"] + stats["neg"]
    print(f"[done] positive={stats['pos']} negative={stats['neg']} neg_fail={stats['neg_fail']}")
    print(f"  total={total}  pos_ratio={stats['pos']/max(total,1):.2f}")
    print(f"  → {OUT_JSONL}")


if __name__ == "__main__":
    main()
