import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from api.dino_processor import run_grounding_dino

RAW_DIR        = Path("data/style_lora/raw")
OUT_JSONL      = Path("data/layout_dataset.jsonl")
CKPT_FILE      = Path("data/.layout_dataset_ckpt.json")
FRONTVIEW_LIST = Path("data/frontview_images.json")
DEBUG_DIR      = Path("data/debug/layout")

DESK_PROMPT = "desk. table. wooden desk."
OBJ_PROMPT  = (
    "monitor. computer monitor. display screen. lcd screen. led monitor. computer display. screen. "
    "keyboard. mechanical keyboard. computer keyboard. wireless keyboard. "
    "mouse. computer mouse. wireless mouse. gaming mouse. "
    "mouse pad. mousepad. desk mat. extended mouse pad. large desk mat. "
    "speaker. desktop speaker. computer speaker. bookshelf speaker. soundbar. "
    "desk lamp. table lamp. lamp. "
    "desk shelf. monitor riser. monitor stand."
)

# DINO 라벨을 카테고리로 매핑. substring 매칭으로 동작 (예: "led monitor" 라벨 → MONITOR).
# 매핑 순서 중요 — 더 구체적인 키를 위에 두기 (mouse pad가 mouse보다 먼저).
_DINO_TO_CAT_RULES = [
    ("monitor riser",   "DESK_SHELF"),
    ("monitor stand",   "DESK_SHELF"),
    ("desk shelf",      "DESK_SHELF"),
    ("mouse pad",       "MOUSEPAD"),
    ("mousepad",        "MOUSEPAD"),
    ("desk mat",        "MOUSEPAD"),
    ("desk lamp",       "DESK_LAMP"),
    ("table lamp",      "DESK_LAMP"),
    ("soundbar",        "SPEAKER"),
    ("speaker",         "SPEAKER"),
    ("keyboard",        "KEYBOARD"),
    ("monitor",         "MONITOR"),
    ("display",         "MONITOR"),
    ("lcd",             "MONITOR"),
    ("led monitor",     "MONITOR"),
    ("screen",          "MONITOR"),
    ("mouse",           "MOUSE"),
    ("lamp",            "DESK_LAMP"),
]


def _map_dino_label(label: str) -> str | None:
    lab = (label or "").lower().strip()
    for key, cat in _DINO_TO_CAT_RULES:
        if key in lab:
            return cat
    return None


# 카테고리별 aspect ratio (w/h) 허용 범위.
# MONITOR: DINO가 받침대까지 묶어 박스를 그리면 h가 커져 ar < 1.1 → 폐기되던 문제.
#          (0.8, 4.0)으로 완화 — 받침대 포함 모니터도 통과.
# KEYBOARD: (2.5, 8.0) → (1.8, 8.0). 컴팩트/TKL 키보드도 포함.
_CAT_ASPECT_VALID = {
    "MONITOR":   (0.8, 4.0),
    "KEYBOARD":  (1.8, 8.0),
    "MOUSE":     (0.5, 2.5),
    "MOUSEPAD":  (1.3, 6.0),
    "SPEAKER":   (0.3, 5.0),
    "DESK_LAMP": (0.15, 2.0),
    "DESK_SHELF":(1.5, 6.0),
}

# 카테고리별 desk bbox 대비 최소 면적 비율.
# MONITOR: 0.020 → 0.006 — 모니터는 책상 위로 솟아 있어 desk_bbox 내부 면적이 작게 잡힘.
#          기존 임계값이 모니터를 대량 누락시킨 주된 원인.
_CAT_MIN_AREA_RATIO = {
    "MONITOR":   0.006,
    "KEYBOARD":  0.010,
    "MOUSE":     0.002,
    "MOUSEPAD":  0.020,
    "SPEAKER":   0.003,
    "DESK_LAMP": 0.002,
    "DESK_SHELF":0.010,
}

# debug overlay 색상 (BGR → PIL RGB)
_CAT_COLOR = {
    "MONITOR":   (255,  80,  80),
    "KEYBOARD":  ( 80, 200,  80),
    "MOUSE":     ( 80,  80, 255),
    "MOUSEPAD":  (200, 200,  80),
    "SPEAKER":   (200,  80, 200),
    "DESK_LAMP": ( 80, 200, 200),
    "DESK_SHELF":(200, 140,  80),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _to_small_bgr(pil_img, scale):
    w = max(1, int(pil_img.width  * scale))
    h = max(1, int(pil_img.height * scale))
    return np.array(pil_img.convert("RGB").resize((w, h)))[:, :, ::-1].copy()


def detect_desk(pil_img, scale):
    bgr  = _to_small_bgr(pil_img, scale)
    dets = run_grounding_dino(bgr, DESK_PROMPT, box_threshold=0.25, max_area_ratio=0.98)
    if not dets:
        return None
    best = max(dets, key=lambda d: (d.box_xyxy[2]-d.box_xyxy[0])*(d.box_xyxy[3]-d.box_xyxy[1]))
    inv  = 1.0 / scale
    return [int(v * inv) for v in best.box_xyxy]


def _passes_filter(cat: str, ox1: int, oy1: int, ox2: int, oy2: int,
                   desk_bbox: list) -> tuple[bool, str]:
    bw = max(1, ox2 - ox1)
    bh = max(1, oy2 - oy1)
    ar = bw / bh

    ar_range = _CAT_ASPECT_VALID.get(cat)
    if ar_range and not (ar_range[0] <= ar <= ar_range[1]):
        return False, f"aspect_ratio={ar:.2f} out of {ar_range}"

    dx1, dy1, dx2, dy2 = desk_bbox
    desk_area = max(1, (dx2-dx1) * (dy2-dy1))
    obj_area  = bw * bh
    min_ratio = _CAT_MIN_AREA_RATIO.get(cat, 0.001)
    if obj_area / desk_area < min_ratio:
        return False, f"area_ratio={obj_area/desk_area:.4f} < {min_ratio}"

    return True, ""


def detect_objects(pil_img, scale, desk_bbox):
    dx1, dy1, dx2, dy2 = desk_bbox
    dw   = max(1, dx2 - dx1)
    dh   = max(1, dy2 - dy1)
    bgr  = _to_small_bgr(pil_img, scale)
    dets = run_grounding_dino(bgr, OBJ_PROMPT, box_threshold=0.28, max_area_ratio=0.80)

    objects      = []
    filter_stats = {}

    for d in dets:
        cat = _map_dino_label(d.label)
        if cat is None:
            continue
        inv = 1.0 / scale
        ox1, oy1, ox2, oy2 = [int(v * inv) for v in d.box_xyxy]
        cx = (ox1 + ox2) / 2
        cy = (oy1 + oy2) / 2
        rx = (cx - dx1) / dw
        ry = (cy - dy1) / dh
        rw = (ox2 - ox1) / dw
        rh = (oy2 - oy1) / dh

        # MONITOR는 책상 위로 솟아 있어 ry < 0인 경우가 정상.
        # 카테고리별 허용 범위 — MONITOR/DESK_SHELF/DESK_LAMP/SPEAKER는 ry < -0.30까지 허용.
        if cat in ("MONITOR", "DESK_SHELF", "DESK_LAMP", "SPEAKER"):
            ry_min = -0.50
        else:
            ry_min = -0.15
        if not (-0.15 <= rx <= 1.15 and ry_min <= ry <= 1.15):
            filter_stats[cat] = filter_stats.get(cat, 0) + 1
            continue

        ok, reason = _passes_filter(cat, ox1, oy1, ox2, oy2, desk_bbox)
        if not ok:
            filter_stats[cat] = filter_stats.get(cat, 0) + 1
            continue

        objects.append({
            "category":   cat,
            "bbox":       [ox1, oy1, ox2, oy2],
            "score":      round(float(d.score), 3),
            "rx":         round(float(rx), 4),
            "ry":         round(float(ry), 4),
            "rw":         round(float(rw), 4),
            "rh":         round(float(rh), 4),
        })

    return objects, filter_stats


def save_debug_overlay(pil_img, desk_bbox, objects, out_path: Path):
    vis = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    dx1, dy1, dx2, dy2 = desk_bbox
    draw.rectangle([dx1, dy1, dx2, dy2], outline=(255, 255, 0), width=3)
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        color = _CAT_COLOR.get(obj["category"], (200, 200, 200))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1+2, y1+2), f"{obj['category']} {obj['score']:.2f}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="detection overlay 저장")
    args = parser.parse_args()

    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = lambda x, **kw: x

    processed: set = set()
    if CKPT_FILE.exists():
        processed = set(json.loads(CKPT_FILE.read_text(encoding="utf-8")))
        print(f"[resume] 이미 처리됨: {len(processed)}개")

    if FRONTVIEW_LIST.exists():
        _fv_data  = json.loads(FRONTVIEW_LIST.read_text(encoding="utf-8"))
        _fv_paths = set(_fv_data.get("front_view", []))
        all_images = sorted(
            p for p in RAW_DIR.rglob("*")
            if p.suffix.lower() in IMG_EXTS
            and str(p) in _fv_paths
            and str(p) not in processed
        )
        print(f"[build] front-view 필터 적용: {len(_fv_paths)}장 중 처리 대상 {len(all_images)}개")
    else:
        all_images = sorted(
            p for p in RAW_DIR.rglob("*")
            if p.suffix.lower() in IMG_EXTS and str(p) not in processed
        )
        print(f"[build] 처리 대상: {len(all_images)}개 (front-view 필터 없음)")

    stats        = {"total": 0, "desk_fail": 0, "no_objects": 0, "written": 0, "error": 0}
    cat_counts   = {}
    filter_total = {}

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("a", encoding="utf-8") as fout:
        for img_path in _tqdm(all_images, desc="layout_dataset"):
            stats["total"] += 1
            try:
                pil_img = Image.open(img_path).convert("RGB")
                w, h    = pil_img.size
                scale   = min(512 / max(w, 1), 512 / max(h, 1))

                desk_bbox = detect_desk(pil_img, scale)
                if desk_bbox is None:
                    stats["desk_fail"] += 1
                    processed.add(str(img_path))
                    continue

                objects, fstats = detect_objects(pil_img, scale, desk_bbox)
                for k, v in fstats.items():
                    filter_total[k] = filter_total.get(k, 0) + v

                if not objects:
                    stats["no_objects"] += 1
                    processed.add(str(img_path))
                    continue

                for obj in objects:
                    cat_counts[obj["category"]] = cat_counts.get(obj["category"], 0) + 1

                parts = img_path.relative_to(RAW_DIR).parts
                style = parts[0] if parts else "unknown"

                row = {
                    "image_id":   img_path.stem,
                    "image_path": str(img_path),
                    "style":      style,
                    "img_w":      w,
                    "img_h":      h,
                    "desk_bbox":  desk_bbox,
                    "objects":    objects,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written"] += 1

                if args.debug and stats["written"] <= 50:
                    save_debug_overlay(
                        pil_img, desk_bbox, objects,
                        DEBUG_DIR / f"{img_path.stem}_overlay.jpg",
                    )

            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")
                stats["error"] += 1
            finally:
                processed.add(str(img_path))

            if stats["total"] % 50 == 0:
                CKPT_FILE.write_text(json.dumps(list(processed)), encoding="utf-8")
                print(f"  [{stats['total']}] written={stats['written']} "
                      f"desk_fail={stats['desk_fail']} no_obj={stats['no_objects']}")

    CKPT_FILE.write_text(json.dumps(list(processed)), encoding="utf-8")
    print(f"\n[done] total={stats['total']} desk_fail={stats['desk_fail']} "
          f"no_objects={stats['no_objects']} written={stats['written']} error={stats['error']}")
    print(f"[category counts] {cat_counts}")
    if filter_total:
        print(f"[filtered out]    {filter_total}")
    print(f"  → {OUT_JSONL}")


if __name__ == "__main__":
    main()
