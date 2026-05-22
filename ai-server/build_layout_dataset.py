import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from api.dino_processor import run_grounding_dino

RAW_DIR   = Path("data/style_lora/raw")
OUT_JSONL = Path("data/layout_dataset.jsonl")
CKPT_FILE = Path("data/.layout_dataset_ckpt.json")

DESK_PROMPT = "desk. table. wooden desk."
OBJ_PROMPT  = (
    "monitor. keyboard. mouse. mouse pad. mousepad. "
    "speaker. desk lamp. lamp. desk shelf. monitor riser. clock. laptop."
)

_DINO_TO_CAT = {
    "monitor":        "MONITOR",
    "keyboard":       "KEYBOARD",
    "mouse":          "MOUSE",
    "mouse pad":      "MOUSEPAD",
    "mousepad":       "MOUSEPAD",
    "speaker":        "SPEAKER",
    "desk lamp":      "DESK_LAMP",
    "lamp":           "DESK_LAMP",
    "desk shelf":     "DESK_SHELF",
    "monitor riser":  "DESK_SHELF",
    "clock":          "CLOCK",
    "laptop":         "LAPTOP_STAND",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _to_small_bgr(pil_img, scale):
    w = max(1, int(pil_img.width  * scale))
    h = max(1, int(pil_img.height * scale))
    return np.array(pil_img.convert("RGB").resize((w, h)))[:, :, ::-1].copy()


def detect_desk(pil_img, scale):
    bgr = _to_small_bgr(pil_img, scale)
    dets = run_grounding_dino(bgr, DESK_PROMPT, box_threshold=0.25, max_area_ratio=0.98)
    if not dets:
        return None
    best = max(dets, key=lambda d: (d.box_xyxy[2]-d.box_xyxy[0])*(d.box_xyxy[3]-d.box_xyxy[1]))
    inv = 1.0 / scale
    return [int(v * inv) for v in best.box_xyxy]


def detect_objects(pil_img, scale, desk_bbox):
    dx1, dy1, dx2, dy2 = desk_bbox
    dw = max(1, dx2 - dx1)
    dh = max(1, dy2 - dy1)
    bgr  = _to_small_bgr(pil_img, scale)
    dets = run_grounding_dino(bgr, OBJ_PROMPT, box_threshold=0.28, max_area_ratio=0.80)
    objects = []
    for d in dets:
        cat = _DINO_TO_CAT.get(d.label.lower().strip())
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
        # 책상 bbox 완전 이탈은 제외 (약간 벗어나는 건 허용)
        if not (-0.15 <= rx <= 1.15 and -0.15 <= ry <= 1.15):
            continue
        objects.append({
            "category": cat,
            "bbox":     [ox1, oy1, ox2, oy2],
            "score":    round(float(d.score), 3),
            "rx":       round(float(rx), 4),
            "ry":       round(float(ry), 4),
            "rw":       round(float(rw), 4),
            "rh":       round(float(rh), 4),
        })
    return objects


def main():
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = lambda x, **kw: x

    processed: set = set()
    if CKPT_FILE.exists():
        processed = set(json.loads(CKPT_FILE.read_text(encoding="utf-8")))
        print(f"[resume] 이미 처리됨: {len(processed)}개")

    all_images = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.suffix.lower() in IMG_EXTS and str(p) not in processed
    )
    print(f"[build] 처리 대상: {len(all_images)}개")

    stats = {"total": 0, "desk_fail": 0, "no_objects": 0, "written": 0, "error": 0}

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

                objects = detect_objects(pil_img, scale, desk_bbox)
                if not objects:
                    stats["no_objects"] += 1
                    processed.add(str(img_path))
                    continue

                # style = raw/ 바로 아래 폴더명
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
    print(f"  → {OUT_JSONL}")


if __name__ == "__main__":
    main()
