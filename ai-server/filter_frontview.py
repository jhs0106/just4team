import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

RAW_DIR      = Path("data/style_lora/raw")
OUT_LIST     = Path("data/frontview_images.json")
THRESHOLD    = 0.40   # front-view 최소 softmax 확률
BATCH_SIZE   = 32
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp"}

PROMPTS = [
    "a desk setup photographed straight from the front",   # [0] front
    "a desk photographed from above or bird's eye view",   # [1] top
    "a desk photographed from the side",                   # [2] side
]


def load_clip():
    from transformers import CLIPProcessor, CLIPModel
    print("[CLIP] 모델 로드 중...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"[CLIP] 로드 완료 device={device}")
    return model, processor, device


def score_batch(model, processor, device, images: list) -> np.ndarray:
    inputs = processor(
        text=PROMPTS,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits_per_image   # (B, 3)
        probs   = logits.softmax(dim=-1).cpu().numpy()
    return probs   # shape (B, 3)


def main():
    all_images = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    )
    print(f"[filter] 총 {len(all_images)}장 탐색")

    model, processor, device = load_clip()

    results   = []
    front_ok  = []
    stats     = {"total": 0, "front": 0, "top": 0, "side": 0, "error": 0}

    for i in range(0, len(all_images), BATCH_SIZE):
        batch_paths = all_images[i:i + BATCH_SIZE]
        batch_imgs  = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
                valid_paths.append(p)
            except Exception:
                stats["error"] += 1

        if not batch_imgs:
            continue

        probs = score_batch(model, processor, device, batch_imgs)

        for path, prob in zip(valid_paths, probs):
            stats["total"] += 1
            front_prob = float(prob[0])
            pred_idx   = int(prob.argmax())
            label      = ["front", "top", "side"][pred_idx]
            stats[label] += 1

            results.append({
                "path":       str(path),
                "front_prob": round(front_prob, 4),
                "top_prob":   round(float(prob[1]), 4),
                "side_prob":  round(float(prob[2]), 4),
                "pred":       label,
            })

            if label == "front" and front_prob >= THRESHOLD:
                front_ok.append(str(path))

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  [{stats['total']}/{len(all_images)}] "
                  f"front={stats['front']} top={stats['top']} side={stats['side']}")

    OUT_LIST.parent.mkdir(parents=True, exist_ok=True)
    OUT_LIST.write_text(
        json.dumps({"front_view": front_ok, "all": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[done] total={stats['total']} error={stats['error']}")
    print(f"  front={stats['front']} top={stats['top']} side={stats['side']}")
    print(f"  threshold≥{THRESHOLD} 통과: {len(front_ok)}장")
    print(f"  → {OUT_LIST}")


if __name__ == "__main__":
    main()
