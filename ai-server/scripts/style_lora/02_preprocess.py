"""
style_lora 학습 데이터 전처리 파이프라인

[Stage 1] CPU 기반 기초 필터링
  - 파일 무결성 / 해상도 / 종횡비 / 블러 / 밝기 / 색상 엔트로피
  - pHash 기반 중복 제거

[Stage 2] CLIP 기반 콘텐츠 필터링
  - 책상 셋업 이미지인지 확인
  - 렌더링 / 아트워크 / 인물 사진 제거

[Stage 3] BLIP-2 캡셔닝
  - 스타일 중심 4문항 질문 → 텍스트 캡션 생성
  - 책상 키워드 최소 2개 이상 포함 검증

[Stage 4] 최종 저장
  - 512×512 center-crop → PNG
  - 트리거워드 포함 캡션 → .txt

사용법:
  python scripts/style_lora/02_preprocess.py
  python scripts/style_lora/02_preprocess.py --style white
  python scripts/style_lora/02_preprocess.py --style white --no-blip  # BLIP-2 스킵 (빠른 테스트)
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter, UnidentifiedImageError
from scipy.stats import entropy
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
)

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent          # ai-server/
CFG_PATH  = ROOT / "configs" / "config.yaml"
RAW_DIR   = ROOT / "data" / "style_lora" / "raw"
OUT_DIR   = ROOT / "data" / "style_lora" / "processed"
LOG_DIR   = ROOT / "logs"
STATE_FILE = ROOT / "data" / "style_lora" / "preprocess_state.json"  # 재개용


# ── CLIP 판별 프롬프트 ─────────────────────────────────────────────────────────
DESK_PROMPTS = [
    "a desk setup with monitor and keyboard",
    "a battlestation setup with computer peripherals",
    "a home office desk with computer",
    "a gaming setup with rgb lighting",
    "a clean minimal desk workspace",
]

NON_DESK_PROMPTS = [
    "a 3d render or digital art",
    "a person portrait or selfie",
    "outdoor scenery or landscape",
    "food or product photo",
    "anime or cartoon artwork",
    "a living room or bedroom without a desk",
]

# ── BLIP-2 질문 세트 ──────────────────────────────────────────────────────────
BLIP_QUESTIONS = [
    "What is the color theme of this desk setup?",
    "Describe the lighting mood and atmosphere of this desk.",
    "What design style does this desk setup follow?",
    "List the main peripherals and items visible on this desk.",
]

DESK_KEYWORDS = {
    "desk", "monitor", "keyboard", "mouse", "setup", "battlestation",
    "speaker", "lamp", "chair", "headset", "mousepad", "cable",
    "white", "black", "wooden", "glass", "minimal", "gaming", "cozy",
    "nordic", "retro", "industrial", "modern", "rgb", "led",
}

# ── 로드 ──────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ── Stage 1: CPU 필터 ─────────────────────────────────────────────────────────

def check_file_valid(path: Path) -> tuple[bool, str]:
    try:
        img = Image.open(path)
        img.verify()
        return True, ""
    except Exception as e:
        return False, f"파일 손상: {e}"


def load_image_safe(path: Path) -> tuple[Image.Image | None, str]:
    try:
        img = Image.open(path).convert("RGB")
        return img, ""
    except UnidentifiedImageError:
        return None, "인식 불가 이미지 포맷"
    except Exception as e:
        return None, f"로드 실패: {e}"


def check_resolution(img: Image.Image, min_res: int) -> tuple[bool, str]:
    w, h = img.size
    if w < min_res or h < min_res:
        return False, f"해상도 부족 ({w}×{h} < {min_res})"
    return True, ""


def check_aspect_ratio(img: Image.Image, max_ratio: float) -> tuple[bool, str]:
    w, h = img.size
    ratio = max(w, h) / min(w, h)
    if ratio > max_ratio:
        return False, f"종횡비 초과 ({ratio:.2f} > {max_ratio})"
    return True, ""


def check_blur(img: Image.Image, threshold: float) -> tuple[bool, str]:
    gray = np.array(img.convert("L"))
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < threshold:
        return False, f"블러 ({lap_var:.1f} < {threshold})"
    return True, ""


def check_brightness(img: Image.Image, mn: int, mx: int) -> tuple[bool, str]:
    mean_brightness = np.array(img.convert("L")).mean()
    if mean_brightness < mn:
        return False, f"너무 어두움 ({mean_brightness:.1f} < {mn})"
    if mean_brightness > mx:
        return False, f"너무 밝음 ({mean_brightness:.1f} > {mx})"
    return True, ""


def check_color_entropy(img: Image.Image, min_entropy: float) -> tuple[bool, str]:
    arr = np.array(img.convert("L"))
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    hist = hist[hist > 0].astype(float)
    ent = entropy(hist / hist.sum(), base=2)
    if ent < min_entropy:
        return False, f"색상 다양성 부족 (entropy {ent:.2f} < {min_entropy})"
    return True, ""


def compute_phash(img: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(img, hash_size=16)


def run_stage1(
    paths: list[Path],
    cfg_pre: dict,
    seen_hashes: dict[str, str],
) -> tuple[list[tuple[Path, Image.Image]], dict]:
    """CPU 기초 필터 + pHash 중복 제거."""
    passed: list[tuple[Path, Image.Image]] = []
    stats = {
        "total":      len(paths),
        "corrupt":    0,
        "resolution": 0,
        "aspect":     0,
        "blur":       0,
        "brightness": 0,
        "entropy":    0,
        "duplicate":  0,
        "passed":     0,
    }
    min_res    = cfg_pre["min_resolution"]
    max_aspect = cfg_pre["max_aspect_ratio"]
    blur_thr   = cfg_pre["blur_threshold"]
    min_br     = cfg_pre["min_brightness"]
    max_br     = cfg_pre["max_brightness"]
    min_ent    = cfg_pre["min_color_entropy"]
    phash_dist = cfg_pre["phash_distance"]

    for path in tqdm(paths, desc="Stage1 CPU필터", leave=False):
        # 파일 무결성
        ok, reason = check_file_valid(path)
        if not ok:
            stats["corrupt"] += 1
            continue

        img, reason = load_image_safe(path)
        if img is None:
            stats["corrupt"] += 1
            continue

        # 해상도
        ok, reason = check_resolution(img, min_res)
        if not ok:
            stats["resolution"] += 1
            continue

        # 종횡비
        ok, reason = check_aspect_ratio(img, max_aspect)
        if not ok:
            stats["aspect"] += 1
            continue

        # 블러
        ok, reason = check_blur(img, blur_thr)
        if not ok:
            stats["blur"] += 1
            continue

        # 밝기
        ok, reason = check_brightness(img, min_br, max_br)
        if not ok:
            stats["brightness"] += 1
            continue

        # 색상 엔트로피
        ok, reason = check_color_entropy(img, min_ent)
        if not ok:
            stats["entropy"] += 1
            continue

        # pHash 중복 검사
        ph = compute_phash(img)
        is_dup = False
        for seen_hash_str in seen_hashes:
            seen_ph = imagehash.hex_to_hash(seen_hash_str)
            if ph - seen_ph <= phash_dist:
                is_dup = True
                break
        if is_dup:
            stats["duplicate"] += 1
            continue

        seen_hashes[str(ph)] = str(path)
        passed.append((path, img))

    stats["passed"] = len(passed)
    return passed, stats


# ── Stage 2: CLIP 필터 ────────────────────────────────────────────────────────

def load_clip(device: str):
    model_id = "openai/clip-vit-base-patch32"
    print(f"  CLIP 로드: {model_id}")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def clip_score_batch(
    images: list[Image.Image],
    prompts: list[str],
    model,
    processor,
    device: str,
) -> np.ndarray:
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    outputs = model(**inputs)
    logits = outputs.logits_per_image  # (n_images, n_prompts)
    probs = logits.softmax(dim=-1).cpu().numpy()
    return probs


def run_stage2(
    items: list[tuple[Path, Image.Image]],
    cfg_pre: dict,
    device: str,
    batch_size: int = 8,
) -> tuple[list[tuple[Path, Image.Image]], dict]:
    """CLIP 기반 콘텐츠 필터링."""
    if not items:
        return items, {"total": 0, "rejected_non_desk": 0, "rejected_render": 0, "passed": 0}

    model, processor = load_clip(device)
    desk_thr     = cfg_pre["clip_desk_threshold"]
    non_desk_thr = cfg_pre["clip_non_desk_threshold"]

    all_prompts    = DESK_PROMPTS + NON_DESK_PROMPTS
    n_desk         = len(DESK_PROMPTS)

    passed: list[tuple[Path, Image.Image]] = []
    stats = {"total": len(items), "rejected_non_desk": 0, "rejected_render": 0, "passed": 0}

    for i in tqdm(range(0, len(items), batch_size), desc="Stage2 CLIP", leave=False):
        batch = items[i : i + batch_size]
        imgs  = [img for _, img in batch]

        probs = clip_score_batch(imgs, all_prompts, model, processor, device)

        for j, (path, img) in enumerate(batch):
            desk_score     = probs[j, :n_desk].max()
            non_desk_score = probs[j, n_desk:].max()

            if desk_score < desk_thr:
                stats["rejected_non_desk"] += 1
                continue
            if non_desk_score > non_desk_thr:
                stats["rejected_render"] += 1
                continue

            passed.append((path, img))

    stats["passed"] = len(passed)

    # GPU 해제
    del model
    torch.cuda.empty_cache()
    return passed, stats


# ── Stage 3: BLIP-2 캡셔닝 ───────────────────────────────────────────────────

def load_blip2(device: str):
    model_id = "Salesforce/blip2-opt-2.7b"
    print(f"  BLIP-2 로드: {model_id}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def blip2_answer(
    img: Image.Image,
    question: str,
    model,
    processor,
    device: str,
    max_new_tokens: int = 60,
) -> str:
    dtype = next(model.parameters()).dtype
    inputs = processor(images=img, text=question, return_tensors="pt").to(device, dtype)
    ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()


def build_caption(answers: list[str]) -> str:
    return ". ".join(a.strip(".").strip() for a in answers if a)


def has_enough_keywords(caption: str, min_kw: int = 2) -> bool:
    words = set(caption.lower().split())
    return len(words & DESK_KEYWORDS) >= min_kw


def run_stage3(
    items: list[tuple[Path, Image.Image]],
    style_name: str,
    trigger_word: str,
    device: str,
    use_blip: bool = True,
) -> tuple[list[tuple[Path, Image.Image, str]], dict]:
    """BLIP-2 캡셔닝 + 키워드 검증."""
    stats = {"total": len(items), "rejected_no_keywords": 0, "passed": 0}

    if not use_blip:
        # BLIP 스킵 시 기본 캡션 사용
        result = [
            (path, img, f"{trigger_word}, {style_name} style desk setup")
            for path, img in items
        ]
        stats["passed"] = len(result)
        return result, stats

    model, processor = load_blip2(device)
    result: list[tuple[Path, Image.Image, str]] = []

    for path, img in tqdm(items, desc="Stage3 BLIP-2", leave=False):
        answers = []
        for q in BLIP_QUESTIONS:
            try:
                ans = blip2_answer(img, q, model, processor, device)
                answers.append(ans)
            except Exception:
                answers.append("")

        raw_caption = build_caption(answers)

        if not has_enough_keywords(raw_caption, min_kw=2):
            stats["rejected_no_keywords"] += 1
            continue

        final_caption = f"{trigger_word}, {style_name} style, {raw_caption}"
        result.append((path, img, final_caption))

    stats["passed"] = len(result)

    del model
    torch.cuda.empty_cache()
    return result, stats


# ── Stage 4: 저장 ────────────────────────────────────────────────────────────

def center_crop_512(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left   = (w - side) // 2
    top    = (h - side) // 2
    right  = left + side
    bottom = top  + side
    return img.crop((left, top, right, bottom)).resize((512, 512), Image.LANCZOS)


def run_stage4(
    items: list[tuple[Path, Image.Image, str]],
    out_dir: Path,
    style_name: str,
    processed_set: set[str],
) -> dict:
    """최종 저장: center-crop 512×512 PNG + 캡션 txt."""
    style_dir = out_dir / style_name
    style_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(items), "saved": 0, "skipped_existing": 0}

    for path, img, caption in tqdm(items, desc="Stage4 저장", leave=False):
        stem = path.stem
        out_img  = style_dir / f"{stem}.png"
        out_txt  = style_dir / f"{stem}.txt"

        if str(out_img) in processed_set:
            stats["skipped_existing"] += 1
            continue

        try:
            cropped = center_crop_512(img)
            cropped.save(out_img, format="PNG", optimize=True)
            out_txt.write_text(caption, encoding="utf-8")
            processed_set.add(str(out_img))
            stats["saved"] += 1
        except Exception as e:
            print(f"\n  저장 실패 {path.name}: {e}")

    return stats


# ── 보고서 ────────────────────────────────────────────────────────────────────

def print_report(style: str, s1: dict, s2: dict, s3: dict, s4: dict, elapsed: float):
    total_in   = s1["total"]
    total_out  = s4["saved"] + s4["skipped_existing"]
    reject_tot = total_in - total_out

    print(f"\n{'='*55}")
    print(f"  [{style.upper()}] 전처리 완료  ({elapsed:.1f}s)")
    print(f"{'='*55}")
    print(f"  입력 이미지       : {total_in:>5}")
    print(f"  ── Stage1 CPU 필터")
    print(f"     파일 손상/포맷  : {s1['corrupt']:>5}")
    print(f"     해상도 부족     : {s1['resolution']:>5}")
    print(f"     종횡비 초과     : {s1['aspect']:>5}")
    print(f"     블러            : {s1['blur']:>5}")
    print(f"     밝기 이상       : {s1['brightness']:>5}")
    print(f"     색 다양성 부족  : {s1['entropy']:>5}")
    print(f"     중복(pHash)     : {s1['duplicate']:>5}")
    print(f"     → 통과          : {s1['passed']:>5}")
    print(f"  ── Stage2 CLIP 필터")
    print(f"     비책상 이미지   : {s2['rejected_non_desk']:>5}")
    print(f"     렌더링/아트     : {s2['rejected_render']:>5}")
    print(f"     → 통과          : {s2['passed']:>5}")
    print(f"  ── Stage3 BLIP-2")
    print(f"     키워드 부족     : {s3['rejected_no_keywords']:>5}")
    print(f"     → 통과          : {s3['passed']:>5}")
    print(f"  ── Stage4 저장")
    print(f"     새로 저장       : {s4['saved']:>5}")
    print(f"     이미 있음(재개) : {s4['skipped_existing']:>5}")
    print(f"{'='*55}")
    pct = total_out / total_in * 100 if total_in else 0
    print(f"  최종 학습 데이터  : {total_out:>5}장  ({pct:.1f}%)\n")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def get_all_image_paths(style_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(
        p for p in style_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def process_style(
    style_name: str,
    cfg: dict,
    device: str,
    use_blip: bool,
    seen_hashes: dict,
    processed_set: set,
) -> dict:
    cfg_pre     = cfg["style_lora"]["preprocessing"]
    trigger_word = cfg["style_lora"]["trigger_word"]

    raw_style_dir = RAW_DIR / style_name
    if not raw_style_dir.exists():
        print(f"\n[SKIP] {style_name}: raw 폴더 없음")
        return {}

    paths = get_all_image_paths(raw_style_dir)
    if not paths:
        print(f"\n[SKIP] {style_name}: 이미지 없음")
        return {}

    print(f"\n{'─'*55}")
    print(f"  처리 시작: {style_name} ({len(paths)}장)")
    t0 = time.time()

    # Stage 1
    passed1, s1 = run_stage1(paths, cfg_pre, seen_hashes)

    # Stage 2
    passed2, s2 = run_stage2(passed1, cfg_pre, device)

    # Stage 3
    passed3, s3 = run_stage3(passed2, style_name, trigger_word, device, use_blip)

    # Stage 4
    s4 = run_stage4(passed3, OUT_DIR, style_name, processed_set)

    elapsed = time.time() - t0
    print_report(style_name, s1, s2, s3, s4, elapsed)
    return {"s1": s1, "s2": s2, "s3": s3, "s4": s4}


def main():
    parser = argparse.ArgumentParser(description="style_lora 전처리 파이프라인")
    parser.add_argument("--style",   type=str, default=None, help="특정 스타일만 처리 (기본: 전체)")
    parser.add_argument("--no-blip", action="store_true",    help="BLIP-2 캡셔닝 스킵 (빠른 테스트)")
    args = parser.parse_args()

    cfg    = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")
    print(f"Raw 디렉토리: {RAW_DIR}")
    print(f"출력 디렉토리: {OUT_DIR}")

    # 재개용 상태 로드
    state = load_state()
    seen_hashes: dict[str, str] = state.get("seen_hashes", {})
    processed_set: set[str]     = set(state.get("processed_files", []))

    LOG_DIR.mkdir(exist_ok=True)

    styles = [args.style] if args.style else sorted(
        d.name for d in RAW_DIR.iterdir() if d.is_dir()
    )

    all_stats = {}
    for style_name in styles:
        try:
            result = process_style(
                style_name, cfg, device,
                use_blip=not args.no_blip,
                seen_hashes=seen_hashes,
                processed_set=processed_set,
            )
            if result:
                all_stats[style_name] = result
        except KeyboardInterrupt:
            print("\n\n중단됨 — 진행 상태 저장 중...")
            break
        except Exception as e:
            print(f"\n[ERROR] {style_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            save_state({
                "seen_hashes":     seen_hashes,
                "processed_files": list(processed_set),
            })

    # 전체 요약
    if all_stats:
        total_in  = sum(v["s1"]["total"]  for v in all_stats.values())
        total_out = sum(v["s4"]["saved"] + v["s4"]["skipped_existing"] for v in all_stats.values())
        print(f"\n{'='*55}")
        print(f"  전체 완료: {total_out}/{total_in}장 → {OUT_DIR}")
        print(f"{'='*55}\n")

    save_state({
        "seen_hashes":     seen_hashes,
        "processed_files": list(processed_set),
    })


if __name__ == "__main__":
    main()
