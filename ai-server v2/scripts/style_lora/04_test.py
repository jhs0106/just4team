"""
style_lora 테스트 스크립트

학습된 LoRA를 SD v1.5 img2img에 적용하여 스타일 변환 결과 확인.
각 스타일별로 결과 이미지를 저장하고 denoising_strength 비교도 지원.

사용법:
  python scripts/style_lora/04_test.py --image data/test/desk_image.jpg
  python scripts/style_lora/04_test.py --image data/test/desk_image.jpg --style white
  python scripts/style_lora/04_test.py --image data/test/desk_image.jpg --compare-strength
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.parent
CFG_PATH   = ROOT / "configs" / "config.yaml"
LORA_DIR   = ROOT / "outputs" / "models" / "style_lora"
OUT_DIR    = ROOT / "outputs" / "test_style_lora"


def load_config() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 스타일별 테스트 프롬프트 ──────────────────────────────────────────────────
STYLE_PROMPTS = {
    "white":   "JU_Style, white style, clean minimal white desk setup, bright soft lighting, white peripherals",
    "black":   "JU_Style, black style, sleek dark desk setup, black accessories, moody ambient lighting",
    "gaming":  "JU_Style, gaming style, gaming desk setup with rgb lighting, dark background, colorful led",
    "cozy":    "JU_Style, cozy style, warm cozy desk setup, soft warm lighting, wooden accents, plants",
    "modern":  "JU_Style, modern style, modern minimal desk setup, clean lines, neutral tones, sleek design",
    "general": "JU_Style, general style, well-organized desk setup, balanced lighting, clean workspace",
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, text, person, face, "
    "hands, ugly, deformed, oversaturated, cartoon, anime, render"
)


# ── 파이프라인 로드 ───────────────────────────────────────────────────────────

def load_pipeline(lora_path: Path, base_model: str, device: str, dtype):
    print(f"  기반 모델 로드: {base_model}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.safety_checker = None

    print(f"  LoRA 로드: {lora_path}")
    # merge_adapter() 대신 PEFT 모델 그대로 유지 — diffusers 파이프라인과의 충돌 방지
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet,
        str(lora_path),
        torch_dtype=dtype,  # dtype 명시적 일치
    )
    pipe.unet.eval()

    pipe.vae.enable_slicing()
    return pipe.to(device)


def preprocess_image(img_path: Path, size: int = 512) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    img  = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


# ── 단일 스타일 테스트 ────────────────────────────────────────────────────────

def test_single_style(
    pipe,
    image: Image.Image,
    style: str,
    strength: float,
    out_dir: Path,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
):
    prompt = STYLE_PROMPTS.get(style, f"JU_Style, {style} style desk setup")
    print(f"  [{style}] strength={strength:.2f} 생성 중...")

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        strength=strength,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    out_path = out_dir / f"{style}_s{int(strength*100):03d}.png"
    result.save(out_path)
    return result, out_path


# ── denoising_strength 비교 ───────────────────────────────────────────────────

def make_comparison_grid(
    images: list[Image.Image],
    labels: list[str],
    out_path: Path,
    cell_size: int = 256,
):
    n  = len(images)
    w  = cell_size * n
    h  = cell_size + 30
    grid = Image.new("RGB", (w, h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    for i, (img, label) in enumerate(zip(images, labels)):
        thumb = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(thumb, (i * cell_size, 0))
        draw.text((i * cell_size + 5, cell_size + 5), label, fill=(220, 220, 220))

    grid.save(out_path)
    print(f"  비교 이미지 저장: {out_path.name}")


def test_strength_compare(
    pipe,
    image: Image.Image,
    style: str,
    out_dir: Path,
    strengths: list[float] = [0.2, 0.3, 0.4, 0.5, 0.6],
):
    images = [image]
    labels = ["original"]

    for s in tqdm(strengths, desc=f"{style} strength 비교"):
        result, _ = test_single_style(pipe, image, style, strength=s, out_dir=out_dir)
        images.append(result)
        labels.append(f"s={s:.1f}")

    make_comparison_grid(
        images, labels,
        out_dir / f"{style}_strength_compare.png",
    )


# ── 전체 스타일 비교 ──────────────────────────────────────────────────────────

def test_all_styles(
    pipe,
    image: Image.Image,
    styles: list[str],
    strength: float,
    out_dir: Path,
):
    images = [image]
    labels = ["original"]

    for style in tqdm(styles, desc="전체 스타일 테스트"):
        result, _ = test_single_style(pipe, image, style, strength=strength, out_dir=out_dir)
        images.append(result)
        labels.append(style)

    make_comparison_grid(
        images, labels,
        out_dir / f"all_styles_s{int(strength*100):03d}.png",
        cell_size=300,
    )


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="style_lora 결과 테스트")
    parser.add_argument("--image",           type=str,   default="data/test/desk_image.jpg", help="입력 책상 이미지 경로")
    parser.add_argument("--style",           type=str,   default=None,   help="특정 스타일만 테스트 (기본: 전체)")
    parser.add_argument("--strength",        type=float, default=0.4,    help="denoising strength (기본: 0.4)")
    parser.add_argument("--compare-strength",action="store_true",        help="strength 0.2~0.6 비교 그리드 생성")
    parser.add_argument("--lora-dir",        type=str,   default=None,   help="LoRA 경로 오버라이드")
    parser.add_argument("--steps",           type=int,   default=30,     help="inference steps")
    args = parser.parse_args()

    cfg        = load_config()
    sl_cfg     = cfg["style_lora"]
    base_model = sl_cfg["base_model"]
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    dtype      = torch.float16 if device == "cuda" else torch.float32

    # LoRA 경로 결정
    lora_base = Path(args.lora_dir) if args.lora_dir else LORA_DIR
    lora_path = lora_base / "style_lora_final" / "unet_lora"
    if not lora_path.exists():
        # 최신 체크포인트로 폴백
        ckpts = sorted(lora_base.glob("checkpoint-epoch*/unet_lora"))
        if not ckpts:
            raise FileNotFoundError(
                f"LoRA 가중치 없음: {lora_path}\n"
                "03_train.py를 먼저 실행하세요."
            )
        lora_path = ckpts[-1]
        print(f"  [폴백] 최신 체크포인트 사용: {lora_path.parent.parent.name}")

    # 이미지 로드
    img_path = ROOT / args.image
    if not img_path.exists():
        raise FileNotFoundError(f"이미지 없음: {img_path}")
    image = preprocess_image(img_path)

    # 출력 디렉토리
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = OUT_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / "original.png")
    print(f"\n출력 디렉토리: {out_dir}")

    # 파이프라인 로드
    print("\n파이프라인 로드 중...")
    pipe = load_pipeline(lora_path, base_model, device, dtype)

    # 테스트 실행
    styles = [args.style] if args.style else list(STYLE_PROMPTS.keys())

    if args.compare_strength:
        target_style = args.style or "white"
        print(f"\n[strength 비교] 스타일: {target_style}")
        test_strength_compare(pipe, image, target_style, out_dir)
    elif args.style:
        print(f"\n[단일 스타일] {args.style}")
        _, out_path = test_single_style(
            pipe, image, args.style,
            strength=args.strength,
            out_dir=out_dir,
            num_steps=args.steps,
        )
        print(f"  결과: {out_path}")
    else:
        print(f"\n[전체 스타일] strength={args.strength}")
        test_all_styles(pipe, image, styles, args.strength, out_dir)

    print(f"\n완료. 결과 위치: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
