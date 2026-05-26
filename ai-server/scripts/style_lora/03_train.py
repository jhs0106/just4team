"""
style_lora 학습 스크립트

SD v1.5 UNet attention layers에 LoRA 적용.
6개 스타일(general/black/cozy/white/modern/gaming) 동시 학습.
캡션에 트리거워드 + 스타일명이 포함되어 스타일 구분.

사용법:
  python scripts/style_lora/03_train.py
  python scripts/style_lora/03_train.py --resume          # 마지막 체크포인트에서 재개
  python scripts/style_lora/03_train.py --epochs 100      # 에폭 수 오버라이드
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent
CFG_PATH  = ROOT / "configs" / "config.yaml"
DATA_DIR  = ROOT / "data" / "style_lora" / "processed"
LOG_DIR   = ROOT / "logs"


def load_config() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 데이터셋 ──────────────────────────────────────────────────────────────────

class StyleDataset(Dataset):
    def __init__(self, data_dir: Path, styles: list[str], tokenizer, size: int = 512):
        self.items: list[tuple[Path, str]] = []
        for style in styles:
            style_dir = data_dir / style
            if not style_dir.exists():
                print(f"  [경고] processed/{style} 없음, 스킵")
                continue
            for img_path in sorted(style_dir.glob("*.png")):
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    caption = txt_path.read_text(encoding="utf-8").strip()
                    self.items.append((img_path, caption))

        if not self.items:
            raise ValueError("학습 데이터 없음. 02_preprocess.py를 먼저 실행하세요.")

        random.shuffle(self.items)
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, caption = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)

        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": input_ids}


# ── 체크포인트 유틸 ────────────────────────────────────────────────────────────

def save_checkpoint(unet, optimizer, scaler, epoch: int, out_dir: Path):
    ckpt_dir = out_dir / f"checkpoint-epoch{epoch:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(ckpt_dir / "unet_lora")
    torch.save({
        "epoch":     epoch,
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
    }, ckpt_dir / "train_state.pt")
    print(f"  체크포인트 저장: {ckpt_dir.name}")


def load_latest_checkpoint(out_dir: Path) -> tuple[Path | None, int]:
    ckpts = sorted(out_dir.glob("checkpoint-epoch*"))
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    epoch = int(latest.name.split("epoch")[1])
    return latest, epoch


# ── 학습 ──────────────────────────────────────────────────────────────────────

def train(args):
    cfg      = load_config()
    sl_cfg   = cfg["style_lora"]
    tr_cfg   = sl_cfg["training"]
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    dtype    = torch.float16 if tr_cfg["mixed_precision"] == "fp16" and device == "cuda" else torch.float32

    out_dir  = ROOT / tr_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    base_model   = sl_cfg["base_model"]
    train_styles = sl_cfg["train_styles"]
    num_epochs   = args.epochs or tr_cfg["num_train_epochs"]
    lr           = tr_cfg["learning_rate"]
    batch_size   = tr_cfg["train_batch_size"]
    grad_accum   = tr_cfg["gradient_accumulation_steps"]
    warmup_steps = tr_cfg["warmup_steps"]
    save_every   = tr_cfg["save_every_n_epochs"]
    lora_rank    = tr_cfg["lora_rank"]
    lora_alpha   = tr_cfg["lora_alpha"]

    print(f"디바이스: {device}")
    print(f"학습 스타일: {train_styles}")
    print(f"에폭: {num_epochs}  배치: {batch_size}  누적: {grad_accum}")
    print(f"LoRA rank={lora_rank}, alpha={lora_alpha}")

    # ── 모델 로드 ───────────────────────────────────────────────────────────
    print("\n모델 로드 중...")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_enc  = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device)
    vae       = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device)
    unet      = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_sch = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # 그라디언트 불필요한 모델 고정
    text_enc.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # ── LoRA 설정 ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.to(device)
    unet.print_trainable_parameters()

    # ── 재개 체크포인트 ──────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt_path, start_epoch = load_latest_checkpoint(out_dir)
        if ckpt_path:
            from peft import PeftModel
            unet = PeftModel.from_pretrained(unet.base_model.model, ckpt_path / "unet_lora")
            unet = get_peft_model(unet, lora_config)
            unet.to(device)
            print(f"  체크포인트 재개: epoch {start_epoch}")
        else:
            print("  체크포인트 없음. 처음부터 시작.")

    # ── 데이터셋 ─────────────────────────────────────────────────────────────
    dataset    = StyleDataset(DATA_DIR, train_styles, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    print(f"\n학습 데이터: {len(dataset)}장")

    # ── 옵티마이저 & 스케줄러 ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=lr,
        weight_decay=tr_cfg["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = math.ceil(len(dataloader) / grad_accum) * num_epochs
    lr_scheduler = get_scheduler(
        tr_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=(device == "cuda" and dtype == torch.float16))

    if args.resume and ckpt_path:
        state = torch.load(ckpt_path / "train_state.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    print(f"\n학습 시작 (epoch {start_epoch+1} ~ {num_epochs})")
    t0 = time.time()

    vae.eval()
    text_enc.eval()

    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids    = batch["input_ids"].to(device)

            with torch.no_grad():
                latents      = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                encoder_out  = text_enc(input_ids)[0]

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_sch.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_lat = noise_sch.add_noise(latents, noise, timesteps)

            with autocast(enabled=(device == "cuda" and dtype == torch.float16)):
                pred  = unet(noisy_lat, timesteps, encoder_out).sample
                loss  = F.mse_loss(pred.float(), noise.float(), reduction="mean")
                loss  = loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        avg_loss = epoch_loss / len(dataloader)
        elapsed  = (time.time() - t0) / 60
        print(f"  Epoch {epoch+1:>4}/{num_epochs}  loss={avg_loss:.4f}  ({elapsed:.1f}min)")

        # 체크포인트 저장
        if (epoch + 1) % save_every == 0:
            save_checkpoint(unet, optimizer, scaler, epoch + 1, out_dir)

    # 최종 저장
    final_dir = out_dir / "style_lora_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(final_dir / "unet_lora")
    print(f"\n최종 LoRA 저장: {final_dir}")
    print(f"총 학습 시간: {(time.time()-t0)/60:.1f}분")


def main():
    parser = argparse.ArgumentParser(description="style_lora 학습")
    parser.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 재개")
    parser.add_argument("--epochs", type=int, default=None, help="에폭 수 오버라이드")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
