import torch
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageFilter
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def b64_to_image(b64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


def image_to_b64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class Img2ImgProcessor:
    """
    SD v1.5 img2img + LoRA 스타일 변환.

    IP-Adapter 미사용 이유:
      img2img는 원본 이미지를 strength 비율로 노이징한 x_t에서 디노이징을 시작함.
      x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
      원본 정보 x_0가 시작 latent에 수학적으로 포함되어 있으므로
      IP-Adapter로 원본을 재주입하면 신호 충돌이 발생함.
    """

    def __init__(self):
        self.config = load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        proj_cfg = self.config.get("project", {})
        self.trigger_word = proj_cfg.get("trigger_word", "JU_DeskStyle")
        gen_cfg = self.config.get("generation", {})
        self.negative_prompt = gen_cfg.get(
            "negative_prompt",
            "blurry, low quality, distorted, watermark, text, person, face"
        )

        models_dir = Path(__file__).parent.parent / "outputs" / "models"
        lora_path = models_dir / "lora_final"
        if not lora_path.exists():
            checkpoints = sorted(models_dir.glob("lora_epoch_*"))
            lora_path = checkpoints[-1] if checkpoints else None
        self.lora_path = lora_path

        print("[Img2Img] 모델 로드 중...")
        self._load_pipeline()
        print("[Img2Img] 로드 완료.")

    def _load_pipeline(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=self.dtype,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipe.safety_checker = None

        if self.lora_path:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(self.lora_path))
            print(f"[Img2Img] LoRA 로드: {self.lora_path.name}")
        else:
            print("[Img2Img] LoRA 없음 — 기본 SD v1.5로 실행")

        pipe.vae.enable_slicing()

        self.pipe = pipe.to(self.device)

    def build_prompt(self, prompt: str, style: str | None) -> str:
        parts = [self.trigger_word]
        if style:
            parts.append(f"{style} style")
        parts.append(prompt)
        return ", ".join(parts)

    def process(
        self,
        image: Image.Image,
        prompt: str,
        style: str | None = None,
        strength: float = 0.7,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        full_prompt = self.build_prompt(prompt, style)
        image_512 = image.resize((512, 512)).convert("RGB")

        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=self.negative_prompt,
            image=image_512,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        return result.images[0]


_img2img_instance: Img2ImgProcessor | None = None


def get_img2img_processor() -> Img2ImgProcessor:
    global _img2img_instance
    if _img2img_instance is None:
        _img2img_instance = Img2ImgProcessor()
    return _img2img_instance
