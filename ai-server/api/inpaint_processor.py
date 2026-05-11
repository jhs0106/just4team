import torch
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
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


class InpaintProcessor:
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

        print("[Inpaint] 모델 로드 중...")
        self._load_pipeline()
        print("[Inpaint] 로드 완료.")

    def _load_pipeline(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
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
            print(f"[Inpaint] LoRA 로드: {self.lora_path.name}")

        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus_sd15.bin",
        )
        pipe.set_ip_adapter_scale(0.8)
        pipe.vae.enable_slicing()

        self.pipe = pipe.to(self.device)

    def build_prompt(self, prompt: str, style: str | None) -> str:
        parts = [self.trigger_word]
        if style:
            parts.append(f"{style} style")
        parts.append(prompt)
        return ", ".join(parts)

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        product_image: Image.Image,
        prompt: str,
        style: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        ip_adapter_scale: float = 0.8,
    ) -> Image.Image:
        full_prompt = self.build_prompt(prompt, style)

        image_512 = image.resize((512, 512)).convert("RGB")
        mask_512 = mask.resize((512, 512)).convert("L")
        product_512 = product_image.resize((512, 512)).convert("RGB")
        mask_feathered = mask_512.filter(ImageFilter.GaussianBlur(radius=4))

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=self.negative_prompt,
            image=image_512,
            mask_image=mask_feathered,
            ip_adapter_image=product_512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        return result.images[0]


_inpaint_instance: InpaintProcessor | None = None


def get_inpaint_processor() -> InpaintProcessor:
    global _inpaint_instance
    if _inpaint_instance is None:
        _inpaint_instance = InpaintProcessor()
    return _inpaint_instance
