import torch
import base64
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor


def b64_to_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(BytesIO(data)).convert("RGB")


def image_to_b64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class SAM2Processor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[SAM-2] 모델 로드 중...")
        # from_pretrained 사용 — config 경로 문제 없이 HuggingFace에서 직접 로드
        # hiera_small: 프로토타입 기준 속도/품질 균형
        self.predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-large",
            device=self.device,
        )
        print("[SAM-2] 로드 완료.")

    def segment(
        self,
        image: Image.Image,
        points: list[list[int]],
        point_labels: list[int],
        max_size: int = 1024,
    ) -> dict:
        orig_w, orig_h = image.size

        # SAM-2 입력 크기 제한 — 큰 이미지는 리사이즈 후 포인트 좌표 스케일링
        if orig_w > max_size or orig_h > max_size:
            scale = min(max_size / orig_w, max_size / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            points = [[int(x * scale), int(y * scale)] for x, y in points]

        img_array = np.array(image.convert("RGB"))
        self.predictor.set_image(img_array)

        input_points = np.array(points, dtype=np.float32)
        input_labels = np.array(point_labels, dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        # 가장 높은 score의 마스크 선택
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)  # 명시적으로 bool 변환

        # 마스크 이미지 생성
        mask_img = Image.fromarray((best_mask * 255).astype(np.uint8), mode="L")

        # 오버레이 생성 (원본 + 반투명 마스크)
        overlay = image.copy().convert("RGBA")
        mask_colored = np.zeros((*best_mask.shape, 4), dtype=np.uint8)
        mask_colored[best_mask] = [255, 0, 0, 120]  # 빨간색 반투명
        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_colored))

        return {
            "mask": image_to_b64(mask_img),
            "overlay": image_to_b64(overlay.convert("RGB")),
        }


# 싱글톤
_sam2_instance: SAM2Processor | None = None

def get_sam2_processor() -> SAM2Processor:
    global _sam2_instance
    if _sam2_instance is None:
        _sam2_instance = SAM2Processor()
    return _sam2_instance