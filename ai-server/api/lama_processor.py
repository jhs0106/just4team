import torch
from PIL import Image


class LamaProcessor:
    # LaMa 기반 물체 제거. 반복 텍스처(책상 표면 등) 복원에 특화, SD 대비 빠르고 가벼움

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[LaMa] 모델 로드 중...")
        from simple_lama_inpainting import SimpleLama
        self.model = SimpleLama()
        print("[LaMa] 로드 완료.")

    def inpaint(self, image: Image.Image, mask: Image.Image, use_fallback: bool = True) -> Image.Image:
        # image: PIL RGB, mask: PIL L (흰색=제거 영역). 실패 시 OpenCV fallback
        import cv2
        import numpy as np

        orig_size = image.size
        try:
            result = self.model(image.convert("RGB"), mask.convert("L"))
            if result.size != orig_size:
                result = result.resize(orig_size, Image.LANCZOS)
            return result.convert("RGB")
        except Exception as e:
            if not use_fallback:
                raise
            print(f"[LaMa] 실패 ({e}), OpenCV fallback 사용")
            img_bgr  = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            mask_arr = np.array(mask.convert("L"))
            result   = cv2.inpaint(img_bgr, mask_arr, 3, cv2.INPAINT_TELEA)
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


_lama_instance: LamaProcessor | None = None


def get_lama_processor() -> LamaProcessor:
    global _lama_instance
    if _lama_instance is None:
        _lama_instance = LamaProcessor()
    return _lama_instance
