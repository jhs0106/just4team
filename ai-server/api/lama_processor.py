import torch
from PIL import Image


class LamaProcessor:
    """
    LaMa (Large Mask inpainting) 기반 물체 제거 프로세서.
    SAM-2 Auto로 감지한 넓은 마스크 영역을 채우는 데 특화.
    SD Inpainting 대비: 빠르고 가볍고, 반복 텍스처(책상 표면 등) 복원에 강함.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[LaMa] 모델 로드 중...")
        from simple_lama_inpainting import SimpleLama
        self.model = SimpleLama()
        print("[LaMa] 로드 완료.")

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Parameters
        ----------
        image : PIL RGB — 원본 이미지
        mask  : PIL L   — 흰색=제거 영역, 검정=유지 영역

        Returns
        -------
        PIL RGB — 물체가 제거된 이미지 (원본 해상도 유지)
        """
        orig_size = image.size
        result = self.model(image.convert("RGB"), mask.convert("L"))
        if result.size != orig_size:
            result = result.resize(orig_size, Image.LANCZOS)
        return result.convert("RGB")


_lama_instance: LamaProcessor | None = None


def get_lama_processor() -> LamaProcessor:
    global _lama_instance
    if _lama_instance is None:
        _lama_instance = LamaProcessor()
    return _lama_instance
