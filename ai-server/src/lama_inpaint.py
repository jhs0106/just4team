from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def prepare_lama_mask(
    mask: Optional[np.ndarray],
    dilate_kernel: int = 5,
    dilate_iter: int = 1,
) -> Optional[np.ndarray]:
    """
    LaMa 입력용 remove mask 생성.

    mask를 너무 크게 키우면 주변까지 지워져 어색해질 수 있으므로
    기본 dilation은 작게 둔다.
    """
    if mask is None:
        return None

    mask = (mask > 0).astype(np.uint8) * 255

    if dilate_kernel > 0 and dilate_iter > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    return mask


@lru_cache(maxsize=1)
def get_lama_model():
    """
    SimpleLama 모델 로드.
    한 번만 로드하고 재사용한다.
    """
    from simple_lama_inpainting import SimpleLama

    print("[LaMa] Loading SimpleLama model...")
    return SimpleLama()


def run_lama_inpaint(
    image_bgr: np.ndarray,
    remove_mask: Optional[np.ndarray],
    use_fallback: bool = False,
) -> np.ndarray:
    """
    remove_mask 영역을 실제 LaMa로 인페인팅한다.
    """
    if remove_mask is None or np.count_nonzero(remove_mask) == 0:
        print("[LaMa] remove_mask가 비어 있어 원본 이미지를 반환합니다.")
        return image_bgr.copy()

    lama_mask = prepare_lama_mask(remove_mask)

    try:
        lama = get_lama_model()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        mask_pil = Image.fromarray(lama_mask).convert("L")

        print("[LaMa] Inpainting start")
        result_pil = lama(image_pil, mask_pil)
        print("[LaMa] Inpainting done")

        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr

    except Exception as e:
        print(f"[ERROR] LaMa 실행 실패: {e}")

        if use_fallback:
            print("[LaMa] OpenCV fallback inpaint 실행")
            return cv2.inpaint(image_bgr, lama_mask, 3, cv2.INPAINT_TELEA)

        raise


def save_lama_inputs(
    out_dir: Path,
    image_bgr: np.ndarray,
    remove_mask: np.ndarray,
):
    """LaMa 디버깅용 입력 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "lama_input_image.png"), image_bgr)
    cv2.imwrite(str(out_dir / "lama_remove_mask.png"), remove_mask)