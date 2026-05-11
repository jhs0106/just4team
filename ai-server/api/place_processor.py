import cv2
import numpy as np
from PIL import Image


class PlaceProcessor:
    """
    상품 이미지를 마스크 영역에 픽셀 정확도로 합성.

    IP-Adapter-Plus(/inpaint)와 달리 상품 외형을 재생성하지 않음.
    cv2.seamlessClone(MIXED_CLONE)으로 경계 조명/색상만 자연스럽게 보정.
    GPU 불필요 — 순수 CPU OpenCV.
    """

    def place(
        self,
        background: Image.Image,
        mask: Image.Image,
        product: Image.Image,
    ) -> Image.Image:
        bg_rgb = np.array(background.convert("RGB"))
        mask_gray = np.array(mask.convert("L"))

        h_bg, w_bg = bg_rgb.shape[:2]

        if mask_gray.shape[:2] != (h_bg, w_bg):
            mask_gray = cv2.resize(mask_gray, (w_bg, h_bg), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(mask_gray > 128)
        if len(xs) == 0:
            return background

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w = max(x2 - x1 + 1, 1)
        h = max(y2 - y1 + 1, 1)

        product_rgb = np.array(product.convert("RGB"))
        product_resized = cv2.resize(product_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # seamlessClone은 소스 전체를 쓰므로 src_mask는 순백
        src_mask = np.full((h, w), 255, dtype=np.uint8)

        # center가 경계를 넘으면 clamp — 초과 시 OpenCV 에러 발생
        cx = max(w // 2 + 1, min((x1 + x2) // 2, w_bg - w // 2 - 2))
        cy = max(h // 2 + 1, min((y1 + y2) // 2, h_bg - h // 2 - 2))

        bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
        src_bgr = cv2.cvtColor(product_resized, cv2.COLOR_RGB2BGR)

        result_bgr = cv2.seamlessClone(
            src_bgr,
            bg_bgr,
            src_mask,
            (cx, cy),
            cv2.MIXED_CLONE,
        )

        return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))


_place_instance: PlaceProcessor | None = None


def get_place_processor() -> PlaceProcessor:
    global _place_instance
    if _place_instance is None:
        _place_instance = PlaceProcessor()
    return _place_instance
