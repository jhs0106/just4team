import os
import base64
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional


def _guess_mime(b64: str) -> str:
    # base64 첫 바이트로 PNG/JPEG 판별. 잘못된 mime로 업로드하면 OpenAI가 거부할 수 있음.
    if b64.startswith("/9j/"):       # JPEG magic (0xFFD8)
        return "image/jpeg"
    return "image/png"               # PNG(iVBOR...) 및 기본값


def _b64_to_bytes(b64: str) -> bytes:
    # data URI 접두("data:image/png;base64,")가 붙어오면 제거 후 디코드
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


class NanoBananaProcessor:
    def __init__(self):
        # NanoBanana(Gemini) → OpenAI 이미지 편집(gpt-image-1)로 전환.
        # 클래스/함수명은 호환 위해 유지하되 내부는 OpenAI /v1/images/edits 호출.
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_url = os.getenv("OPENAI_IMAGE_URL", "https://api.openai.com/v1/images/edits")
        self.model   = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        self.size    = os.getenv("OPENAI_IMAGE_SIZE", "auto")
        self.timeout = int(os.getenv("OPENAI_TIMEOUT", "180"))

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    def _headers(self) -> Dict[str, str]:
        # multipart/form-data라 Content-Type은 requests가 boundary 포함해 자동 설정 → 인증만.
        return {"Authorization": f"Bearer {self.api_key}"}

    def _read_product_image_base64(self, image_id: Any) -> Optional[str]:
        if image_id is None:
            return None

        candidates = [
            Path("data/test/processed_images") / f"{image_id}.png",
            Path("ai-server/data/test/processed_images") / f"{image_id}.png",
            ]

        for path in candidates:
            if path.exists():
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

        return None

    def build_prompt(
            self,
            theme: str,
            mode: str,
            products: List[Dict[str, Any]],
            space_constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        product_lines = []

        for idx, product in enumerate(products, start=1):
            name = product.get("name") or "recommended product"
            category = product.get("category") or ""
            brand = product.get("brand") or ""
            price = product.get("price") or ""

            line = f"{idx}. {name}"
            if category:
                line += f" / category: {category}"
            if brand:
                line += f" / brand: {brand}"
            if price:
                line += f" / price: {price}"

            product_lines.append(line)

        product_text = "\n".join(product_lines) if product_lines else "No product information."

        space_text = "No explicit available-space constraints."
        if space_constraints:
            space_text = str(space_constraints)

        return f"""
You are a professional deskterior designer.

Create a realistic deskterior simulation using the user's original desk photo and the recommended products.

Deskterior style theme:
{theme}

User desk mode:
{mode}

Recommended products:
{product_text}

Available-space information:
{space_text}

Important generation rules:
- Preserve the original desk, camera angle, perspective, and room structure.
- Keep the image looking like a real photograph.
- Use the recommended products as the main items to add.
- Place the products naturally on the desk.
- Do not overcrowd the desk.
- Keep product scale realistic.
- Do not place products floating in the air.
- Do not completely change the user's room or desk.
- If there is not enough space for all products, prioritize the most suitable products.
- The final result should look like a realistic deskterior photo based on the user's actual desk.
""".strip()

    def build_products_payload(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload_products = []

        for product in products:
            image_id = product.get("image_id") or product.get("id")

            payload_products.append({
                "id": image_id,
                "name": product.get("name"),
                "category": product.get("category"),
                "price": product.get("price"),
                "brand": product.get("brand"),
                "image_url": product.get("image_url"),
                "product_url": product.get("product_url"),
                "image_base64": self._read_product_image_base64(image_id),
            })

        return payload_products

    def generate(
            self,
            image_base64: str,
            theme: str,
            mode: str,
            products: List[Dict[str, Any]],
            space_constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(
            theme=theme,
            mode=mode,
            products=products,
            space_constraints=space_constraints,
        )

        # OpenAI gpt-image-1 이미지 편집: 멀티파트로 [책상 사진] + [제품 이미지들] 업로드.
        # image[]: 첫 번째가 편집 베이스(책상), 이후는 합성 참조(제품). gpt-image-1은 다중 입력 지원.
        desk_mime = _guess_mime(image_base64)
        desk_ext = "jpg" if desk_mime == "image/jpeg" else "png"
        files = [
            ("image[]", (f"desk.{desk_ext}", _b64_to_bytes(image_base64), desk_mime)),
        ]
        for p in products:
            img_b64 = self._read_product_image_base64(p.get("image_id") or p.get("id"))
            if img_b64:
                pid = p.get("image_id") or p.get("id")
                files.append(("image[]", (f"product_{pid}.png", _b64_to_bytes(img_b64), "image/png")))

        form = {
            "model": self.model,
            "prompt": prompt,
            "size": self.size,
        }

        response = requests.post(
            self.api_url,
            headers=self._headers(),
            data=form,
            files=files,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RuntimeError(f"OpenAI image API error {response.status_code}: {response.text}")

        data = response.json()

        items = data.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise RuntimeError(
                f"OpenAI 응답에 이미지 없음 (keys={list(data.keys())}, body={response.text[:500]})"
            )

        return {
            "result_image_base64": items[0]["b64_json"],
            "prompt": prompt,
            "raw_response": data,
        }


_nanobanana_processor: Optional[NanoBananaProcessor] = None


def get_nanobanana_processor() -> NanoBananaProcessor:
    global _nanobanana_processor

    if _nanobanana_processor is None:
        _nanobanana_processor = NanoBananaProcessor()

    return _nanobanana_processor