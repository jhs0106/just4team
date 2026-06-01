# nanobanana -> openai 변경
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

    def _read_product_image_base64(self, image_id: Any, image_url: Optional[str] = None) -> Optional[str]:
        if image_id is None:
            return None

        candidates = [
            Path("data/test/processed_images") / f"{image_id}.png",
            Path("ai-server/data/test/processed_images") / f"{image_id}.png",
            Path("data/test/processed_images") / f"{image_id}.fallback.png",
        ]

        for path in candidates:
            if path.exists():
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

        # 로컬에 없으면 image_url에서 다운로드 (배경 제거 안 됨, OpenAI는 raw도 사용 가능)
        if image_url:
            try:
                resp = requests.get(image_url, timeout=10)
                if resp.status_code == 200 and len(resp.content) >= 1024:
                    cache_dir = Path("data/test/processed_images")
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = cache_dir / f"{image_id}.fallback.png"
                    cache_path.write_bytes(resp.content)
                    print(f"[_read_product_image_base64] 폴백 다운로드 → {cache_path}")
                    return base64.b64encode(resp.content).decode("utf-8")
            except Exception as e:
                print(f"[_read_product_image_base64] 폴백 예외 image_id={image_id}: {e}")
        return None

    def build_prompt(
            self,
            theme: str,
            mode: str,
            products: List[Dict[str, Any]],
            space_constraints: Optional[Dict[str, Any]] = None,
            desk_width_mm: Optional[int] = None,
            desk_depth_mm: Optional[int] = None,
    ) -> str:
        product_lines = []

        for idx, product in enumerate(products, start=1):
            name     = product.get("name") or "recommended product"
            category = product.get("category") or ""
            brand    = product.get("brand")
            w_mm     = product.get("width_mm")
            d_mm     = product.get("depth_mm")

            line = f"{idx}. [{category}] {name}" if category else f"{idx}. {name}"
            if brand:
                line += f" (brand: {brand})"
            if w_mm and d_mm:
                line += f" — actual size {w_mm}mm × {d_mm}mm"
            elif w_mm:
                line += f" — actual width {w_mm}mm"
            product_lines.append(line)

        product_text = "\n".join(product_lines) if product_lines else "No product information."

        # 책상 실측 사이즈 — 제품 간 상대 스케일을 OpenAI가 맞출 수 있도록 명시.
        desk_lines = []
        if desk_width_mm:
            desk_lines.append(f"width: {desk_width_mm}mm ({desk_width_mm/10:.0f}cm)")
        if desk_depth_mm:
            desk_lines.append(f"depth: {desk_depth_mm}mm ({desk_depth_mm/10:.0f}cm)")
        desk_text = " / ".join(desk_lines) if desk_lines else "size not provided"

        space_text = "No explicit available-space constraints."
        if space_constraints:
            space_text = str(space_constraints)

        return f"""
You are a professional deskterior designer. Compose a realistic photograph by adding the recommended products to the user's actual desk photo.

Style theme: {theme}
Desk mode: {mode}
User desk actual size: {desk_text}

Recommended products (place ALL of them; sizes are real-world measurements):
{product_text}

Available space on desk:
{space_text}

Composition rules:
- Preserve the original desk, walls, room, camera angle, and perspective exactly.
- Use the provided product reference images for shape, material, and color identity.
- Scale each product to match its actual width/depth in mm relative to the desk size given above.
  (Example: a 440mm keyboard on a 1200mm desk should occupy ~37% of desk width.)
- Place products on the desk surface — never floating, never overlapping incorrectly.
- Respect typical deskterior layout: monitor at back center, keyboard in front of monitor, mouse to the right of keyboard.

Photorealism rules:
- Match the room's lighting direction, color temperature, and shadow style for every product.
- Monitor screens should show neutral desktop wallpaper or be powered off — do NOT render marketing text, spec numbers, or product-photo screens.
- RGB lighting on peripherals should be subtle and ambient, not the saturated glow from product catalog photos.
- Add realistic contact shadows where products meet the desk; no cutout halos or hard composite edges.

Do NOT:
- Change the desk shape, room, wallpaper, or camera angle.
- Skip products or substitute different products.
- Add watermarks, text overlays, brand logos that aren't on the actual product, or any extra furniture.
""".strip()

    def build_products_payload(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload_products = []

        for product in products:
            image_id  = product.get("image_id") or product.get("id")
            image_url = product.get("image_url")

            payload_products.append({
                "id":           image_id,
                "name":         product.get("name"),
                "category":     product.get("category"),
                "price":        product.get("price"),
                "brand":        product.get("brand"),
                "image_url":    image_url,
                "product_url":  product.get("product_url"),
                "image_base64": self._read_product_image_base64(image_id, image_url),
            })

        return payload_products

    def generate(
            self,
            image_base64: str,
            theme: str,
            mode: str,
            products: List[Dict[str, Any]],
            space_constraints: Optional[Dict[str, Any]] = None,
            desk_width_mm: Optional[int] = None,
            desk_depth_mm: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(
            theme=theme,
            mode=mode,
            products=products,
            space_constraints=space_constraints,
            desk_width_mm=desk_width_mm,
            desk_depth_mm=desk_depth_mm,
        )

        # OpenAI gpt-image-1 이미지 편집: 멀티파트로 [책상 사진] + [제품 이미지들] 업로드.
        # image[]: 첫 번째가 편집 베이스(책상), 이후는 합성 참조(제품). gpt-image-1은 다중 입력 지원.
        desk_mime = _guess_mime(image_base64)
        desk_ext = "jpg" if desk_mime == "image/jpeg" else "png"
        files = [
            ("image[]", (f"desk.{desk_ext}", _b64_to_bytes(image_base64), desk_mime)),
        ]
        for p in products:
            pid     = p.get("image_id") or p.get("id")
            img_b64 = self._read_product_image_base64(pid, p.get("image_url"))
            if img_b64:
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

    def harmonize(self, image_base64: str, products: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Tier 3: CV 합성본(제거+배치+그림자)을 받아 OpenAI로 사실감 다듬음.
        # 멀티 입력: [1번] 합성본 = 레이아웃/위치 참조, [2번~] 각 제품 원본 = 외형 참조.
        #   → 합성본에서 마우스가 마우스패드를 가려도 제품 원본을 보고 정확히 재현 가능.
        # 위치·정체성·실루엣은 보존하되, 광고용 화면 콘텐츠·과도한 RGB·합성 티는 자연화 허용.
        # 이전 "exactly as in input" + input_fidelity=high는 cv 합성판이 그대로 출력되는 원인이라 완화.

        # 제품 참조 이미지 수집 (순서 유지 — 프롬프트 번호와 일치)
        ref_files = []
        ref_lines = []
        for _p in (products or []):
            _pid     = _p.get("image_id") or _p.get("id")
            _img_b64 = self._read_product_image_base64(_pid, _p.get("image_url"))
            if not _img_b64:
                continue
            _idx  = len(ref_files) + 1
            _cat  = (_p.get("category") or "").strip()
            _name = (_p.get("name") or _cat or "product").strip()
            ref_files.append(("image[]", (f"ref_{_idx}_{_pid}.png", _b64_to_bytes(_img_b64), "image/png")))
            ref_lines.append(f"  Reference image {_idx}: {_cat} — \"{_name}\"")

        if ref_lines:
            ref_block = (
                "\n\nIMAGE ROLES:\n"
                "- The FIRST image is the rough scene: it defines the LAYOUT — where each product sits, its "
                "position, footprint, and how items overlap.\n"
                "- The following images are clean close-up references of the SAME products already in the scene, "
                "in this order:\n"
                + "\n".join(ref_lines) +
                "\nUse each reference image ONLY to reproduce that product's true shape, proportions, color, and "
                "details at the position shown in the first image. This is critical where products overlap or are "
                "partially occluded in the first image (e.g. a mouse on top of a mousepad): rebuild each product "
                "faithfully from its reference instead of guessing. Do NOT add extra copies of any product, and "
                "do NOT introduce products that are not in the first image."
            )
        else:
            ref_block = ""

        prompt = (
            "This is a rough composite of a desk setup — products have already been placed on the user's desk. "
            "Re-render it as one cohesive, realistic photograph taken in this room. "
            "PRESERVE for each product: its position on the desk, overall silhouette/shape, category identity, "
            "approximate size, and dominant material/color. Do NOT move products to different desk regions, "
            "swap products for different categories, add new products, or remove any product. "
            "When a mouse rests on a mousepad, render them as TWO clearly separate objects — the mouse is a "
            "distinct 3D peripheral with its own outline sitting on top of the flat pad; never fuse the mouse "
            "and the pad into a single blob, and keep the mouse's recognizable mouse shape. "
            "NATURALIZE these details: monitor screens should show neutral desktop wallpaper or be turned off "
            "(remove any advertisement text, spec numbers, marketing graphics on screens); "
            "RGB lighting on keyboards/speakers/mousepads should be subtle and match the room ambience "
            "(no neon product-photo glow); product cutout edges, harsh shadows, and copy-paste seams must "
            "blend into the desk surface with realistic contact shadows and reflections; "
            "match the room's actual lighting direction, color temperature, and ambient occlusion. "
            "Keep the desk geometry, walls, and room structure intact. "
            "Output must look like a single photograph, not a collage."
            + ref_block
        )

        mime = _guess_mime(image_base64)
        ext = "jpg" if mime == "image/jpeg" else "png"
        files = [("image[]", (f"composite.{ext}", _b64_to_bytes(image_base64), mime))]
        files.extend(ref_files)   # 합성본 다음에 제품 참조 이미지들
        # input_fidelity 미설정(=auto). high는 입력 픽셀을 거의 그대로 보존해 cv 합성티가 남음 →
        # 화면 콘텐츠/RGB 자연화·블렌딩이 작동하지 않아서 제거.
        form = {"model": self.model, "prompt": prompt, "size": self.size}

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
                f"OpenAI harmonize 응답에 이미지 없음 (keys={list(data.keys())}, body={response.text[:500]})"
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