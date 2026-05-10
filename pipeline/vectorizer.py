# =============================================================================
# vectorizer.py — 상품 이미지 → KoCLIP vision encoder → embedding 벡터화
# =============================================================================

import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

# PostgreSQL이 설정한 CURL_CA_BUNDLE이 requests/pooch의 SSL을 깨뜨리므로 제거
os.environ.pop("CURL_CA_BUNDLE", None)

import certifi
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from rembg import new_session as rembg_new_session, remove as rembg_remove
from transformers import CLIPModel, CLIPProcessor

from core.config import SKIP_REMBG_CATEGORIES
from db.db_manager import DBManager

logger = logging.getLogger(__name__)

LEGACY_CLIP_MODEL_NAME = "Bingsu/clip-vit-base-patch32-ko"
_clip_model = None
_clip_processor = None


def load_clip_model():
    """KoCLIP 모델/프로세서를 최초 1회만 로드한다."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[모델 로드] {LEGACY_CLIP_MODEL_NAME} 로딩 중... (legacy vectorizer)")
        _clip_model = CLIPModel.from_pretrained(LEGACY_CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(LEGACY_CLIP_MODEL_NAME)
        _clip_model.eval()
        print("[모델 로드] 완료")
    return _clip_model, _clip_processor


class ProductVectorizer:
    """DB에 저장된 상품 이미지를 KoCLIP으로 벡터화하는 클래스 (지연 로딩)."""

    def __init__(self, db: DBManager):
        self.db = db
        self._rembg_session = None

    # ── 지연 로딩 ─────────────────────────────────────────────────────────────

    def _load_rembg_session(self):
        """최초 호출 시에만 rembg BiRefNet 세션을 로드 (모델 다운로드 1회)."""
        if self._rembg_session is None:
            print("[rembg] BiRefNet 모델 로드 중... (최초 1회 다운로드)")
            self._rembg_session = rembg_new_session("birefnet-general")
            print("[rembg] 완료")
        return self._rembg_session

    # ── 이미지 임베딩 ─────────────────────────────────────────────────────────

    def _embed_image(self, image_url: str, skip_rembg: bool = False) -> torch.Tensor:
        """이미지 URL을 다운로드해 KoCLIP vision encoder를 통과시킨 뒤
        L2 정규화된 (D,) 텐서를 반환.

        skip_rembg=False: rembg로 배경 제거 후 제품 bounding box 크롭.
        skip_rembg=True : MOUSEPAD 등 납작한 면 제품은 rembg가 역효과를 내므로
                          정사각형 패딩만 적용.
        """
        model, processor = load_clip_model()

        resp = requests.get(image_url, timeout=10, verify=certifi.where())
        resp.raise_for_status()
        raw_bytes = resp.content

        if skip_rembg:
            # rembg는 생략하되 정사각형 흰 배경 패딩은 동일하게 적용
            raw_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            w, h = raw_img.size
            side = max(w, h)
            background = Image.new("RGB", (side, side), (255, 255, 255))
            offset = ((side - w) // 2, (side - h) // 2)
            background.paste(raw_img, offset)
            img = background
        else:
            rembg_session = self._load_rembg_session()
            img_no_bg = rembg_remove(Image.open(BytesIO(raw_bytes)).convert("RGBA"), session=rembg_session)

            alpha = img_no_bg.split()[3]
            bbox = alpha.getbbox()
            if bbox is None:
                img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            else:
                img_no_bg = img_no_bg.crop(bbox)

                # 가로세로 비율 왜곡 방지: 긴 쪽 기준 정사각형 패딩
                w, h = img_no_bg.size
                side = max(w, h)
                background = Image.new("RGB", (side, side), (255, 255, 255))
                offset = ((side - w) // 2, (side - h) // 2)
                background.paste(img_no_bg, offset, mask=img_no_bg.split()[3])
                img = background

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            vision_out = model.vision_model(
                pixel_values=inputs["pixel_values"],
            )
            pooled = vision_out.pooler_output
            features = model.visual_projection(pooled)

        features = F.normalize(features, dim=-1)
        return features[0]  # (D,) tensor

    # ── 텍스트 임베딩 ─────────────────────────────────────────────────────────

    def _embed_text(self, title: str, category: str | None = None) -> torch.Tensor:
        """상품명 텍스트를 512차원 L2 정규화 텐서로 변환."""
        model, processor = load_clip_model()
        text_input = title or ""
        inputs = processor(text=[text_input], return_tensors="pt", padding=True)

        with torch.no_grad():
            text_out = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            pooled = text_out.pooler_output
            features = model.text_projection(pooled)

        return F.normalize(features, dim=-1)[0]  # (512,)

    # ── 전체 벡터화 ───────────────────────────────────────────────────────────

    def vectorize_all(self) -> int:
        """embedding IS NULL인 상품을 모두 처리하여 DB를 업데이트. 업데이트 수 반환."""
        self.db.ensure_split_embedding_columns()
        rows = self.db.get_products_missing_split_embeddings()
        total = len(rows)
        print(f"\n[벡터화] {total}개 상품 처리 시작...")

        updated = 0
        for product_id, image_url, category, title in rows:
            try:
                skip_rembg = category in SKIP_REMBG_CATEGORIES

                img_vec = self._embed_image(image_url, skip_rembg=skip_rembg)  # (D,)
                txt_vec = self._embed_text(title or "", category=category)     # (D,)

                self.db.update_split_embeddings(
                    product_id,
                    str(img_vec.tolist()),
                    str(txt_vec.tolist()),
                )
                updated += 1
                if updated % 20 == 0:
                    print(f"  {updated}/{total} 처리 완료...")
            except requests.exceptions.RequestException as e:
                logger.warning("  ID %s 이미지 다운로드 실패: %s", product_id, e)
            except Exception as e:
                logger.warning("  ID %s 건너뜀: %s", product_id, e)

        print(f"[벡터화] 완료: {updated}/{total}개 업데이트")
        return updated

    def vectorize_from_local_images(self, image_source: str) -> int:
        """로컬 이미지({id}.png) 기준으로 분리/하이브리드 임베딩을 생성해 DB를 업데이트.

        image_source가 zip 파일이면 자동 압축 해제 후 폴더를 사용한다.
        """
        source_path = Path(image_source)
        if not source_path.exists():
            fallback = Path("data") / "raw" / source_path.name
            if fallback.exists():
                source_path = fallback

        if image_source.lower().endswith(".zip"):
            image_dir = str(source_path.with_suffix(""))
            if not os.path.isdir(image_dir):
                print(f"[압축 해제] {source_path} -> {image_dir}/")
                with zipfile.ZipFile(source_path, "r") as zf:
                    zf.extractall(image_dir)
                print("[압축 해제 완료]")
            else:
                print(f"[압축 해제 생략] 이미 존재: {image_dir}/")
        else:
            image_dir = str(source_path)

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")

        self.db.ensure_split_embedding_columns()
        rows = self.db.get_products_missing_split_embeddings()
        total = len(rows)
        print(f"\n[벡터화-로컬] 분리 임베딩 미완료 상품: {total}개")
        print(f"[이미지 폴더] {image_dir}\n")

        updated = 0
        skipped_no_image = 0

        for product_id, _image_url, category, title in rows:
            img_path = os.path.join(image_dir, f"{product_id}.png")
            if not os.path.exists(img_path):
                skipped_no_image += 1
                continue

            try:
                model, processor = load_clip_model()
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")

                with torch.no_grad():
                    vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
                    pooled = vision_out.pooler_output
                    img_features = model.visual_projection(pooled)

                img_vec = F.normalize(img_features, dim=-1)[0]
                txt_vec = self._embed_text(title or "", category=category)

                self.db.update_split_embeddings(
                    product_id,
                    str(img_vec.tolist()),
                    str(txt_vec.tolist()),
                )
                updated += 1
                if updated % 20 == 0:
                    print(f"  {updated}/{total} 처리 완료...")
            except Exception as e:
                logger.warning("  ID %s 건너뜀: %s", product_id, e)

        print(f"[벡터화-로컬] 완료: {updated}/{total}개 업데이트 (이미지 없음: {skipped_no_image})")
        return updated