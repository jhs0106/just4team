# === searcher.py — 상품 검색 (Jina CLIP v2 임베딩 + pgvector 유사도)

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from core.config import CATEGORY_KEYWORDS, MODEL_NAME, TITLE_BOOST_WEIGHT
from db.db_manager import DBManager
from core.scoring import _title_match_scores, _zscore

logger = logging.getLogger(__name__)

_jina_model = None


def _load_jina_model():
    """Jina CLIP v2 모델을 최초 1회만 로드."""
    global _jina_model
    if _jina_model is None:
        print(f"[모델 로드] {MODEL_NAME} 로딩 중...")
        _jina_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=False)
        _jina_model.eval()
        print("[모델 로드] 완료")
    return _jina_model


class ProductSearcher:
    """사용자 텍스트를 Jina CLIP v2 query 임베딩으로 변환해 상품을 검색."""

    def __init__(self, db: DBManager):
        self.db = db

    def get_query_embedding(self, query_text: str) -> np.ndarray:
        """텍스트 쿼리를 Jina CLIP v2 query 벡터로 변환 후 L2 정규화."""
        model = _load_jina_model()

        with torch.no_grad():
            feat = model.encode_text([query_text])

            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)

            feat = feat.float()
            feat = F.normalize(feat, dim=-1)
            return feat[0].cpu().numpy()

    @staticmethod
    def detect_category(query_text: str) -> str | None:
        """검색어에 카테고리 키워드가 포함되어 있으면 해당 카테고리 코드 반환."""
        query_lower = query_text.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    return category
        return None

    def search(
        self,
        query_text: str,
        category_filter: str = None,
        max_price: int = None,
        top_k: int = 10,
        mode: Literal["image_only", "text_only", "raw_sum", "equal_zsum", "weighted_sum"] = "equal_zsum",
        image_weight: float = 1.0,
        text_weight: float = 1.0,
    ) -> list:
        """
        late-fusion 방식으로 상품 검색.

        mode:
            - image_only: query↔image
            - text_only:  query↔text
            - raw_sum:    image_score + text_score
            - equal_zsum: z(image_score) + z(text_score)
            - weighted_sum: image_weight*image_score + text_weight*text_score
        """
        if mode not in {"image_only", "text_only", "raw_sum", "equal_zsum", "weighted_sum"}:
            raise ValueError("mode는 image_only/text_only/raw_sum/equal_zsum/weighted_sum 중 하나여야 합니다.")

        query_vec = self.get_query_embedding(query_text)
        vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vec.tolist()) + "]"

        conditions = ["(embedding_img IS NOT NULL OR embedding_txt IS NOT NULL)"]
        where_params: list = []

        if category_filter:
            conditions.append("category = %s")
            where_params.append(category_filter)
        if max_price is not None:
            conditions.append("lprice <= %s")
            where_params.append(max_price)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT id, title, lprice, category, brand, mall_name, link,
                   CASE WHEN embedding_img IS NULL THEN NULL ELSE 1 - (embedding_img <=> %s::vector) END AS image_score,
                   CASE WHEN embedding_txt IS NULL THEN NULL ELSE 1 - (embedding_txt <=> %s::vector) END AS text_score
            FROM products
            WHERE {where_clause}
        """

        params = [vec_str, vec_str] + where_params
        with self.db.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            return []

        image_scores = np.array([float(r[7]) if r[7] is not None else np.nan for r in rows], dtype=np.float32)
        text_scores = np.array([float(r[8]) if r[8] is not None else np.nan for r in rows], dtype=np.float32)

        image_valid = np.isfinite(image_scores)
        text_valid = np.isfinite(text_scores)

        image_scores[~image_valid] = 0.0
        text_scores[~text_valid] = 0.0

        if mode == "image_only":
            final_scores = image_scores
        elif mode == "text_only":
            final_scores = text_scores
        elif mode == "raw_sum":
            final_scores = image_scores + text_scores
        elif mode == "weighted_sum":
            final_scores = (image_weight * image_scores) + (text_weight * text_scores)
        else:
            image_z = _zscore(image_scores, image_valid)
            text_z = _zscore(text_scores, text_valid)
            title_scores = _title_match_scores(query_text, [r[1] or "" for r in rows])
            final_scores = image_z + text_z + (TITLE_BOOST_WEIGHT * title_scores)

        top_k = min(top_k, len(rows))
        top_idx = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for rank_idx in top_idx:
            row = rows[int(rank_idx)]
            results.append(
                {
                    "id": int(row[0]),
                    "title": row[1],
                    "lprice": row[2],
                    "category": row[3],
                    "brand": row[4],
                    "mall_name": row[5],
                    "link": row[6],
                    "image_score": float(image_scores[rank_idx]),
                    "text_score": float(text_scores[rank_idx]),
                    "final_score": float(final_scores[rank_idx]),
                    "mode": mode,
                }
            )

        logger.info("[검색] '%s' mode=%s → %d개 결과", query_text, mode, len(results))
        return results

    @staticmethod
    def print_results(results: list, query_text: str = ""):
        """검색 결과를 콘솔에 출력."""
        label = f"'{query_text}' " if query_text else ""
        print(f"\n[검색결과] {label}({len(results)}개)")
        print("=" * 70)
        if not results:
            print("  결과 없음 (embedding_img/embedding_txt 확인 필요)")
        else:
            mode = results[0].get("mode", "unknown")
            print(f"  mode={mode}")
            for i, r in enumerate(results, 1):
                title = (r["title"] or "")[:45]
                print(f"  #{i:2} [final={r['final_score']:.4f} | img={r['image_score']:.4f} | txt={r['text_score']:.4f}] {title}")
                print(f"        {r['lprice']:,}원 | {r['brand']} | {r['mall_name']}")
                print(f"        링크: {r['link']}")
        print("=" * 70)
