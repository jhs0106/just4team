"""
tests/test_optional_gain.py — OptionalGain 계산 로직 테스트

실행:
  python -m pytest tests/test_optional_gain.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from deskterior.recommender.engine import (
    compute_optional_gain,
    compute_theme_evidence,
    compute_item_score,
    Bundle,
    Product,
    ScoredProduct,
)
from deskterior.recommender.config import OPTIONAL_GAIN_THRESHOLD


class TestOptionalGain(unittest.TestCase):
    """compute_optional_gain 함수 테스트."""

    def _make_scored_product(
        self,
        name: str,
        category: str,
        theme: str,
        price: int = 30000,
        image_sim: float = 0.7,
        text_sim: float = 0.6,
    ) -> ScoredProduct:
        product = Product(
            id=str(hash(name)),
            name=name,
            category=category,
            price=price,
            color_tags=[],
            style_tags=[],
            image_sim=image_sim,
            text_sim=text_sim,
            image_embedding=None,
            text_embedding=None,
            product_url=None,
            image_url=None,
        )
        te = compute_theme_evidence(product, theme)
        item_score = compute_item_score(image_sim, text_sim, te, 0.5)
        return ScoredProduct(product=product, theme_evidence=te, value_score=0.5, item_score=item_score)

    def test_gain_range(self):
        """OptionalGain은 항상 0~1 사이여야 한다."""
        mandatory = [
            self._make_scored_product("화이트 모니터 32인치", "MONITOR", "white"),
            self._make_scored_product("화이트 아이보리 키보드", "KEYBOARD", "white"),
            self._make_scored_product("화이트 무선 마우스", "MOUSE", "white"),
        ]
        bundle = Bundle(
            items=mandatory,
            total_price=sum(sp.product.price for sp in mandatory),
        )
        candidate = self._make_scored_product("화이트 데스크매트", "MOUSEPAD", "white")
        gain = compute_optional_gain(bundle, candidate, "white", "MOUSEPAD")
        self.assertGreaterEqual(gain, 0.0)
        self.assertLessEqual(gain, 1.0)

    def test_theme_matching_candidate_has_higher_gain(self):
        """테마에 맞는 후보가 그렇지 않은 후보보다 gain이 높아야 한다."""
        mandatory = [
            self._make_scored_product("화이트 모니터", "MONITOR", "white"),
            self._make_scored_product("화이트 키보드", "KEYBOARD", "white"),
            self._make_scored_product("화이트 마우스", "MOUSE", "white"),
        ]
        bundle = Bundle(items=mandatory, total_price=150000)

        good = self._make_scored_product("화이트 아이보리 데스크매트", "MOUSEPAD", "white")
        bad = self._make_scored_product("블랙 다크 게이밍 패드", "MOUSEPAD", "white")

        gain_good = compute_optional_gain(bundle, good, "white", "MOUSEPAD")
        gain_bad = compute_optional_gain(bundle, bad, "white", "MOUSEPAD")
        self.assertGreater(gain_good, gain_bad)

    def test_threshold_values_are_positive(self):
        """OPTIONAL_GAIN_THRESHOLD의 모든 값은 양수여야 한다."""
        for theme, categories in OPTIONAL_GAIN_THRESHOLD.items():
            for cat, threshold in categories.items():
                self.assertGreater(threshold, 0.0, f"{theme}/{cat} threshold should be > 0")
                self.assertLess(threshold, 1.0, f"{theme}/{cat} threshold should be < 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
