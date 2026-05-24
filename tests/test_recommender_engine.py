"""
tests/test_recommender_engine.py — 추천 엔진 통합 테스트 (DB 없이 순수 로직만)

실행:
  python -m pytest tests/test_recommender_engine.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from deskterior.recommender.engine import (
    compute_setup_score,
    passes_theme_gate,
    pareto_filter_bundles,
    diversify_top_bundles,
    recommend_setup,
    Bundle,
    Product,
    ScoredProduct,
    compute_theme_evidence,
    compute_item_score,
)
from deskterior.recommender.config import MANDATORY_CATEGORIES, OPTIONAL_CATEGORIES


def _make_sp(name: str, category: str, theme: str, price: int = 50000) -> ScoredProduct:
    p = Product(
        id=str(abs(hash(name + category))),
        name=name,
        category=category,
        price=price,
        color_tags=[],
        style_tags=[],
        image_sim=0.7,
        text_sim=0.6,
        image_embedding=None,
        text_embedding=None,
        product_url=None,
        image_url=None,
    )
    te = compute_theme_evidence(p, theme)
    item_score = compute_item_score(0.7, 0.6, te, 0.5)
    return ScoredProduct(product=p, theme_evidence=te, value_score=0.5, item_score=item_score)


class TestSetupScore(unittest.TestCase):
    """compute_setup_score 테스트."""

    def test_score_range(self):
        """SetupScore는 0~1 사이여야 한다."""
        theme = "white"
        items = [
            _make_sp("화이트 모니터", "MONITOR", theme, 200000),
            _make_sp("화이트 키보드", "KEYBOARD", theme, 80000),
            _make_sp("화이트 마우스", "MOUSE", theme, 50000),
        ]
        bundle = Bundle(items=items, total_price=330000)
        compute_setup_score(bundle, theme)
        self.assertGreaterEqual(bundle.setup_score, 0.0)
        self.assertLessEqual(bundle.setup_score, 1.0)

    def test_empty_bundle_score_zero(self):
        bundle = Bundle(items=[], total_price=0)
        compute_setup_score(bundle, "white")
        self.assertEqual(bundle.setup_score, 0.0)


class TestParetoFilter(unittest.TestCase):
    """pareto_filter_bundles 테스트."""

    def _make_bundle(self, score: float, price: int) -> Bundle:
        b = Bundle(total_price=price)
        b.setup_score = score
        b.final_score = score
        return b

    def test_dominated_bundle_removed(self):
        """더 비싸고 점수도 낮은 조합은 제거되어야 한다."""
        good = self._make_bundle(0.8, 200000)
        bad = self._make_bundle(0.5, 300000)  # dominated: 비싸고 점수 낮음
        result = pareto_filter_bundles([good, bad])
        self.assertIn(good, result)
        self.assertNotIn(bad, result)

    def test_nondominated_both_kept(self):
        """서로 dominate하지 않으면 둘 다 유지한다."""
        a = self._make_bundle(0.8, 200000)
        b = self._make_bundle(0.6, 100000)  # b는 a보다 싸지만 점수도 낮음
        result = pareto_filter_bundles([a, b])
        self.assertEqual(len(result), 2)


class TestRecommendSetupNoDb(unittest.TestCase):
    """recommend_setup 함수를 DB 없이 mock 후보로 테스트."""

    def _candidates(self, theme: str) -> dict:
        cats = {
            "MONITOR":  [("화이트 모니터 32인치", 200000), ("화이트 모니터 27인치", 150000)],
            "KEYBOARD": [("화이트 아이보리 키보드", 80000), ("화이트 무접점 키보드", 120000)],
            "MOUSE":    [("화이트 무선 마우스", 50000), ("화이트 버티컬 마우스", 60000)],
            "MOUSEPAD": [("화이트 데스크매트", 30000)],
            "SPEAKER":  [("화이트 미니멀 스피커", 50000)],
            "DESK_LAMP": [("화이트 LED 조명", 40000)],
            "HEADSET":  [("화이트 헤드셋", 70000)],
        }
        result = {}
        for cat, products in cats.items():
            result[cat] = [_make_sp(name, cat, theme, price) for name, price in products]
        return result

    def test_returns_bundles(self):
        """충분한 예산에서 번들이 반환되어야 한다."""
        candidates = self._candidates("white")
        bundles = recommend_setup(
            theme="white",
            budget=500000,
            candidates_by_category=candidates,
            beam_size=10,
            top_m=5,
            top_k=3,
        )
        self.assertIsInstance(bundles, list)

    def test_budget_constraint(self):
        """예산 내 조합만 반환되어야 한다."""
        candidates = self._candidates("white")
        budget = 500000
        bundles = recommend_setup(
            theme="white",
            budget=budget,
            candidates_by_category=candidates,
            beam_size=20,
            top_m=5,
            top_k=3,
        )
        for bundle in bundles:
            self.assertLessEqual(bundle.total_price, budget,
                                 f"Bundle price {bundle.total_price} exceeds budget {budget}")

    def test_insufficient_budget_returns_empty(self):
        """너무 낮은 예산에서는 빈 리스트를 반환해야 한다."""
        candidates = self._candidates("white")
        bundles = recommend_setup(
            theme="white",
            budget=1000,  # 모든 상품보다 훨씬 낮은 예산
            candidates_by_category=candidates,
        )
        self.assertEqual(bundles, [])

    def test_mandatory_categories_covered(self):
        """반환된 번들은 필수 3종(MONITOR, KEYBOARD, MOUSE)을 모두 포함해야 한다."""
        candidates = self._candidates("white")
        bundles = recommend_setup(
            theme="white",
            budget=500000,
            candidates_by_category=candidates,
        )
        for bundle in bundles:
            covered = {sp.product.category for sp in bundle.items}
            for cat in MANDATORY_CATEGORIES:
                self.assertIn(cat, covered, f"Bundle missing mandatory category: {cat}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
