"""
tests/test_theme_evidence.py — ThemeEvidence 계산 로직 테스트

실행:
  python -m pytest tests/test_theme_evidence.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from deskterior.recommender.engine import compute_theme_evidence, Product


class TestThemeEvidence(unittest.TestCase):
    """compute_theme_evidence 함수 테스트."""

    def _make_product(self, name: str, category: str = "KEYBOARD") -> Product:
        return Product(
            id="1", name=name, category=category,
            price=50000, color_tags=[], style_tags=[],
            image_sim=0.5, text_sim=0.5,
            image_embedding=None, text_embedding=None,
            product_url=None, image_url=None,
        )

    def test_white_theme_positive(self):
        p = self._make_product("화이트 아이보리 키보드")
        score = compute_theme_evidence(p, "white")
        self.assertGreater(score, 0.0, "화이트 키워드가 있으면 양수여야 한다")

    def test_black_theme_positive(self):
        p = self._make_product("블랙 무광 기계식 키보드")
        score = compute_theme_evidence(p, "black")
        self.assertGreater(score, 0.0)

    def test_gaming_theme_positive(self):
        p = self._make_product("RGB 게이밍 기계식 키보드", category="KEYBOARD")
        score = compute_theme_evidence(p, "gaming")
        self.assertGreater(score, 0.0)

    def test_wood_theme_positive(self):
        p = self._make_product("우드 원목 베이지 키보드")
        score = compute_theme_evidence(p, "wood")
        self.assertGreater(score, 0.0)

    def test_negative_keyword_penalty(self):
        """반대 테마 키워드가 있으면 점수가 낮아야 한다."""
        white_negative = self._make_product("블랙 다크 게이밍 키보드")
        score_white = compute_theme_evidence(white_negative, "white")
        white_clean = self._make_product("화이트 클린 키보드")
        score_white_clean = compute_theme_evidence(white_clean, "white")
        self.assertLess(score_white, score_white_clean)

    def test_score_range(self):
        """점수는 항상 0~1 사이여야 한다."""
        products = [
            self._make_product("화이트 키보드", "KEYBOARD"),
            self._make_product("블랙 RGB 게이밍 마우스", "MOUSE"),
            self._make_product("우드 베이지 스피커", "SPEAKER"),
        ]
        for theme in ("white", "black", "gaming", "wood"):
            for p in products:
                score = compute_theme_evidence(p, theme)
                self.assertGreaterEqual(score, 0.0, f"score must be >= 0 for theme={theme}")
                self.assertLessEqual(score, 1.0, f"score must be <= 1 for theme={theme}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
