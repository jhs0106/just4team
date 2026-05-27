"""
tests/test_category.py — 카테고리 분류 로직 테스트

실행:
  python -m pytest tests/test_category.py -v
  python -m unittest tests.test_category -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from deskterior.core.config import (
    CATEGORY_KEYWORDS,
    CATEGORY_ALIASES,
    KEYBOARD_MOUSE_CATEGORIES,
)


# ── 분류 헬퍼 (CATEGORY_KEYWORDS 기반 타이틀 분류) ───────────────────────────

def classify_by_keywords(title: str) -> str | None:
    """CATEGORY_KEYWORDS를 순서대로 순회하여 첫 번째 매칭 카테고리를 반환."""
    title_lower = title.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw.lower() in title_lower for kw in keywords):
            return category
    return None


def resolve_alias(alias: str) -> str | None:
    """CATEGORY_ALIASES에서 카테고리 코드를 반환."""
    return CATEGORY_ALIASES.get(alias.lower())


# ── 테스트 ─────────────────────────────────────────────────────────────────────

class TestHeadsetStandPriority(unittest.TestCase):
    """HEADSET_STAND > HEADSET 우선순위 테스트."""

    def test_headset_stand_before_headset_in_keywords(self):
        """CATEGORY_KEYWORDS dict에서 HEADSET_STAND가 HEADSET보다 앞에 있어야 한다.
        또한 HEADSET_STAND가 DESK_LAMP보다 앞에 있어야
        '헤드셋 스탠드'가 DESK_LAMP의 '스탠드' 키워드에 잘못 매칭되지 않는다.
        """
        keys = list(CATEGORY_KEYWORDS.keys())
        idx_stand = keys.index("HEADSET_STAND")
        idx_headset = keys.index("HEADSET")
        idx_desk_lamp = keys.index("DESK_LAMP")
        self.assertLess(
            idx_stand, idx_headset,
            "HEADSET_STAND must appear before HEADSET in CATEGORY_KEYWORDS"
        )
        self.assertLess(
            idx_stand, idx_desk_lamp,
            "HEADSET_STAND must appear before DESK_LAMP to prevent '헤드셋 스탠드' → DESK_LAMP"
        )

    def test_headset_stand_classified_correctly(self):
        """'헤드셋 거치대'는 HEADSET이 아닌 HEADSET_STAND로 분류되어야 한다."""
        titles = [
            "헤드셋 거치대 알루미늄 스탠드",
            "헤드폰 거치대 USB 허브",
            "헤드셋 스탠드 게이밍",
            "headset stand wood",
            "headphone stand",
        ]
        for title in titles:
            result = classify_by_keywords(title)
            self.assertEqual(
                result, "HEADSET_STAND",
                f"'{title}' should be HEADSET_STAND, got {result}"
            )

    def test_headset_classified_correctly(self):
        """'게이밍 헤드셋'은 HEADSET으로 분류되어야 한다."""
        titles = [
            "게이밍 헤드셋 7.1 채널",
            "무선 헤드폰 블루투스",
            "PC 헤드셋 마이크 포함",
            "gaming headset rgb",
            "bluetooth headphone",
        ]
        for title in titles:
            result = classify_by_keywords(title)
            self.assertEqual(
                result, "HEADSET",
                f"'{title}' should be HEADSET, got {result}"
            )

    def test_headset_stand_not_classified_as_headset(self):
        """거치대/스탠드가 포함된 제목이 HEADSET으로 잘못 분류되지 않아야 한다."""
        false_positive_titles = [
            "헤드셋 거치대",
            "헤드폰 스탠드 책상",
        ]
        for title in false_positive_titles:
            result = classify_by_keywords(title)
            self.assertNotEqual(
                result, "HEADSET",
                f"'{title}' must NOT be classified as HEADSET (got {result})"
            )


class TestCategoryAliases(unittest.TestCase):
    """CATEGORY_ALIASES 테스트."""

    def test_headset_aliases(self):
        aliases = ["headset", "headphone", "헤드셋", "헤드폰"]
        for alias in aliases:
            self.assertEqual(resolve_alias(alias), "HEADSET", f"alias '{alias}' should map to HEADSET")

    def test_headset_stand_aliases(self):
        aliases = [
            "headset_stand", "headsetstand", "headphone_stand", "headphonestand",
            "헤드셋거치대", "헤드폰거치대", "헤드셋스탠드", "헤드폰스탠드",
        ]
        for alias in aliases:
            self.assertEqual(resolve_alias(alias), "HEADSET_STAND", f"alias '{alias}' should map to HEADSET_STAND")


class TestKeyboardMouseCategories(unittest.TestCase):
    """KEYBOARD/MOUSE 카테고리 상수 검증."""

    def test_keyboard_mouse_categories_constant(self):
        self.assertIn("KEYBOARD", KEYBOARD_MOUSE_CATEGORIES)
        self.assertIn("MOUSE", KEYBOARD_MOUSE_CATEGORIES)
        self.assertNotIn("DESK", KEYBOARD_MOUSE_CATEGORIES)
        self.assertNotIn("HEADSET", KEYBOARD_MOUSE_CATEGORIES)


class TestNewCategories(unittest.TestCase):
    """HEADSET, HEADSET_STAND가 모든 설정에 추가되었는지 확인."""

    def test_headset_in_category_queries(self):
        from deskterior.core.config import CATEGORY_QUERIES
        self.assertIn("HEADSET", CATEGORY_QUERIES)
        self.assertIn("HEADSET_STAND", CATEGORY_QUERIES)

    def test_headset_in_category_labels(self):
        from deskterior.core.config import CATEGORY_LABELS
        self.assertIn("HEADSET", CATEGORY_LABELS)
        self.assertIn("HEADSET_STAND", CATEGORY_LABELS)
        self.assertEqual(CATEGORY_LABELS["HEADSET"], "헤드셋/헤드폰")
        self.assertEqual(CATEGORY_LABELS["HEADSET_STAND"], "헤드셋 거치대")

    def test_headset_in_exclude_keywords(self):
        from deskterior.core.config import CATEGORY_EXCLUDE_KEYWORDS
        self.assertIn("HEADSET", CATEGORY_EXCLUDE_KEYWORDS)
        self.assertIn("HEADSET_STAND", CATEGORY_EXCLUDE_KEYWORDS)

    def test_headset_not_in_keyboard_mouse(self):
        self.assertNotIn("HEADSET", KEYBOARD_MOUSE_CATEGORIES)
        self.assertNotIn("HEADSET_STAND", KEYBOARD_MOUSE_CATEGORIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
