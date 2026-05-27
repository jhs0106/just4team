# recommendation의 자유 텍스트(color/theme) → AI 서버 StyleName enum 매핑.
# theme이 우선순위 높음 (분위기가 색감보다 데스크 스타일을 더 강하게 결정).
# 매칭 안 되면 "general"로 폴백.

from ..models import StyleName


# 색감 키워드 → StyleName
_COLOR_TO_STYLE: dict[str, str] = {
    # white 계열
    "화이트":   "white",
    "white":    "white",
    "흰색":     "white",
    "하양":     "white",
    "아이보리": "white",
    "베이지":   "white",
    # black 계열
    "블랙":     "black",
    "black":    "black",
    "검정":     "black",
    "검은":     "black",
    "다크":     "black",
    "dark":     "black",
    # 우드/내추럴 계열 → cozy 매핑
    "우드":     "cozy",
    "원목":     "cozy",
    "wood":     "cozy",
    "내추럴":   "nordic",
    "natural":  "nordic",
    "그레이":   "modern",
    "gray":     "modern",
    "grey":     "modern",
    # 기타
    "rgb":      "gaming",
    "RGB":      "gaming",
    "네온":     "gaming",
    "neon":     "gaming",
}

# 테마 키워드 → StyleName (theme이 color보다 우선)
_THEME_TO_STYLE: dict[str, str] = {
    "미니멀":           "modern",
    "minimal":          "modern",
    "모던":             "modern",
    "modern":           "modern",
    "심플":             "modern",
    "simple":           "modern",
    "게이밍":           "gaming",
    "gaming":           "gaming",
    "rgb":              "gaming",
    "감성":             "cozy",
    "아늑":             "cozy",
    "cozy":             "cozy",
    "코지":             "cozy",
    "따뜻":             "cozy",
    "warm":             "cozy",
    "북유럽":           "nordic",
    "노르딕":           "nordic",
    "nordic":           "nordic",
    "스칸디나비안":     "nordic",
    "scandinavian":     "nordic",
    "레트로":           "retro",
    "retro":            "retro",
    "빈티지":           "retro",
    "vintage":          "retro",
    "인더스트리얼":     "industrial",
    "industrial":       "industrial",
    "공장":             "industrial",
    "메탈":             "industrial",
    "metal":            "industrial",
}


def _match_text_to_style(text: str, mapping: dict[str, str]) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    for keyword, style in mapping.items():
        if keyword.lower() in t:
            return style
    return None


def map_to_style(color_text: str = "", theme_text: str = "") -> StyleName:
    # theme이 우선, theme 없거나 매칭 안 되면 color, 둘 다 없으면 general
    theme_style = _match_text_to_style(theme_text, _THEME_TO_STYLE)
    if theme_style:
        return StyleName(theme_style)

    color_style = _match_text_to_style(color_text, _COLOR_TO_STYLE)
    if color_style:
        return StyleName(color_style)

    return StyleName.general
