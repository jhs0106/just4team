"""Configuration for recommender/engine.py — themes, categories, scoring presets."""

# ============================================================
# CATEGORIES
# ============================================================

CANDIDATE_LIMIT = 50
BEAM_SIZE_DEFAULT = 50
TOP_M_DEFAULT = 10
TOP_K_DEFAULT = 1
ALLOW_THEME_GATE_FALLBACK = True

MANDATORY_CATEGORIES = ["MONITOR", "KEYBOARD", "MOUSE"]

OPTIONAL_CATEGORIES = ["MOUSEPAD", "SPEAKER", "DESK_LAMP", "HEADSET"]

# 카테고리별 절대 최소 가격 — 예산과 무관한 노이즈 제거용
ABS_MIN_PRICE = {
    "MONITOR":   70_000,
    "KEYBOARD":  15_000,
    "MOUSE":      8_000,
    "MOUSEPAD":  10_000,
    "SPEAKER":   15_000,
    "DESK_LAMP": 12_000,
    "HEADSET":   15_000,
}

CATEGORY_LABELS = {
    "MONITOR":   "모니터",
    "KEYBOARD":  "키보드",
    "MOUSE":     "마우스",
    "MOUSEPAD":  "마우스패드",
    "SPEAKER":   "스피커",
    "DESK_LAMP": "데스크 조명",
    "HEADSET":   "헤드셋/헤드폰",
}

CATEGORY_EXCLUDE_KEYWORDS = {
    "MONITOR":   ["TV", "tv", "티비", "벽걸이", "사이니지", "업소용", "차량용"],
    "KEYBOARD":  ["커버", "스킨", "보호필름", "청소", "키캡"],
    "MOUSE":     ["마우스피트", "마우스 번지", "차량용", "케이스"],
    "MOUSEPAD":  ["바닥", "욕실", "주방", "차량", "요가"],
    "SPEAKER":   ["차량용", "방수", "야외", "공연용", "업소용"],
    "DESK_LAMP": ["차량용", "실외", "수중", "방수", "가로등", "천장", "야외"],
    "HEADSET":   [
        "이어폰", "earphone", "이어버드", "earbuds", "보청기",
        "차량용", "헬멧", "마이크 단품", "케이블", "커버",
        "파우치", "이어패드", "쿠션", "거치대", "스탠드",
    ],
}

# ============================================================
# THEME PRESETS
# ============================================================

THEME_PRESETS = {
    "white": {
        "label": "화이트 클린 셋업",
        "description": "화이트, 아이보리, 크림 계열의 밝고 깔끔한 데스크 셋업",
        "positive_keywords": [
            "화이트", "아이보리", "크림", "실버", "밝은", "깔끔한", "클린",
        ],
        "negative_keywords": [
            "블랙", "검정", "다크", "rgb", "네온", "게이밍", "우드", "원목", "빈티지",
        ],
        "theme_prompt": "화이트 아이보리 크림 밝고 깔끔한 데스크 셋업",
        "mandatory_rule": {
            "min_matching_mandatory": 1,
            "min_avg_theme_evidence": 0.58,
        },
        "optional_priority": {
            "MOUSEPAD":  0.80,
            "SPEAKER":   0.60,
            "DESK_LAMP": 0.75,
            "HEADSET":   0.40,
        },
    },

    "black": {
        "label": "블랙 다크 셋업",
        "description": "블랙, 다크그레이, 무광 계열의 차분한 데스크 셋업",
        "positive_keywords": [
            "블랙", "검정", "다크그레이", "다크", "무광", "매트", "차분한",
        ],
        "negative_keywords": [
            "화이트", "아이보리", "크림", "핑크", "파스텔",
            "우드", "원목", "rgb", "네온",
        ],
        "theme_prompt": "블랙 다크그레이 무광 차분한 데스크 셋업",
        "mandatory_rule": {
            "min_matching_mandatory": 2,
            "min_avg_theme_evidence": 0.60,
        },
        "optional_priority": {
            "MOUSEPAD":  0.80,
            "SPEAKER":   0.70,
            "DESK_LAMP": 0.60,
            "HEADSET":   0.50,
        },
    },

    "gaming": {
        "label": "RGB 게이밍 셋업",
        "description": "RGB, LED, 고주사율, 게이밍 기어 중심의 데스크 셋업",
        "positive_keywords": [
            "게이밍", "rgb", "led", "네온", "기계식", "고성능",
            "144hz", "165hz", "dpi", "7.1", "서라운드", "장패드",
        ],
        "negative_keywords": [
            "우드", "원목", "북유럽", "빈티지", "클래식", "파스텔", "코지",
        ],
        "theme_prompt": "RGB LED 게이밍 고성능 몰입형 데스크 셋업",
        "mandatory_rule": {
            "min_matching_mandatory": 2,
            "min_avg_theme_evidence": 0.65,
        },
        "optional_priority": {
            "MOUSEPAD":  0.90,
            "SPEAKER":   0.70,
            "DESK_LAMP": 0.80,
            "HEADSET":   0.90,
        },
    },

    "wood": {
        "label": "우드 내추럴 셋업",
        "description": "우드, 원목, 베이지, 브라운 계열의 따뜻한 데스크 셋업",
        "positive_keywords": [
            "우드", "원목", "나무", "베이지", "브라운", "내추럴", "따뜻한", "월넛", "오크",
        ],
        "negative_keywords": [
            "rgb", "네온", "형광", "차가운", "메탈", "철제",
        ],
        "theme_prompt": "우드 원목 베이지 브라운 따뜻한 내추럴 데스크 셋업",
        "mandatory_rule": {
            "min_matching_mandatory": 1,
            "min_avg_theme_evidence": 0.53,
        },
        "optional_priority": {
            "MOUSEPAD":  0.50,
            "SPEAKER":   0.70,
            "DESK_LAMP": 0.90,
            "HEADSET":   0.30,
        },
    },
}

# ============================================================
# THEME-CATEGORY SEARCH QUERIES
# ============================================================

THEME_CATEGORY_QUERIES = {
    "white": {
        "MONITOR":   "화이트 모니터",
        "KEYBOARD":  "화이트 아이보리 키보드",
        "MOUSE":     "화이트 무선 마우스",
        "MOUSEPAD":  "화이트 데스크매트 마우스패드 장패드",
        "SPEAKER":   "화이트 미니멀 스피커",
        "DESK_LAMP": "화이트 LED 데스크 조명 스탠드",
        "HEADSET":   "화이트 헤드셋 헤드폰",
    },
    "black": {
        "MONITOR":   "블랙 모니터",
        "KEYBOARD":  "블랙 무광 키보드",
        "MOUSE":     "블랙 무선 마우스",
        "MOUSEPAD":  "블랙 데스크매트 장패드 마우스패드",
        "SPEAKER":   "블랙 데스크탑 스피커",
        "DESK_LAMP": "블랙 데스크 조명 스탠드",
        "HEADSET":   "블랙 헤드셋 헤드폰",
    },
    "gaming": {
        "MONITOR":   "게이밍 모니터 144Hz 165Hz",
        "KEYBOARD":  "RGB 게이밍 기계식 키보드",
        "MOUSE":     "게이밍 마우스 DPI RGB 경량",
        "MOUSEPAD":  "RGB 게이밍 장패드 마우스패드 데스크매트",
        "SPEAKER":   "RGB 게이밍 스피커",
        "DESK_LAMP": "RGB LED 데스크 조명",
        "HEADSET":   "게이밍 헤드셋 7.1 RGB",
    },
    "wood": {
        "MONITOR":   "베이지 화이트 모니터",
        "KEYBOARD":  "베이지 우드 키보드",
        "MOUSE":     "베이지 무선 마우스",
        "MOUSEPAD":  "우드 베이지 데스크매트 마우스패드",
        "SPEAKER":   "우드 원목 블루투스 스피커",
        "DESK_LAMP": "우드 원목 데스크 조명 스탠드",
        "HEADSET":   "베이지 헤드셋 헤드폰",
    },
}

# ============================================================
# OPTIONAL GAIN THRESHOLDS — 테마별 선택 상품 추가 기준
# ============================================================

# 카테고리별 OptionalGain 임계값.
# MOUSEPAD는 입력 장치군과 직접 연결되는 보완재라 threshold를 낮게.
# SPEAKER/DESK_LAMP/HEADSET은 테마·예산에 따라 선택적으로 추가.
OPTIONAL_GAIN_THRESHOLD = {
    "MOUSEPAD":  0.45,
    "SPEAKER":   0.50,
    "DESK_LAMP": 0.55,
    "HEADSET":   0.55,
}

# ============================================================
# ROLE-PAIR WEIGHTS — 카테고리 간 상호보완 관계
# ============================================================

ROLE_PAIR_WEIGHTS: dict[frozenset, float] = {
    frozenset(("KEYBOARD", "MOUSE")):     1.00,
    frozenset(("KEYBOARD", "MOUSEPAD")):  0.95,
    frozenset(("MOUSE",    "MOUSEPAD")):  0.95,
    frozenset(("MONITOR",  "SPEAKER")):   0.70,
    frozenset(("MONITOR",  "DESK_LAMP")): 0.60,
    frozenset(("KEYBOARD", "HEADSET")):   0.50,
    frozenset(("MOUSE",    "HEADSET")):   0.50,
}

# ============================================================
# THEME NEGATIVES — 대조 임베딩 (참고용)
# ============================================================

THEME_NEGATIVES = {
    "white":  ["블랙 다크 게이밍 데스크 셋업", "RGB 네온 게이밍 데스크 셋업"],
    "black":  ["화이트 아이보리 미니멀 데스크 셋업", "우드 내추럴 따뜻한 데스크 셋업"],
    "gaming": ["우드 북유럽 따뜻한 데스크 셋업", "화이트 미니멀 심플 데스크 셋업"],
    "wood":   ["RGB 게이밍 네온 데스크 셋업", "블랙 다크 게이밍 데스크 셋업"],
}
