_CATEGORY_ALIASES = {
    "KEYBOARD":      "KEYBOARD",
    "MOUSE":         "MOUSE",
    "MOUSEPAD":      "MOUSEPAD",
    "MOUSE_PAD":     "MOUSEPAD",
    "MOUSE PAD":     "MOUSEPAD",
    "MONITOR":       "MONITOR",
    "SPEAKER":       "SPEAKER",
    "LAMP":          "DESK_LAMP",
    "DESK LAMP":     "DESK_LAMP",
    "DESK_LAMP":     "DESK_LAMP",
    "DESK SHELF":    "DESK_SHELF",
    "DESK_SHELF":    "DESK_SHELF",
    "MONITOR RISER": "DESK_SHELF",
    "LAPTOP STAND":  "LAPTOP_STAND",
    "LAPTOP_STAND":  "LAPTOP_STAND",
    "DECO":          "DECO",
    "DECOR":         "DECO",
    "CLOCK":         "CLOCK",
    "LIGHTING":      "LIGHTING",
    "LIGHT BAR":     "LIGHTING",
    "LIGHTBAR":      "LIGHTING",
    "SCREEN BAR":    "LIGHTING",
    "SCREENBAR":     "LIGHTING",
    "MONITOR LIGHT": "LIGHTING",
}

_CATEGORY_DIMS_MM = {
    "KEYBOARD":     (440, 130),
    "MOUSE":        (70,  120),
    "MOUSEPAD":     (900, 400),
    "MONITOR":      (600, 200),
    "SPEAKER":      (90,  120),
    "DESK_LAMP":    (80,  400),
    "DESK_SHELF":   (600, 200),
    "LAPTOP_STAND": (280, 250),
    "DECO":         (80,  80),
    "CLOCK":        (100, 100),
    "LIGHTING":     (500, 50),
}

_PLACEMENT_ORDER = {
    "MONITOR":      10,
    "LIGHTING":     12,
    "KEYBOARD":     15,
    "DESK_SHELF":   20,
    "MOUSEPAD":     40,
    "MOUSE":        50,
    "SPEAKER":      60,
    "DESK_LAMP":    70,
    "CLOCK":        80,
    "DECO":         90,
}

_CV_ONLY_CATS: set[str] = set()

_CV_CAT_MAX_SCALE: dict[str, float] = {
    "KEYBOARD":   4.0,
    "MOUSE":      3.0,
    "MONITOR":    2.5,
    "SPEAKER":    2.5,
    "DESK_LAMP":  2.5,
    "DESK_SHELF": 2.0,
    "LIGHTING":   3.0,
}

_PREFERRED_POS = {
    "MONITOR":      {"rx": 0.50, "ry": 0.20},
    "DESK_SHELF":   {"rx": 0.50, "ry": 0.20},
    "KEYBOARD":     {"rx": 0.50, "ry": 0.62},
    "MOUSEPAD":     {"rx": 0.50, "ry": 0.65},
    "MOUSE":        {"rx": 0.70, "ry": 0.62},
    "SPEAKER":      {"rx": 0.25, "ry": 0.25},
    "DESK_LAMP":    {"rx": 0.12, "ry": 0.30},
    "DECO":         {"rx": 0.75, "ry": 0.35},
    "CLOCK":        {"rx": 0.80, "ry": 0.30},
    "LAPTOP_STAND": {"rx": 0.50, "ry": 0.45},
    "LIGHTING":     {"rx": 0.50, "ry": 0.08},
}

_MIN_FRONT_SIZE = {
    "MONITOR":   (220, 150),
    "KEYBOARD":  (180, 45),
    "MOUSE":     (70, 52),
    "SPEAKER":   (55, 55),
    "DESK_LAMP": (80, 120),
    "DECO":      (45, 45),
    "CLOCK":     (55, 40),
    "LIGHTING":  (200, 18),
}

_CAT_ASPECT_VALID: dict[str, tuple[float, float]] = {
    "KEYBOARD":  (2.0, 99.0),
    "MOUSE":     (0.5, 2.0),
    "MONITOR":   (0.9, 3.5),
    "SPEAKER":   (0.3, 2.5),
    "DESK_LAMP": (0.2, 3.0),
    "LIGHTING":  (5.0, 30.0),
}

_CONTACT_Y_OFFSET = {
    "KEYBOARD": 32,
    "MOUSE":    20,
}

_OVERLAP_TOLERANCE: dict = {
    frozenset({"MONITOR",    "KEYBOARD"}):   0.05,
    frozenset({"MONITOR",    "DESK_SHELF"}): 0.30,
    frozenset({"DESK_SHELF", "KEYBOARD"}):   0.40,
    frozenset({"KEYBOARD",   "MOUSEPAD"}):   0.50,
    frozenset({"MOUSE",      "MOUSEPAD"}):   0.60,
    frozenset({"MONITOR",    "LIGHTING"}):   0.40,
}
_DEFAULT_OVERLAP_THR = 0.10

_FRONT_HEIGHT_RATIO = {
    "MONITOR":      0.68,
    "KEYBOARD":     0.22,
    "MOUSE":        0.75,
    "MOUSEPAD":     0.25,
    "SPEAKER":      1.10,
    "DESK_LAMP":    1.70,
    "DESK_SHELF":   0.20,
    "LAPTOP_STAND": 0.40,
    "DECO":         0.90,
    "CLOCK":        0.90,
    "LIGHTING":     0.08,
}

_DESK_W_RATIO = {
    "KEYBOARD":     0.33,
    "MOUSE":        0.07,
    "MOUSEPAD":     0.55,
    "MONITOR":      0.55,
    "SPEAKER":      0.09,
    "DESK_LAMP":    0.06,
    "DESK_SHELF":   0.45,
    "LAPTOP_STAND": 0.22,
    "DECO":         0.07,
    "CLOCK":        0.08,
    "LIGHTING":     0.38,
}

_DINO_LABEL_TO_CATEGORY = {
    "keyboard":      "KEYBOARD",
    "mouse":         "MOUSE",
    "mouse pad":     "MOUSEPAD",
    "mousepad":      "MOUSEPAD",
    "monitor":       "MONITOR",
    "speaker":       "SPEAKER",
    "desk lamp":     "DESK_LAMP",
    "lamp":          "DESK_LAMP",
    "headset":       "HEADSET",
    "desk shelf":    "DESK_SHELF",
    "monitor riser": "DESK_SHELF",
    "laptop stand":  "LAPTOP_STAND",
    "clock":         "CLOCK",
    "cup":           "DECO",
    "mug":           "DECO",
    "light bar":     "LIGHTING",
    "screen bar":    "LIGHTING",
}

_RANKER_CAT_ID = {
    "MONITOR": 0, "KEYBOARD": 1, "MOUSE": 2, "MOUSEPAD": 3,
    "SPEAKER": 4, "DESK_LAMP": 5, "DESK_SHELF": 6,
    "LAPTOP_STAND": 7, "DECO": 8, "CLOCK": 9,
    "LIGHTING": 10,
}
# LIGHTING은 학습 샘플 없음 → ranker skip, rule_score만 사용
_RANKER_SKIP_CATS = {"MONITOR", "MOUSEPAD", "LIGHTING"}

_FRONT_CATS = {"KEYBOARD", "MOUSE", "MOUSEPAD"}
_BACK_CATS  = {"MONITOR", "SPEAKER", "DESK_LAMP", "DESK_SHELF", "LAPTOP_STAND", "DECO", "CLOCK", "LIGHTING"}

_REMOVAL_PROMPT = (
    "laptop. laptop computer. notebook computer. monitor. keyboard. mouse. "
    "mouse pad. mousepad. headset. cup. mug. book. books. book stack. "
    "notebook. notepad. paper. document. folder. file. binder. "
    "pen. pencil. pen holder. pencil holder. ruler. scissors. tape. "
    "phone. smartphone. tablet. speaker. desk lamp. lamp. "
    "clock. digital clock. diffuser. perfume bottle. vase. "
    "light bar. screen bar. monitor light."
)
