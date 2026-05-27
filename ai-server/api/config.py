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
    "MOUSEPAD":     13,
    "KEYBOARD":     15,
    "DESK_SHELF":   20,
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
    # KEYBOARD/MOUSE/MOUSEPAD: ry 0.55. 책상 detection이 윗면 + 다리까지 잡는 경우
    # ry 0.72는 책상 밖(다리 사이)으로 떨어짐. 0.55는 책상 윗면 중앙-앞쪽이라 안정적.
    "KEYBOARD":     {"rx": 0.50, "ry": 0.55},
    "MOUSEPAD":     {"rx": 0.50, "ry": 0.55},
    "MOUSE":        {"rx": 0.70, "ry": 0.55},
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
    # KEYBOARD: DB 마케팅 컷이 위에서 본 사선(ar≈1.3~1.8)도 많아 (2.0, 99.0)이면 거의 항상
    # CV fallback에 떨어져 SD/ControlNet/IP-Adapter 통과 안 됨. (1.2, 99.0)으로 완화.
    "KEYBOARD":  (1.2, 99.0),
    "MOUSE":     (0.5, 2.0),
    "MONITOR":   (0.9, 3.5),
    # SPEAKER: DB가 전부 사운드바형(가로로 긴 ar 3~5). (0.3, 2.5)면 사운드바가 CV fallback에
    # 떨어져 SD 안 거침. (0.3, 6.0)으로 완화 — 사운드바도 ControlNet+IP-Adapter 통과.
    "SPEAKER":   (0.3, 6.0),
    "DESK_LAMP": (0.2, 3.0),
    "LIGHTING":  (5.0, 30.0),
}

# 카테고리별 ry(top-view depth axis) 안전 범위.
# ranker가 산출한 ry를 front-view bbox로 변환할 때 이 범위로 clamp하여
# 비현실적 배치(예: 모니터가 책상 앞쪽 끝에 옴)를 방지함.
# ry: 0=책상 back edge(벽 쪽), 1=책상 front edge(사용자 쪽)
_CAT_RY_RANGE: dict[str, tuple[float, float]] = {
    "MONITOR":      (0.18, 0.35),   # 책상 뒤 1/3 (모니터는 보통 책상 뒤)
    "DESK_SHELF":   (0.15, 0.32),   # 모니터 받침대 — 모니터와 비슷한 깊이
    "LIGHTING":     (0.05, 0.20),   # 모니터 위쪽 (벽 가까이)
    "DESK_LAMP":    (0.20, 0.50),   # 책상 뒤~중간
    "SPEAKER":      (0.18, 0.45),   # 책상 뒤~중간
    "LAPTOP_STAND": (0.30, 0.55),   # 책상 중간
    "DECO":         (0.20, 0.55),   # 자유로움
    "CLOCK":        (0.18, 0.45),   # 자유로움
    # KEYBOARD/MOUSE/MOUSEPAD: 책상 detection이 윗면+다리까지 잡으면 ry 0.62~0.85는
    # 책상 밖(다리 사이)으로 떨어짐. 0.35~0.70으로 보수적 — 책상 중앙~앞쪽 안전 범위.
    "KEYBOARD":     (0.35, 0.70),
    "MOUSE":        (0.35, 0.70),
    "MOUSEPAD":     (0.35, 0.70),
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

# 제품 입체감 분류 (3-tier).
# flat:      책상 위에 완전히 누워있는 형태. 책상 plane perspective warp 적용.
# semi_flat: 약간의 부피 있음. warp 미적용, upright prompt도 미적용.
# upright:   수직으로 서 있는 형태. warp 금지 + "standing upright" prompt 추가.
_PRODUCT_FORM_TIER = {
    "KEYBOARD":     "flat",
    "MOUSEPAD":     "flat",
    "MOUSE":        "flat",        # semi_flat → flat (2026-05-25): warp으로 perspective 매칭
    "LAPTOP_STAND": "semi_flat",
    "MONITOR":      "upright",
    "SPEAKER":      "upright",
    "DESK_LAMP":    "upright",
    "DESK_SHELF":   "upright",
    "CLOCK":        "upright",
    "LIGHTING":     "upright",
    "DECO":         "upright",
}

# 하위 호환: 기존 코드가 _PRODUCT_IS_FLAT_ON_DESK 참조 시 'flat' tier만 True로 매핑.
_PRODUCT_IS_FLAT_ON_DESK = {
    cat: (tier == "flat") for cat, tier in _PRODUCT_FORM_TIER.items()
}

# 기본 책상 카메라 pitch angle (도). 사용자 책상 사진의 일반적 각도 가정.
# 향후 depth map plane fitting으로 자동 추정 가능 (TODO).
_DEFAULT_DESK_TILT_DEG = 30.0

# 카테고리별 warp depression angle override.
# 낮을수록 더 압축 (수평에 가까운 평면). 높을수록 덜 압축 (입체감 보존).
#   MOUSEPAD: 가장 납작 → 20° (h를 sin(20°)=0.34로 압축)
#   KEYBOARD: 기본 30° (h를 0.50로 압축)
#   MOUSE:    곡면 있음 → 45° (h를 0.71로 압축, 마우스 형태 보존)
#   LAPTOP_STAND: semi-flat (warp 없음, 이 값 무시됨)
_CAT_TILT_DEG = {
    "MOUSEPAD":     20.0,
    "KEYBOARD":     30.0,
    "MOUSE":        45.0,
    "LAPTOP_STAND": 50.0,
}

# 카테고리별 depth shading 강도 (post-blend AO 효과).
# 제품 silhouette 하단을 norm_y에 비례해 darken — 입체감 illusion.
# 값 = 바닥에서의 최대 darken 비율 (0.15 = 15% 어둡게).
_CAT_DEPTH_SHADING = {
    "MONITOR":      0.18,   # 큰 수직물, 강한 음영
    "SPEAKER":      0.18,
    "DESK_LAMP":    0.15,
    "DESK_SHELF":   0.15,
    "MOUSE":        0.12,   # 곡면 마우스, 중간 음영
    "KEYBOARD":     0.10,   # 평면, 약한 음영
    "MOUSEPAD":     0.05,   # 거의 평면
    "LAPTOP_STAND": 0.15,
    "CLOCK":        0.12,
    "DECO":         0.12,
    "LIGHTING":     0.05,
}

# upright 제품 prompt에 추가할 문구 — 책상에 세워 서 있음을 명시.
_UPRIGHT_PROMPT_TOKEN = (
    "standing upright on desk, front view, visible front face, base touching desk surface, "
)

# === LaMa removal prompt 정책 ===
# Grounding DINO로 검출할 "책상 위 제거 대상 물체" 텍스트 목록.
# 단어 추가 시 false-positive 위험 평가 필수:
#   - 일반명사("desk", "wall" 등): 책상 자체/벽 잡힐 위험 → 금지
#   - 가구류("shelf", "chair" 등): 옆 가구 잡힐 위험 → max_area_ratio 필터로 1차 방어
#   - 합성어/특수어("light bar"): DINO가 "light"만 매칭하여 책상 밝은 영역 잡을 위험
# 안전하게 후보 단어 추가 후 desk_image.jpg/desk_image2.jpg로 검출 결과 검증할 것.
# 검증 안 된 단어는 추가하지 말 것.
_REMOVAL_PROMPT = (
    # PC 및 컴퓨터 주변기기
    "laptop. laptop computer. notebook computer. monitor. keyboard. mouse. "
    "mouse pad. mousepad. headset. "
    # 책상 위 일상 소품
    "cup. mug. book. books. book stack. "
    "notebook. notepad. paper. document. folder. file. binder. "
    "pen. pencil. pen holder. pencil holder. ruler. scissors. tape. "
    # 전자기기
    "phone. smartphone. tablet. speaker. desk lamp. lamp. "
    # 장식 소품
    "clock. digital clock. diffuser. perfume bottle. vase."
    # 주의: "light bar", "screen bar", "monitor light"는 false-positive 위험으로 제외.
    # LIGHTING 카테고리는 사용자가 입력으로 명시 시에만 생성, 기존 물체 제거 대상 아님.
)

# === Tabletop bbox 정책 ===
# 빈 책상 모드에서 front-view tabletop 영역을 잡을 때의 sanity 임계값.
# top-view anchor를 front-view에 투영할 때 이 영역이 기준이 되므로 검증 필수.
#
# 과거 hard clamp(fv_dy1=max(0.45h), fv_dy2=min(0.72h))는 desk_image2 한 장에 맞춘 보정으로,
# 다른 구도 사진(책상이 화면 더 위/아래에 있거나 더 큰 경우)에서 실제 책상 하단을 잘라먹어
# placement 좌표계를 망쳤음. 기본 OFF, flag로만 활성화.
TABLETOP_BBOX_HARD_CLAMP_ENABLED = False

# tabletop 검증 임계값.
#   너무 얇음: edge band만 잡힌 케이스. fv_dh가 image_h 대비 10% 미만 또는 80px 미만
#   너무 두꺼움: 책상 전체/다리/바닥까지 포함된 케이스. image_h 대비 45% 초과
#   너무 아래: y2가 이미지 바닥에 너무 가까우면 다리/바닥 영역 포함 가능성
#   너무 가로형: fv_dw/fv_dh가 18 초과면 얇은 edge만 잡은 것
TABLETOP_MIN_HEIGHT_RATIO         = 0.10
TABLETOP_MIN_HEIGHT_PX            = 80
# 0.45 → 0.65: desk_image2.jpg 한 장 기준 fit. 사용자 책상이 두꺼우면(다리까지 포함)
# 0.45를 자주 초과해 sam2_override로 폴백 발생. 0.65는 더 관대.
TABLETOP_MAX_HEIGHT_RATIO         = 0.65
TABLETOP_MIN_BOTTOM_MARGIN_RATIO  = 0.03
TABLETOP_MAX_WIDTH_HEIGHT_RATIO   = 18.0

# LightGBM ranker가 final_score에 기여하는 비중.
# layout_ranker.pkl이 특정 책상(desk_image2.jpg) 기반 합성 데이터로 학습된 흔적 — 모든
# 사진에서 ry=0.64를 매번 best로 추천하는 패턴. 일반 책상에 대한 generic 동작을 위해
# 0.0으로 비활성화 (rule_score만 사용). ranker 재학습 후엔 0.3 정도로 복귀 가능.
# final_score = rule_score × (1 - W) + learned_score × W
RANKER_WEIGHT = 0.0
