# =============================================================================
# config.py — 전체 프로젝트 설정값 중앙 관리
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── 네이버 쇼핑 API ──────────────────────────────────────────────────────────
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NAVER_SHOP_URL = "https://openapi.naver.com/v1/search/shop.json"

# ── PostgreSQL ────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "postgres"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

# ── HuggingFace ──────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")

# ── Jina CLIP v2 모델 ─────────────────────────────────────────────────────────
MODEL_NAME = "jinaai/jina-clip-v2"
EMBEDDING_DIM = 1024

# ── 카테고리별 검색어 (수집용: 넓고 포괄적으로 던짐) ──────────────────────────
CATEGORY_QUERIES = {
    "DESK" :       "책상",
    "MONITOR":      "모니터",
    "KEYBOARD":     "키보드",
    "MOUSE":        "마우스",
    "MONITOR_ARM":  "모니터암",
    "LAPTOP_STAND": "노트북 거치대",
    "MOUSEPAD":     "마우스패드 데스크매트",
    "DESK_SHELF":   "모니터 받침대 책상 선반",
    "LIGHTING":     "모니터 라이트바 스크린바",
    "SPEAKER":      "스피커",
    # ── 인테리어 소품 (네이버 카테고리: 시계 / 스탠드 / 기타 장식용품) ──────────
    "CLOCK":        "인테리어 시계 탁상시계",
    "DESK_LAMP":    "스탠드 조명 독서등",
    "DECO":         "인테리어 소품 장식 오브제",
    "HEADSET":      "헤드셋 헤드폰 게이밍 헤드셋",
    "HEADSET_STAND": "헤드셋 거치대 헤드폰 거치대",
}



# ── 카테고리 매핑 키워드 (분류/필터링용: 좁고 명확하게) ────────────────────────
# 우선순위: MONITOR_ARM > LAPTOP_STAND > DESK_SHELF 순으로 먼저 매칭
CATEGORY_KEYWORDS = {
    "DESK" :        ["책상", "데스크", "desk", "컴퓨터 책상", "사무용 책상",
                    "서재 책상", "1인용 책상", "게이밍 책상", "전동 책상", "모션데스크"],
    "MONITOR":      ["모니터", "monitor", "디스플레이", "울트라와이드"],
    "KEYBOARD":     ["키보드", "keyboard", "기계식", "무접점", "텐키리스"],
    "MOUSE":        ["마우스", "mouse", "버티컬 마우스", "트랙볼"],
    "MONITOR_ARM":  ["모니터암", "모니터 암", "싱글암", "듀얼암", "vesa", "베사"],
    "LAPTOP_STAND": ["노트북 거치대", "노트북 스탠드", "맥북 거치대", "랩탑 거치대"],
    "MOUSEPAD":     ["마우스패드", "mousepad", "데스크매트", "deskmat", "장패드"],
    "DESK_SHELF":   ["책상 선반", "데스크 선반", "모니터 받침대", "타공판", "desk shelf"],
    "LIGHTING":     ["라이트바", "스크린바", "모니터 조명", "데스크 램프", "lighting bar"],
    "SPEAKER":      ["스피커", "speaker", "사운드바", "pc 스피커"],
    "CLOCK":        ["시계", "탁상시계", "벽시계", "알람시계", "clock"],
    "DESK_LAMP":    ["스탠드", "독서등", "스탠드 조명", "desk lamp", "테이블 램프"],
    "DECO":         ["인테리어 소품", "오브제", "장식품", "데코", "소품", "캔들", "화분"],
    "HEADSET_STAND": ["헤드셋 거치대", "헤드폰 거치대", "헤드셋 스탠드", "헤드폰 스탠드",
                      "headset stand", "headphone stand"],
    "HEADSET":      ["헤드셋", "헤드폰", "headset", "headphone", "게이밍 헤드셋", "무선 헤드셋"],
}

# ── 카테고리별 표준 사이즈 (mm) ───────────────────────────────────────────────
DEFAULT_SIZES = {
    "DESK":             {"width_mm": 1200, "depth_mm": 600},
    "CHAIR":            {"width_mm": 680,  "depth_mm": 680},
    "MONITOR":          {"width_mm": 598,  "height_mm": 336},   # 27인치 16:9 패널 기준
    "KEYBOARD":         {"width_mm": 440,  "depth_mm": 150},
    "MOUSE":            {"width_mm": 70,   "depth_mm": 120},
    "MONITOR_ARM":      {"width_mm": 120,  "depth_mm": 500},
    "LAPTOP_STAND":     {"width_mm": 260,  "depth_mm": 230},
    "MOUSEPAD":         {"width_mm": 900,  "depth_mm": 400},
    "DESK_SHELF":       {"width_mm": 600,  "depth_mm": 200},
    "LIGHTING":         {"width_mm": 500,  "depth_mm": 30},
    "SPEAKER":          {"width_mm": 150,  "depth_mm": 150},
    "HEADSET":          {"width_mm": 180,  "depth_mm": 85},
}

# 사이즈 적재 대상 카테고리 (DESK_LAMP / SPEAKER 는 보류)
SIZE_RELEVANT = {"MONITOR", "KEYBOARD", "MOUSE", "HEADSET", "MOUSEPAD"}

# rembg 배경제거를 스킵할 카테고리 (납작한 면 제품 등 rembg가 오히려 해가 되는 경우)
SKIP_REMBG_CATEGORIES = {"MOUSEPAD", "DESK_SHELF"}


# ── 카테고리별 제외 키워드 (책상에 올릴 수 없는 상품 필터링) ─────────────────────
CATEGORY_EXCLUDE_KEYWORDS = {
    "DESK" :       ["식탁", "다이닝", "카페", "좌식", "접이식", "야외", "캠핑",
                    "소파 테이블", "거실 탁자", "티테이블", "업소용", "화장대", "콘솔", "어린이용"],
    "MONITOR":      ["TV", "tv", "티비", "벽걸이", "사이니지", "업소용", "차량용"],
    "KEYBOARD":     ["커버", "스킨", "보호필름", "청소", "키캡"],
    "MOUSE":        ["마우스피트", "마우스 번지", "차량용", "케이스"],
    "MONITOR_ARM":  ["TV", "tv", "벽걸이", "천장", "바닥"],
    "LAPTOP_STAND": ["차량용", "침대", "쇼파", "소파", "독서대"],
    "MOUSEPAD":     ["바닥", "욕실", "주방", "차량", "요가"],
    "DESK_SHELF":   ["벽", "주방", "욕실", "신발", "창고", "야외"],
    "LIGHTING":     ["차량용", "실외", "수중", "방수", "가로등", "천장"],
    "SPEAKER":      ["차량용", "방수", "야외", "공연용", "업소용"],
    "CLOCK":        ["차량용", "주방", "욕실", "야외", "수중", "방수", "벽시계"],
    "DESK_LAMP":    ["차량용", "실외", "수중", "방수", "가로등", "천장", "야외"],
    "DECO":         ["식품", "먹는", "차량용", "야외", "공사", "업소용"],
    "HEADSET_STAND": ["차량용", "야외", "모터사이클", "헬멧"],
    "HEADSET":      ["이어폰", "earphone", "이어버드", "earbuds", "보청기", "차량용", "헬멧", "마이크 단품", "케이블", "커버", "파우치", "이어패드", "쿠션", "거치대", "스탠드"],
}

# ── 카테고리 한글 레이블 ──────────────────────────────────────────────────────
CATEGORY_LABELS = {
    "DESK":         "책상",
    "MONITOR":      "모니터",
    "KEYBOARD":     "키보드",
    "MOUSE":        "마우스",
    "MONITOR_ARM":  "모니터암",
    "LAPTOP_STAND": "노트북 거치대",
    "MOUSEPAD":     "마우스패드",
    "DESK_SHELF":   "데스크 선반",
    "LIGHTING":     "라이트바/스크린바",
    "SPEAKER":      "스피커",
    "CLOCK":        "시계",
    "DESK_LAMP":    "데스크 조명",
    "DECO":         "데코 소품",
    "HEADSET_STAND": "헤드셋 거치대",
    "HEADSET":      "헤드셋/헤드폰",
}

# ── 카테고리 입력 별칭 (CLI 파싱용) ──────────────────────────────────────────
CATEGORY_ALIASES = {
    "desk": "DESK",
    "책상": "DESK",
    "monitor": "MONITOR",
    "모니터": "MONITOR",
    "keyboard": "KEYBOARD",
    "키보드": "KEYBOARD",
    "mouse": "MOUSE",
    "마우스": "MOUSE",
    "monitorarm": "MONITOR_ARM",
    "monitor_arm": "MONITOR_ARM",
    "모니터암": "MONITOR_ARM",
    "laptopstand": "LAPTOP_STAND",
    "laptop_stand": "LAPTOP_STAND",
    "노트북거치대": "LAPTOP_STAND",
    "mousepad": "MOUSEPAD",
    "마우스패드": "MOUSEPAD",
    "desk_shelf": "DESK_SHELF",
    "deskshelf": "DESK_SHELF",
    "데스크선반": "DESK_SHELF",
    "lighting": "LIGHTING",
    "라이트바": "LIGHTING",
    "screenbar": "LIGHTING",
    "스크린바": "LIGHTING",
    "speaker": "SPEAKER",
    "스피커": "SPEAKER",
    "clock": "CLOCK",
    "시계": "CLOCK",
    "desk_lamp": "DESK_LAMP",
    "desklamp": "DESK_LAMP",
    "lamp": "DESK_LAMP",
    "조명": "DESK_LAMP",
    "데스크조명": "DESK_LAMP",
    "deco": "DECO",
    "오브제": "DECO",
    "소품": "DECO",
    "headset_stand": "HEADSET_STAND",
    "headsetstand": "HEADSET_STAND",
    "headphone_stand": "HEADSET_STAND",
    "headphonestand": "HEADSET_STAND",
    "헤드셋거치대": "HEADSET_STAND",
    "헤드폰거치대": "HEADSET_STAND",
    "헤드셋스탠드": "HEADSET_STAND",
    "헤드폰스탠드": "HEADSET_STAND",
    "headset": "HEADSET",
    "headphone": "HEADSET",
    "헤드셋": "HEADSET",
    "헤드폰": "HEADSET",
}

# ── KEYBOARD/MOUSE 카테고리 세트 ──────────────────────────────────────────────
KEYBOARD_MOUSE_CATEGORIES = {"KEYBOARD", "MOUSE"}

# ── 검색 기본값 ───────────────────────────────────────────────────────────────
TITLE_BOOST_WEIGHT = 1.5
DEFAULT_SEARCH_MODE = "image_only"
DEFAULT_IMAGE_WEIGHT = 1.0
DEFAULT_TEXT_WEIGHT = 0.0
DEFAULT_TOP_K = 10

# ── 셋업 추천 기본값 ──────────────────────────────────────────────────────────
DEFAULT_CANDIDATE_PER_CATEGORY = 5
DEFAULT_SETUP_TOP_K = 3
