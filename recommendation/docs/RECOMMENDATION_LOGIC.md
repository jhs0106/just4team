# 추천 로직 설명

데스크테리어 추천 시스템의 엔진(`deskterior/recommender/engine.py`) 동작 방식을 설명합니다.

---

## 전체 흐름

```
테마 + 예산 입력
    ↓
카테고리별 후보 상품 검색 (Jina CLIP v2 + pgvector)
+ 노이즈 필터 (ABS_MIN_PRICE, 제외 키워드)
+ 가용공간 필터 (space_constraints, ai-server 연동 시)
    ↓
개별 상품 점수화 (ThemeEvidence, ImageSim, TextSim → ItemScore)
    ↓
2단계 Beam Search (beam_size=50) — 예산 내 최적 번들 탐색
  Phase 1: 필수 3종 (MONITOR, KEYBOARD, MOUSE)
  Phase 2: 선택 상품 (MOUSEPAD, SPEAKER, DESK_LAMP, HEADSET)
    ↓
SetupScore / FinalScore 계산
    ↓
ThemeGate → Pareto 필터 → 다양성 보정
    ↓
TOP 1 셋업 번들 반환
```

---

## 1. 테마 + 예산 입력

| 항목 | 선택지 |
|---|---|
| 테마 | `white` (화이트 클린) / `black` (블랙 다크) / `gaming` (RGB 게이밍) / `wood` (우드 내추럴) |
| 예산 | 원 단위 정수 (예: 1000000) |

테마별 설정은 `deskterior/recommender/config.py`의 `THEME_PRESETS`에 정의되어 있습니다.

---

## 2. 후보 상품 검색

카테고리별로 테마에 맞는 검색 쿼리를 `THEME_CATEGORY_QUERIES`에서 가져와 Jina CLIP v2로 임베딩 후 pgvector로 검색합니다.

예시 (gaming 테마):

| 카테고리 | 검색 쿼리 |
|---|---|
| MONITOR | "게이밍 모니터 144Hz 165Hz" |
| KEYBOARD | "RGB 게이밍 기계식 키보드" |
| MOUSE | "게이밍 마우스 DPI RGB 경량" |
| MOUSEPAD | "RGB 게이밍 장패드 마우스패드" |

```sql
-- 이미지 유사도 60%, 텍스트 유사도 40% 가중치로 정렬
ORDER BY (0.60 * (embedding_img <=> query_vec)
        + COALESCE(0.40 * (embedding_txt <=> query_vec), 0.20)) ASC
LIMIT 50
```

카테고리 구분:
- **필수(MANDATORY)**: `MONITOR`, `KEYBOARD`, `MOUSE`
- **선택(OPTIONAL)**: `MOUSEPAD`, `SPEAKER`, `DESK_LAMP`, `HEADSET`

### 노이즈 필터 (ABS_MIN_PRICE + 제외 키워드)

검색 결과에서 명백한 노이즈를 제거합니다.

```python
ABS_MIN_PRICE = {
    "MONITOR": 70_000,   # 7만원 미만 모니터 제외
    "KEYBOARD": 15_000,
    "MOUSE": 8_000,
    ...
}
# + CATEGORY_EXCLUDE_KEYWORDS: TV, 차량용, 마우스피트, 이어폰 등
```

### 가용공간 필터 (ai-server 연동 시)

ai-server가 탑뷰 사진을 분석해 카테고리별 가용 공간을 계산하고 `space_constraints`로 전달하면, SQL WHERE절에 사이즈 필터가 추가됩니다.

```sql
-- MONITOR에 [820, 150] 제약이 있으면:
AND ((metadata->>'width_mm') IS NULL OR (metadata->>'width_mm')::int <= 820)
AND ((metadata->>'depth_mm') IS NULL OR (metadata->>'depth_mm')::int <= 150)
```

metadata가 없는 상품은 보수적으로 통과 (크기 모르면 허용).

---

## 3. 개별 상품 점수화

### ThemeEvidence

상품명과 태그에서 테마 키워드 매칭으로 테마 부합도를 측정합니다.

```
ThemeEvidence = min(1.0, positive_hits / 2.0)
              - min(0.6, negative_hits × 0.20)
              + CategoryThemeBonus
```

- `positive_keywords`: 테마와 맞는 키워드 (예: gaming → "rgb", "144hz", "게이밍")
- `negative_keywords`: 테마와 반대되는 키워드 (예: gaming → "우드", "북유럽", "파스텔")
- `CategoryThemeBonus`: 특정 카테고리+테마 조합에 추가 보너스 (예: gaming 모니터에 "144hz" 포함 시 +0.20)

### ItemScore

```
ItemScore = 0.35 × ImageSim + 0.25 × TextSim + 0.40 × ThemeEvidence
```

- **ImageSim**: pgvector 이미지 임베딩 코사인 유사도
- **TextSim**: pgvector 텍스트 임베딩 코사인 유사도
- **ThemeEvidence**: 테마 키워드 직접 매칭 점수

> ValueScore는 SetupScore의 ValueEfficiency 항목(5% 비중)에서만 사용되며 ItemScore에는 포함되지 않습니다.

### BudgetUsageScore

예산 활용도를 나타내는 보조 점수입니다.

```
BudgetUsageScore = min((총가격 / 예산) / 0.85, 1.0)
```

예산의 85%를 사용하면 1.0 만점. 이 점수는 FinalScore와 quick_score에 반영됩니다.

---

## 4. 2단계 Beam Search

### Phase 1 — 필수 3종 조합

MONITOR → KEYBOARD → MOUSE 순서로 빔을 확장합니다.

**가격대별 후보 선택 (`_select_tiered_candidates`)**

item_score 상위만 쓰면 싼 상품만 뽑히므로, 가격대를 세 구간으로 나눠 균형 있게 선택합니다.

```
저가 40% / 중가 30% / 고가 30%
각 구간에서 item_score 상위 순으로 선발
```

**빔 가지치기 (`_prune_beams`)**

```
quick_score = 0.85 × quality + 0.15 × BudgetUsageScore(현재까지)
```

- quality: 필수 상품 item_score 평균 + 선택 상품 보너스
- 상위 80%는 quick_score 기준, 나머지 20%는 예산 구간별(50~70%, 70~85%, 85%+) 대표 빔 보존
- 비싼 조합이 초반에 탈락하지 않도록 예산 사용률을 빔 생존에 반영

### Phase 2 — 선택 상품 추가

MOUSEPAD → SPEAKER → DESK_LAMP → HEADSET 순서로 각 카테고리를 시도합니다.

**OptionalGain**

```
OptionalGain = 0.30 × ItemScore
             + 0.25 × ThemeEvidence
             + 0.20 × RoleCompatibility
             + 0.10 × Priority
             + 0.15 × BudgetGain
             - 부정키워드 패널티(0.15)
```

- **RoleCompatibility**: 기존 담긴 상품과의 카테고리 쌍 가중 평균 테마 일치도
- **Priority**: 테마별로 정해진 선택 상품 우선순위 (예: gaming 테마에서 HEADSET=0.90)
- **BudgetGain**: 이 상품 추가 시 BudgetUsageScore 증가분

OptionalGain이 임계값 이상일 때만 번들에 추가됩니다.

| 카테고리 | 임계값 |
|---|---|
| MOUSEPAD | 0.45 |
| SPEAKER | 0.50 |
| DESK_LAMP | 0.55 |
| HEADSET | 0.55 |

---

## 5. SetupScore / FinalScore

```
SetupScore = 0.35 × SetupThemeEvidence
           + 0.20 × AvgItemScore
           + 0.20 × RoleAwareCompatibility
           + 0.15 × MandatoryCoverage
           + 0.05 × OptionalUsefulness
           + 0.05 × ValueEfficiency
```

```
FinalScore = SetupScore × (0.70 + 0.30 × BudgetUsageScore)
```

예산을 85% 이상 사용하면 FinalScore = SetupScore, 예산을 전혀 안 쓰면 SetupScore × 0.70.

| 항목 | 설명 |
|---|---|
| `SetupThemeEvidence` | 번들 내 전체 상품 ThemeEvidence 평균 |
| `AvgItemScore` | 번들 내 전체 상품 ItemScore 평균 |
| `RoleAwareCompatibility` | 카테고리 쌍의 테마 일치도 가중 평균 |
| `MandatoryCoverage` | 필수 3종 포함 비율 (정상이면 항상 1.0) |
| `OptionalUsefulness` | 선택 상품 테마 우선순위 × 테마 증거 평균 |
| `ValueEfficiency` | 번들 내 ValueScore 평균 (가성비) |
| `BudgetUsageScore` | 예산 85% 사용 시 1.0 만점 |

### RoleAwareCompatibility 가중치 (ROLE_PAIR_WEIGHTS)

| 카테고리 쌍 | 가중치 |
|---|---|
| KEYBOARD ↔ MOUSE | 1.00 |
| KEYBOARD ↔ MOUSEPAD | 0.95 |
| MOUSE ↔ MOUSEPAD | 0.95 |
| MONITOR ↔ SPEAKER | 0.70 |
| MONITOR ↔ DESK_LAMP | 0.60 |
| KEYBOARD ↔ HEADSET | 0.50 |
| MOUSE ↔ HEADSET | 0.50 |

---

## 6. 후처리

### ThemeGate

최종 추천 전 테마 부합도 최소 기준을 검사합니다.

| 테마 | 필수 카테고리 최소 매칭 수 | 평균 ThemeEvidence 최소값 |
|---|---|---|
| white | 1개 이상 | 0.58 |
| black | 2개 이상 | 0.60 |
| gaming | 2개 이상 | 0.65 |
| wood | 1개 이상 (선택상품 있는 경우) | 0.53 |

기준 미달 번들은 `ALLOW_THEME_GATE_FALLBACK = True`인 경우 FinalScore × 0.75 패널티 후 허용.

### Pareto 필터

더 비싸면서 FinalScore가 낮거나 같은 번들을 제거합니다.

### 다양성 보정

상품 구성이 60% 이상 겹치는 번들은 중복으로 보고 하나만 남깁니다.

---

## API 서버 (ai-server 연동)

`deskterior/api/server.py`를 FastAPI 서버로 실행하면 ai-server에서 HTTP로 호출할 수 있습니다.

```bash
uvicorn deskterior.api.server:app --host 0.0.0.0 --port 8001
```

**요청 예시 (`POST /recommend`)**

```json
{
  "theme": "white",
  "budget": 1000000,
  "image_base64": "...",
  "space_constraints": {
    "MONITOR": [820, 150],
    "KEYBOARD": [500, 280]
  }
}
```

**응답 예시**

```json
{
  "theme": "white",
  "budget": 1000000,
  "setup": {
    "setup_score": 0.792,
    "final_score": 0.676,
    "total_price": 435360,
    "items": {
      "MONITOR": {"id": 123, "title": "...", "lprice": 299660, ...},
      "KEYBOARD": {...},
      "MOUSE": {...}
    }
  }
}
```

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `deskterior/recommender/engine.py` | 추천 알고리즘 전체 구현 |
| `deskterior/recommender/config.py` | 테마 프리셋, 카테고리, threshold, ROLE_PAIR_WEIGHTS |
| `deskterior/api/server.py` | FastAPI 추천 API 서버 |
| `deskterior/retrieval/searcher.py` | Jina CLIP v2 임베딩 + pgvector 검색 |
| `deskterior/core/config.py` | DB 연결 설정 |
