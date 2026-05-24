# 추천 로직 설명

데스크테리어 추천 시스템의 신규 엔진(`deskterior/recommender/engine.py`) 동작 방식을 설명합니다.

---

## 전체 흐름

```
테마 + 예산 입력
    ↓
카테고리별 후보 상품 검색 (Jina CLIP v2 + pgvector)
    ↓
개별 상품 점수화 (ThemeEvidence, ImageSim, TextSim, ValueScore → ItemScore)
    ↓
Beam Search (beam_size=50) — 예산 내 최적 번들 탐색
    ↓
번들 점수화 (SetupScore)
    ↓
TOP 3 셋업 번들 반환
```

---

## 1. 테마 + 예산 입력

`python main.py recommend` 실행 시 두 가지를 입력받습니다.

| 항목 | 선택지 |
|---|---|
| 테마 | `white` (화이트 클린) / `black` (블랙 다크) / `gaming` (RGB 게이밍) / `wood` (우드 내추럴) |
| 예산 | 원 단위 정수 (예: 1500000) |

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

카테고리: 필수(MANDATORY) — `MONITOR`, `KEYBOARD`, `MOUSE` / 선택(OPTIONAL) — `MOUSEPAD`, `SPEAKER`, `DESK_LAMP`, `HEADSET`

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
ItemScore = 0.30 × ImageSim
          + 0.25 × TextSim
          + 0.35 × ThemeEvidence
          + 0.10 × ValueScore
```

- `ImageSim`: pgvector 이미지 임베딩 코사인 유사도
- `TextSim`: pgvector 텍스트 임베딩 코사인 유사도
- `ValueScore`: 카테고리 내 min-max 정규화된 가성비 점수 (`relevance / log(1 + price)`)

---

## 4. Beam Search

beam_size=50을 유지하며 카테고리를 순서대로 추가해 예산 내 최적 번들을 탐색합니다.

```
초기 상태: 빈 번들 (beam_size=1)
    ↓
각 카테고리 후보를 확장 → 예산 초과 가지 제거
    ↓
quick_score 기준 상위 beam_size=50 상태만 유지
    ↓
모든 카테고리 처리 후 SetupScore 계산
```

### OptionalGain

선택 카테고리(MOUSEPAD 등)는 번들에 추가했을 때 실제 점수 향상이 임계값 이상일 때만 포함됩니다.

```
OptionalGain = 0.35 × ItemScore
             + 0.35 × ThemeEvidence
             + 0.20 × RoleCompatibility
             + 0.10 × ThemeOptionalPriority
             - conflict_penalty
```

임계값 예시 (config.py의 `OPTIONAL_GAIN_THRESHOLD`):

| 테마 | MOUSEPAD | HEADSET | SPEAKER | DESK_LAMP |
|---|---|---|---|---|
| white | 0.30 | 0.42 | 0.38 | 0.34 |
| gaming | 0.28 | 0.30 | 0.36 | 0.42 |

---

## 5. SetupScore (번들 점수)

```
SetupScore = 0.35 × SetupThemeEvidence
           + 0.20 × AvgItemScore
           + 0.20 × RoleAwareCompatibility
           + 0.15 × MandatoryCoverage
           + 0.05 × OptionalUsefulness
           + 0.05 × ValueEfficiency
```

| 항목 | 설명 |
|---|---|
| `SetupThemeEvidence` | 번들 내 전체 상품의 ThemeEvidence 평균 |
| `AvgItemScore` | 번들 내 전체 상품의 ItemScore 평균 |
| `RoleAwareCompatibility` | 카테고리 쌍의 테마 일치도 가중 평균 (키보드-마우스 등 핵심 쌍에 높은 가중치) |
| `MandatoryCoverage` | 필수 카테고리(MONITOR, KEYBOARD, MOUSE) 포함 비율 |
| `OptionalUsefulness` | 선택 카테고리 상품의 테마 우선순위 점수 |
| `ValueEfficiency` | 번들 내 ValueScore 평균 |

### RoleAwareCompatibility 가중치 (ROLE_PAIR_WEIGHTS)

| 카테고리 쌍 | 가중치 |
|---|---|
| KEYBOARD ↔ MOUSE | 1.00 |
| KEYBOARD ↔ MOUSEPAD | 0.95 |
| MOUSE ↔ MOUSEPAD | 0.95 |
| MONITOR ↔ SPEAKER | 0.60 |
| KEYBOARD ↔ HEADSET | 0.55 |
| MOUSE ↔ HEADSET | 0.55 |
| MONITOR ↔ DESK_LAMP | 0.50 |

---

## 6. ThemeGate (통과 기준)

최종 추천 전 테마 부합도 최소 기준을 검사합니다.

예시 (config.py의 `mandatory_rule`):

| 테마 | 필수 카테고리 최소 매칭 수 | 평균 ThemeEvidence 최소값 |
|---|---|---|
| white | 1개 이상 | 0.58 |
| black | 2개 이상 | 0.60 |
| gaming | 2개 이상 | 0.65 |
| wood | 1개 이상 | 0.53 |

기준 미달 번들은 `ALLOW_THEME_GATE_FALLBACK = True`인 경우 폴백 허용.

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `deskterior/recommender/engine.py` | 추천 알고리즘 전체 구현 |
| `deskterior/recommender/config.py` | 테마 프리셋, 카테고리, threshold, ROLE_PAIR_WEIGHTS |
| `deskterior/retrieval/searcher.py` | Jina CLIP v2 임베딩 + pgvector 검색 |
| `deskterior/core/config.py` | DB 연결 설정, 카테고리 키워드 |
