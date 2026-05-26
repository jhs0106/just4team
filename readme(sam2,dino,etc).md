# AI Server: Deskterior Vision Pipeline

본 AI Server는 사용자 책상 이미지를 기반으로 책상 상판, 기존 객체, 가용공간을 분석하고, 이후 제품 추천 및 이미지 생성/합성 단계에서 활용할 수 있는 mask, 좌표, 공간 정보를 생성하는 역할을 수행함

주요 기능은 다음과 같다.

```text
1. Grounding DINO 기반 객체 검출
2. SAM2 기반 객체/책상 상판 마스킹
3. 책상 상판 영역(desk_mask) 자동 생성
4. 기존 객체 점유 영역(occupied_mask) 생성
5. 가용공간(available_space) 계산
6. LaMa 기반 객체 제거(inpainting)
7. 제품 배치 및 합성 파트 연동용 JSON/이미지 출력
```

---

## 1. 전체 파이프라인 개요

본 파이프라인은 `front-view image`와 `top-view image`를 모두 입력으로 사용함

```text
front-view image
→ 사용자에게 보여지는 최종 결과 기준 이미지
→ 객체 제거, 이미지 생성/합성, 배치 결과 시각화에 사용

top-view image
→ 책상 상판과 객체 위치를 위에서 분석하기 위한 이미지
→ 가용공간 계산, 제품 배치 후보 영역 산출에 사용
```

전체 처리 흐름은 다음과 같음

```text
입력 이미지 로드
↓
top-view 객체 검출
↓
top-view 객체 mask 생성
↓
front-view 객체 검출
↓
front-view 객체 mask 생성
↓
책상 상판 자동 검출
↓
desk_mask 생성 및 후처리
↓
occupied_mask 생성
↓
available_space = desk_mask - occupied_mask 계산
↓
remove_mask 생성
↓
LaMa 기반 객체 제거
↓
제품 추천/배치/합성 파트로 넘길 결과 저장
```

---

## 2. Grounding DINO 기반 객체 검출

Grounding DINO는 텍스트 프롬프트 기반 객체 검출 모델로 사용
사용자는 검출하고 싶은 책상 위 객체를 prompt 형태로 입력함

예시 prompt:

```text
monitor. laptop. keyboard. mouse. mouse pad. cup. mug. book. notebook. notepad. paper. document. diary. planner. lamp. desk lamp. lamp base. plant. plant pot. pot. clock. digital clock. speaker. bottle. tissue. cable. remote control.
```

Grounding DINO의 역할은 다음과 같음

```text
입력 이미지에서 prompt에 해당하는 객체 후보를 bounding box 단위로 검출
각 객체에 label, score, box_xyxy 정보 부여
검출 결과를 SAM2의 box prompt 입력으로 전달
```

검출 결과는 다음 파일로 저장

```text
top_detection_result.png
front_detection_result.png
detection_result.png
```

---

## 3. SAM2 기반 객체 마스킹

Grounding DINO는 bounding box만 생성하므로, 실제 객체 영역을 pixel 단위로 분리하기 위해 SAM2를 사용한다.

처리 방식:

```text
Grounding DINO bbox
→ SAM2 box prompt 입력
→ 객체별 segmentation mask 생성
→ 객체별 mask 후처리
→ 하나의 occupied_mask로 병합
```

객체 mask 생성 후에는 다음 후처리를 적용함

```text
dilate: mask 경계 확장
close: 작은 구멍 및 경계 불연속 보정
merge: 객체별 mask를 하나의 occupied_mask로 병합
```

생성되는 주요 mask는 다음과 같음

```text
raw_occupied_mask.png
processed_occupied_mask.png
sam2_mask_overlay.png
top_raw_occupied_mask.png
top_processed_occupied_mask.png
top_sam2_mask_overlay.png
front_raw_occupied_mask.png
front_processed_occupied_mask.png
front_sam2_mask_overlay.png
```

---

## 4. 책상 상판 자동 검출 및 desk_mask 생성

책상 위 가용공간을 계산하기 위해서는 먼저 책상 상판 영역을 알아야 한다.
이를 위해 별도의 desk prompt를 사용하여 Grounding DINO가 책상 상판 후보 bbox를 검출하고, 해당 bbox를 SAM2에 입력하여 `desk_mask`를 생성한다.

desk prompt 예시:

```text
tabletop. desk top surface. wooden desk surface. flat desk surface.
```

기본 흐름:

```text
Grounding DINO로 책상/상판 bbox 검출
↓
가장 적합한 desk bbox 선택
↓
SAM2에 desk bbox 입력
↓
desk_mask 생성
↓
후처리 및 보정
```

### 4.1 desk_mask 후처리

SAM2가 생성한 책상 mask는 객체가 올라간 부분이 제외되거나, 경계가 끊기는 경우가 있음
따라서 다음 후처리를 적용한다.

```text
1. keep_largest_component
   - 여러 mask 조각 중 가장 큰 연결 영역만 책상으로 사용

2. fill_desk_surface_mask
   - 책상 상판 외곽선을 기준으로 내부를 채움
   - SAM2가 객체가 올라간 부분을 책상에서 제외하는 문제를 완화

3. bbox clip
   - convex hull 보정 후 책상 밖으로 과하게 확장되는 것을 방지
   - DINO가 검출한 desk bbox 내부로 mask 제한

4. conditional bottom clip
   - DINO bbox가 책상 전면부나 아래 구조까지 포함하는 경우를 보정
   - mask가 bbox 하단까지 과하게 확장되었을 때만 하단부 제한
```

계산식으로 표현하면 다음과 같음

```text
desk_mask_raw = SAM2(DINO_desk_bbox)

desk_mask_largest = largest_component(desk_mask_raw)

desk_mask_filled = fill_contour_or_convex_hull(desk_mask_largest)

desk_mask_clipped = desk_mask_filled ∩ desk_bbox_mask

desk_mask_final = bottom_clip_if_overextended(desk_mask_clipped)
```

최종 결과는 다음 파일로 저장

```text
desk_detection_result.png
desk_mask.png
desk_mask_overlay.png
```

---

## 5. occupied_mask 생성

`occupied_mask`는 책상 위에 이미 존재하는 객체들이 차지하는 영역을 의미함

예시 객체:

```text
monitor
keyboard
mouse
mouse pad
cup / mug
book / notebook
lamp
speaker
plant
clock
bottle
cable
```

처리 방식:

```text
top-view image에서 DINO로 객체 bbox 검출
↓
각 bbox를 SAM2에 입력하여 객체별 mask 생성
↓
객체별 mask 후처리
↓
모든 객체 mask를 병합하여 occupied_mask 생성
```

개념적으로는 다음과 같다.

```text
occupied_mask = mask(object_1) ∪ mask(object_2) ∪ ... ∪ mask(object_n)
```

단, 가용공간 분석에서는 책상 상판 밖의 객체가 영향을 주면 안 되므로, 최종적으로는 desk_mask와의 관계를 고려해야 함

```text
occupied_on_desk = occupied_mask ∩ desk_mask
```

향후 안정화를 위해 객체 instance 단위로 다음 필터링을 적용할 수 있음

```text
overlap_ratio = area(object_mask ∩ desk_mask) / area(object_mask)

if overlap_ratio >= threshold:
    keep object
else:
    remove object as off-desk false positive
```

이 방식은 의자, 바닥, 벽, 책장 등이 mouse pad나 book 등으로 오검출되는 경우를 줄이기 위한 후처리임

---

## 6. 가용공간 분석

가용공간은 책상 상판 중 기존 객체가 차지하지 않는 영역임

기본 계산식:

```text
available_space = desk_mask - occupied_mask
```

mask 연산으로 표현하면 다음과 같다.

```text
available_mask = desk_mask ∩ NOT(occupied_mask)
```

replace 모드에서는 제거 대상 객체가 다시 배치 가능 영역이 될 수 있으므로, remove_mask를 반영한다.

```text
keep_occupied_mask = occupied_mask - remove_mask
available_mask = desk_mask - keep_occupied_mask
```

즉 모드별 차이는 다음과 같다.

```text
own_desk / add mode:
available_space = desk_mask - occupied_mask

replace mode:
available_space = desk_mask - (occupied_mask - remove_mask)

empty_desk mode:
available_space = desk_mask
```

---

## 7. connected component 기반 available region 분리

단순히 available mask만 생성하면 제품 배치에 바로 활용하기 어려움
따라서 available mask를 연결된 영역 단위로 분리하여 각 영역의 위치와 크기를 계산함

처리 방식:

```text
available_space_mask
↓
connected component 분석
↓
작은 영역 제거
↓
각 region별 bbox, 중심 좌표, pixel 면적 계산
↓
실제 책상 크기 기준 cm 단위 변환
```

각 region에서 계산하는 정보:

```text
region_id
bbox_xyxy
center_xy
area_px
width_px
height_px
area_cm2
width_cm
height_cm
```

실제 크기 변환은 사용자가 입력한 책상 크기를 기준으로 함

입력값 예:

```text
desk_width_cm = 120
desk_depth_cm = 60
```

픽셀-센티미터 변환 개념:

```text
px_per_cm_x = desk_mask_width_px / desk_width_cm
px_per_cm_y = desk_mask_height_px / desk_depth_cm

width_cm = width_px / px_per_cm_x
height_cm = height_px / px_per_cm_y
area_cm2 = area_px / (px_per_cm_x * px_per_cm_y)
```

이 결과는 제품 추천 파트에서 상품의 실제 크기와 비교하기 위해 사용

---

## 8. LaMa 기반 객체 제거

LaMa는 기존 객체를 제거하고 배경을 복원하기 위한 inpainting 모델로 사용

사용 모드:

```text
own_desk / add:
기존 객체를 유지하므로 기본적으로 LaMa 실행하지 않음

replace:
선택한 객체만 remove_mask로 제거 후 새 제품 배치 가능

empty_desk:
전체 occupied_mask를 제거하여 빈 책상처럼 복원
```

LaMa 입력:

```text
front-view image
remove_mask
```

LaMa 출력:

```text
front_lama_cleaned.png
cleaned_front.png
cleaned_front_combined.png
```

처리 흐름:

```text
remove 대상 객체 선택
↓
front-view 기준 remove_mask 생성
↓
LaMa inpainting 실행
↓
객체가 제거된 front image 생성
↓
후속 제품 배치/합성 단계에서 사용
```

remove_mask 생성 방식:

```text
replace mode:
remove_labels 또는 remove_indices에 해당하는 객체 mask만 제거

empty_desk mode:
전체 occupied_mask 제거

own_desk / add mode:
remove_mask 없음
```

---

## 9. 제품 배치 연동 구조

가용공간 분석 결과는 제품 추천 및 배치 모듈로 전달

전달 가능한 정보:

```text
desk_mask
occupied_mask
available_space_mask
available_regions
desk_size_cm
detected_objects
remove_mask
placement candidates
```

제품 배치에서는 다음 조건을 활용할 수 있음

```text
1. 제품이 available region 내부에 들어가는지
2. 기존 객체와 겹치지 않는지
3. 이미 배치된 다른 제품과 겹치지 않는지
4. 제품의 실제 크기가 available region에 적합한지
5. 카테고리별 선호 위치에 가까운지
6. 테마/스타일/가격 조건에 맞는지
```

현재 제품 배치 결과는 `products_list.json` 형태로 저장되며, 각 제품은 다음 정보를 가짐

```json
{
  "category": "MONITOR",
  "image_id": 196,
  "region": [328, 51, 748, 303],
  "width_px": 420,
  "height_px": 252,
  "placement_source": "available_space_scoring",
  "fallback_reason": null,
  "candidate_count": 49,
  "score": 0.97,
  "available_region_id": 1,
  "anchor_rx": 0.501,
  "anchor_ry": 0.215
}
```

주요 필드 의미:

```text
category:
제품 카테고리

image_id:
제품 이미지 또는 DB item id

region:
배치된 영역 [x1, y1, x2, y2]

width_px / height_px:
배치된 제품의 이미지상 크기

placement_source:
available_space_scoring 또는 fallback

fallback_reason:
정상 배치 실패 시 fallback 원인

candidate_count:
후보 위치 개수

score:
배치 점수

available_region_id:
배치에 사용된 available region id

anchor_rx / anchor_ry:
available region 내부 상대 위치
```

---

## 10. 주요 출력 파일

파이프라인 실행 후 outputs 디렉토리에 다음 결과가 저장됨

```text
front_input_copy.png
top_view_input_copy.png

top_detection_result.png
front_detection_result.png
detection_result.png

raw_occupied_mask.png
processed_occupied_mask.png
sam2_mask_overlay.png

front_raw_occupied_mask.png
front_processed_occupied_mask.png
front_sam2_mask_overlay.png

desk_detection_result.png
desk_mask.png
desk_mask_overlay.png

available_space_mask.png
available_space_overlay.png

top_remove_mask.png
front_remove_mask.png

front_lama_cleaned.png
cleaned_front.png
cleaned_front_combined.png

pipeline_summary.png
space_analysis_summary.json
products_list.json
```

---

## 11. JSON 결과 구조

`space_analysis_summary.json`에는 다음 정보가 저장됨

```text
mode
object_prompt
desk_prompt
desk_mask_mode
thresholds
remove 설정
desk_detections
top_detections
front_detections
space_analysis
outputs
```

이 JSON은 DB/추천/이미지 생성 파트와 연동하기 위한 중간 산출물임

추천 파트에서 주로 사용하는 항목:

```text
available_regions
desk_width_cm
desk_depth_cm
detected_objects
occupied_area
available_area
```

이미지 생성/합성 파트에서 주로 사용하는 항목:

```text
desk_mask
occupied_mask
available_space_mask
remove_mask
placement region
front_lama_cleaned
```

---

## 12. 실행 예시

```bash
python src/pipeline_test.py \
  --front-image inputs/front.jpg \
  --top-view-image inputs/top_view.jpg \
  --desk-width-cm 120 \
  --desk-depth-cm 60 \
  --auto-desk-mask \
  --desk-prompt "tabletop. desk top surface. wooden desk surface. flat desk surface." \
  --prompt "monitor. laptop. keyboard. mouse. mouse pad. cup. mug. book. notebook. notepad. paper. document. diary. planner. lamp. desk lamp. lamp base. plant. plant pot. pot. clock. digital clock. speaker. bottle. tissue. cable. remote control." \
  --mode own_desk \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --sam2-checkpoint weights/sam2.1_hiera_tiny.pt \
  --box-threshold 0.23 \
  --text-threshold 0.18 \
  --desk-box-threshold 0.20 \
  --desk-text-threshold 0.15 \
  --min-region-area-px 1000 \
  --disable-lama \
  --outputs-dir outputs_auto_desk_space
```

---

## 13. 현재 한계 및 개선 방향

현재 파이프라인은 종합 프로젝트 프로토타입으로서 동작하지만, 다음 한계가 존재함

```text
1. 단일 top-view 이미지에서 객체에 가려진 책상 상판을 완전히 복원하기 어려움
2. Grounding DINO는 prompt 기반 모델이므로 객체 누락/중복/오검출 가능
3. top-view와 front-view 간 시점 차이로 동일 객체 검출 결과가 다를 수 있음
4. desk_mask가 책상 전면부나 주변 가구까지 확장될 수 있음
5. LaMa inpainting 결과가 책상 질감과 완전히 자연스럽지 않을 수 있음
6. 제품 배치는 현재 이미지 합성/overlay 수준이며, 실제 생성 모델과의 통합 고도화 필요
```

개선 방향:

```text
1. desk_mask polygon 근사 방식 추가
   - 책상 상판을 사각형/사다리꼴로 근사하여 안정화

2. instance 단위 desk overlap filtering
   - 책상 밖 오검출 객체 제거

3. front-view와 top-view 간 좌표 정합
   - homography 또는 수동 대응점 기반 보정

4. depth/control 기반 이미지 생성 연동
   - 제품 배치 후 자연스러운 생성 결과 도출

5. 제품 DB metadata 고도화
   - 실제 제품 크기, 카테고리, 스타일, 가격 정보 활용

6. 추천 알고리즘 고도화
   - available region 크기
   - 제품 실제 크기
   - 테마 유사도
   - 예산
   - 카테고리별 위치 선호도 반영
```

---


