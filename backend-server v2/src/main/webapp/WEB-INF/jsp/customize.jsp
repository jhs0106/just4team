<!DOCTYPE html>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="jakarta.tags.core" %>
<jsp:include page="layout/header.jsp" />

<main>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="text-center mb-5">
                    <h2 class="fw-bold">나만의 책상 꾸미기</h2>
                    <p class="text-muted">아래 단계를 따라 책상을 꾸며보세요</p>
                </div>

                <c:if test="${not empty error}">
                    <div class="alert alert-danger alert-dismissible fade show" role="alert">
                        <strong>오류:</strong> <c:out value="${error}" />
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                </c:if>
                
                <style>
                    /*미리보기 이미지 화면 폭/높이 cap — PC 큰 모니터에서 사진이 폼을 뚫고 나가는 거 방지*/
                    .preview-image {
                        max-width: 100% !important;
                        max-height: 320px !important;
                        width: auto !important;
                        height: auto !important;
                        object-fit: contain;
                        display: block;
                        margin: 0 auto;
                    }
                    .camera-video {
                        max-width: 100%;
                        max-height: 360px;
                    }
                    /* 클릭 UI — 빈 책상 모드에서 책상 윗면 클릭 */
                    .desk-click-wrapper {
                        position: relative;
                        display: inline-block;
                        max-width: 100%;
                    }
                    .desk-click-marker {
                        position: absolute;
                        width: 24px; height: 24px;
                        border: 3px solid #ff3b3b;
                        border-radius: 50%;
                        background: rgba(255, 59, 59, 0.3);
                        transform: translate(-50%, -50%);
                        pointer-events: none;
                        display: none;
                        box-shadow: 0 0 8px rgba(255, 59, 59, 0.6);
                    }
                    /* 책상 윗면 4점(모서리) 마커 — 번호 표시 */
                    .desk-corner-marker {
                        position: absolute;
                        width: 22px; height: 22px;
                        border: 2px solid #2563eb;
                        border-radius: 50%;
                        background: rgba(37, 99, 235, 0.55);
                        color: #fff; font-size: 12px; font-weight: bold;
                        text-align: center; line-height: 18px;
                        transform: translate(-50%, -50%);
                        pointer-events: none;
                    }
                    /* 남길 제품(keep) 마커 — 초록 */
                    .desk-keep-marker {
                        position: absolute;
                        width: 22px; height: 22px;
                        border: 2px solid #16a34a;
                        border-radius: 50%;
                        background: rgba(22, 163, 74, 0.55);
                        transform: translate(-50%, -50%);
                        pointer-events: none;
                        box-shadow: 0 0 8px rgba(22, 163, 74, 0.6);
                    }
                    /* 검출 객체 박스 — 클릭으로 남길 제품 선택 */
                    .desk-obj-box {
                        position: absolute;
                        border: 2px solid #ff9800;
                        background: rgba(255, 152, 0, 0.12);
                        box-sizing: border-box;
                        border-radius: 4px;
                        cursor: pointer;
                        transition: background .15s ease, border-color .15s ease;
                    }
                    .desk-obj-box:hover { background: rgba(255, 152, 0, 0.28); }
                    .desk-obj-box.kept {
                        border-color: #16a34a;
                        background: rgba(22, 163, 74, 0.32);
                    }
                    .desk-obj-box .obj-tag {
                        position: absolute;
                        top: -9px; left: -2px;
                        font-size: 10px; line-height: 1;
                        padding: 2px 6px; border-radius: 8px;
                        color: #fff; background: #16a34a;
                        white-space: nowrap;
                        display: none;
                    }
                    .desk-obj-box.kept .obj-tag { display: inline-block; }
                    .desk-click-hint {
                        background: #fff3cd;
                        border: 1px solid #ffc107;
                        color: #664d03;
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 0.9rem;
                        margin-top: 8px;
                        display: none;
                    }
                    #previewFront.click-enabled {
                        cursor: crosshair;
                        outline: 2px dashed #ff3b3b;
                        outline-offset: 4px;
                    }
                </style>

                <form id="customizeForm" action="${pageContext.request.contextPath}/api/customize" method="POST" enctype="multipart/form-data">
                    <!-- Step 0: 책상 상태 -->
                    <div class="card mb-3">
                        <div class="card-body p-3">
                            <div class="d-flex align-items-center flex-wrap gap-2">
                                <span class="fw-semibold me-1">책상 상태</span>
                                <input type="radio" class="btn-check" name="deskMode" id="deskModeEmpty"
                                       value="add" autocomplete="off">
                                <label class="btn btn-sm btn-outline-primary" for="deskModeEmpty">비어있는 책상</label>
                                <input type="radio" class="btn-check" name="deskMode" id="deskModeOccupied"
                                       value="own_desk" autocomplete="off" checked>
                                <label class="btn btn-sm btn-outline-primary" for="deskModeOccupied">제품이 놓여있는 책상</label>
                                <small class="text-muted ms-auto d-none d-md-block">선택에 따라 기존 물체 제거 여부가 달라집니다</small>
                            </div>
                        </div>
                    </div>

                    <div class="card mb-4">
                        <div class="card-body p-4">
                            <div class="d-flex align-items-center mb-3">
                                <div class="step-icon me-3" style="width: 40px; height: 40px; font-size: 1rem;">1</div>
                                <h5 class="mb-0">책상 사진 업로드</h5>
                            </div>
                            <div class="row g-3">
                                <!-- 정면 -->
                                <div class="col-md-6">
                                    <div class="border rounded-4 p-3 text-center h-100">
                                        <div class="fw-semibold mb-1">정면 (앞에서)</div>

                                <!-- Imagen de ejemplo -->
                                <img src="img/1image.webp"
                                     class="img-fluid rounded-3 mb-2"
                                     style="max-height: 110px;">

                                <small class="text-muted d-block mb-3">
                                    이미지에 보이는 것처럼 책상 앞면을 사진으로 찍어 주세요.
                                </small>

                                <!-- 마킹 단계: ① 책상 모서리(homography) / ② 남길 제품(own_desk만) -->
                                <div class="btn-group btn-group-sm mb-2 w-100" role="group">
                                    <button type="button" class="btn btn-primary" id="stepCornerBtn"
                                            onclick="setMarkMode('corners')">① 책상 모서리</button>
                                    <button type="button" class="btn btn-outline-success" id="stepKeepBtn"
                                            onclick="setMarkMode('keep')">② 남길 제품 (선택)</button>
                                </div>

                                <!-- PREVIEW (클릭 가능 영역) -->
                                <div class="desk-click-wrapper mb-2">
                                    <img id="previewFront"
                                         class="preview-image"
                                         src="https://placehold.co/400x250/e2e8f0/64748b?text=Front+Photo"
                                         onclick="handleDeskClick(event)">
                                    <div id="deskClickMarker" class="desk-click-marker"></div>
                                </div>
                                <div id="deskClickHint" class="desk-click-hint">
                                    👆 책상 윗면의 가운데를 한 번 클릭해주세요. AI가 책상 영역을 정확히 인식하도록 도와줍니다.
                                </div>

                                <!-- 빈 책상 모드용 클릭 좌표 (0~1 정규화) -->
                                <input type="hidden" id="deskClickX" name="deskClickX" value="">
                                <input type="hidden" id="deskClickY" name="deskClickY" value="">

                                <!-- 책상 윗면 4모서리 (TL,TR,BR,BL 0~1 정규화 JSON) — corner homography 배치용 -->
                                <input type="hidden" id="deskCorners" name="deskCorners" value="">
                                <!-- 남길 기존 제품 탭 좌표 (0~1 정규화 JSON, own_desk 모드) -->
                                <input type="hidden" id="keepPoints" name="keepPoints" value="">
                                <button type="button" id="deskCornersReset"
                                        class="btn btn-sm btn-outline-secondary mt-1"
                                        style="display:none" onclick="resetCurrentMode()">
                                    책상 모서리 다시 찍기
                                </button>

                                <!-- FILE INPUT -->
                                <input type="file"
                                       id="fileFront"
                                       name="frontFile"
                                       accept="image/*"
                                       style="display:none"
                                       onchange="previewFrontFile(this)">

                                <!-- Upload -->
                                    <button type="button"
                                            class="btn btn-secondary"
                                            onclick="document.getElementById('fileFront').click()">
                                        사진 업로드
                                    </button>

                                <!-- CAMERA AREA -->
                                <div class="camera-container mb-3">
                                    <video id="cameraFront"
                                           autoplay
                                           playsinline
                                           class="camera-video"
                                           style="display:none;">
                                    </video>
                                    <div id="frontGuide" class="guide-overlay" style="display:none;">
                                        <svg class="desk-guide-svg" viewBox="0 0 240 180" preserveAspectRatio="xMidYMid meet">
                                            <!-- 상판 윗면(원근) -->
                                            <polygon points="40,74 200,74 218,98 22,98"></polygon>
                                            <!-- 상판 앞면 -->
                                            <rect x="22" y="98" width="196" height="15" rx="2"></rect>
                                        </svg>
                                        <div class="guide-text">책상 전체가 윤곽선 안에 들어오도록 촬영하세요</div>
                                    </div>
                                </div>

                                <!-- CANVAS -->
                                <canvas id="canvasFront" style="display:none;"></canvas>

                                <!-- HIDDEN INPUT -->
                                <input type="hidden"
                                       name="frontImageData"
                                       id="frontImageData">

                                <!-- BUTTONS -->
                                <div class="d-flex justify-content-center gap-2 flex-wrap camera-controls">

                                    <!-- 1 -->
                                    <button type="button"
                                            class="btn btn-primary"
                                            onclick="startFrontCamera()">
                                        카메라 열기
                                    </button>

                                    <!-- 2 -->
                                    <button type="button"
                                            class="btn btn-success"
                                            onclick="takeFrontPhoto()">
                                        촬영
                                    </button>

                                    <!-- 3 -->
                                    <button type="button"
                                            class="btn btn-danger"
                                            onclick="closeFrontCamera()">
                                        카메라 닫기
                                    </button>
                                </div>
                                    </div>
                                </div>

                                <!-- 위에서 -->
                                <div class="col-md-6">
                                    <div class="border rounded-4 p-3 text-center h-100">
                                        <div class="fw-semibold mb-1">위에서 (탑뷰)</div>

                                <!-- Imagen de ejemplo -->
                                <img src="img/2imagen.jpeg"
                                     class="img-fluid rounded-3 mb-2"
                                     style="max-height: 110px;">
                                <small class="text-muted d-block mb-3">
                                    이미지에 보이는 것처럼 책상의 위에서 아래까지 사진을 찍어 주세요.
                                </small>
                                <!-- PREVIEW -->
                                <img id="previewTop"
                                     class="preview-image mb-3"
                                     src="https://placehold.co/400x250/e2e8f0/64748b?text=Top+Photo">
                                <!-- FILE INPUT -->
                                <input type="file"
                                       id="fileTop"
                                       name="topFile"
                                       accept="image/*"
                                       style="display:none"
                                       onchange="previewTopFile(this)">
                                <!-- Upload -->
                                    <button type="button"
                                            class="btn btn-secondary"
                                            onclick="document.getElementById('fileTop').click()">
                                        사진 업로드
                                    </button>

                                <!-- CAMERA AREA -->
                                <div class="camera-container mb-3">

                                    <!-- CAMERA -->
                                    <video id="cameraTop"
                                           autoplay
                                           playsinline
                                           class="camera-video"
                                           style="display:none;">
                                    </video>

                                    <!-- GUIDE PNG -->
                                    <div id="topGuide" class="guide-overlay" style="display:none;">
                                        <div class="guide-rect"></div>
                                        <div class="guide-text">위에서 책상 윗면을 사각형에 꽉 차게 촬영하세요</div>
                                    </div>
                                </div>

                                <!-- CANVAS -->
                                <canvas id="canvasTop" style="display:none;"></canvas>

                                <!-- HIDDEN INPUT -->
                                <input type="hidden"
                                       name="topImageData"
                                       id="topImageData">

                                <!-- BUTTONS -->
                                <div class="d-flex justify-content-center gap-2 flex-wrap camera-controls">

                                    <!-- 1 -->
                                    <button type="button"
                                            class="btn btn-primary"
                                            onclick="startTopCamera()">
                                        카메라 열기
                                    </button>

                                    <!-- 2 -->
                                    <button type="button"
                                            class="btn btn-success"
                                            onclick="takeTopPhoto()">
                                        촬영
                                    </button>

                                    <!-- 3 -->
                                    <button type="button"
                                            class="btn btn-danger"
                                            onclick="closeTopCamera()">
                                        카메라 닫기
                                    </button>
                                </div>
                                    </div>
                                </div>
                            </div>

                            <!-- 책상 치수 -->
                            <div class="row g-2 mt-3 justify-content-center">
                                <div class="col-12 text-center mb-1"><span class="fw-semibold">책상 치수 (cm)</span></div>
                                <div class="col-6 col-md-4">
                                    <input type="number" class="form-control" name="width"
                                           placeholder="가로 (cm) *" required min="1">
                                </div>
                                <div class="col-6 col-md-4">
                                    <input type="number" class="form-control" name="depth"
                                           placeholder="세로 (cm) *" required min="1">
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card mb-4">
                        <div class="card-body p-4">
                            <div class="d-flex align-items-center mb-4">
                                <div class="step-icon me-3" style="width: 40px; height: 40px; font-size: 1rem;">2</div>
                                <h5 class="mb-0">스타일 선택</h5>
                            </div>

                            <!-- Custom ComboBox Dropdown using <select> -->
                            <div class="form-group">
                                <label for="styleSelect" class="form-label">스타일</label>
                                <select class="form-select" id="styleSelect" name="style" required>
                                    <option value="" selected disabled>책상 스타일을 선택하세요</option>
                                    <option value="white">화이트</option>
                                    <option value="black">블랙</option>
                                    <option value="gaming">게이밍</option>
                                    <option value="wood">우드</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <!-- Optional: JavaScript to sync the selection to the hidden input -->

                    <!-- Budget Input -->
                    <div class="card mb-4">
                        <div class="card-body p-4">
                            <div class="d-flex align-items-center mb-4">
                                <div class="step-icon me-3" style="width: 40px; height: 40px; font-size: 1rem;">3</div>
                                <h5 class="mb-0">예산 입력</h5>
                            </div>
                            <div class="mb-3">
                                <label for="budget" class="form-label">예산 (원)</label>
                                <input type="number" class="form-control" id="budget" name="budget" placeholder="예산을 입력하세요" required min="0">
                            </div>
                        </div>
                    </div>

                    <div class="text-center">
                        <button type="submit" class="btn btn-primary btn-lg px-5" id="submitBtn">
                            <span id="btnText">내 책상 생성하기</span>
                            <span id="loadingSpinner" class="spinner-border spinner-border-sm ms-2 d-none" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </span>
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</main>

<script>
/* =========================
   이미지 리사이즈 (최대 변 1280px) + JPEG 변환
   카메라 캡처 또는 파일 업로드 시 server 부담+네트워크 페이로드 줄임
========================= */
const MAX_IMAGE_DIM = 1600;
const JPEG_QUALITY = 0.85;

function resizeDataUrl(dataUrl, maxDim) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            let w = img.naturalWidth, h = img.naturalHeight;
            if (w > maxDim || h > maxDim) {
                if (w >= h) { h = Math.round(h * maxDim / w); w = maxDim; }
                else        { w = Math.round(w * maxDim / h); h = maxDim; }
            }
            const c = document.createElement("canvas");
            c.width = w; c.height = h;
            c.getContext("2d").drawImage(img, 0, 0, w, h);
            resolve(c.toDataURL("image/jpeg", JPEG_QUALITY));
        };
        img.onerror = reject;
        img.src = dataUrl;
    });
}

function loadFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = e => resolve(e.target.result);
        r.onerror = reject;
        r.readAsDataURL(file);
    });
}

async function handleFileSelected(fileInputId, previewId, hiddenInputId) {
    const input = document.getElementById(fileInputId);
    if (!input.files || !input.files[0]) return;
    try {
        const raw = await loadFileAsDataUrl(input.files[0]);
        const resized = await resizeDataUrl(raw, MAX_IMAGE_DIM);
        document.getElementById(previewId).src = resized;
        document.getElementById(hiddenInputId).value = resized;
        // 원본 file input 비움 — frontImageData/topImageData만 서버로 전송
        input.value = "";
    } catch (e) {
        alert("이미지 처리 실패: " + e);
    }
}

async function capturePhoto(videoId, canvasId, previewId, hiddenInputId, closeFn) {
    const video = document.getElementById(videoId);
    const canvas = document.getElementById(canvasId);
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const raw = canvas.toDataURL("image/png");
    try {
        const resized = await resizeDataUrl(raw, MAX_IMAGE_DIM);
        document.getElementById(hiddenInputId).value = resized;
        document.getElementById(previewId).src = resized;
        closeFn();
    } catch (e) {
        alert("이미지 처리 실패: " + e);
    }
}

/* =========================
   빈 책상 모드 — 책상 윗면 클릭 UI
   사용자가 책상 윗면 한 번 클릭하면 SAM2가 정확한 책상 영역 추출
========================= */
function isEmptyDeskMode() {
    const el = document.getElementById("deskModeEmpty");
    return el && el.checked;
}

// 마킹 모드: "corners"(homography 4점, 항상) | "keep"(남길 제품 탭, own_desk만)
let markMode = "corners";
let deskCornerPts = [];
let deskKeepPts = [];
const CORNER_LABELS = ["좌측 뒤", "우측 뒤", "우측 앞", "좌측 앞"];

// 남길 제품 클릭 선택(검출 박스) 상태
let detectedBoxes = [];        // [{x1,y1,x2,y2,label,score}] 0~1 정규화
let keptBoxIdx = new Set();    // 남길(제거 제외) 박스 인덱스
let detectState = "idle";      // idle | loading | done | none | error

function updateClickUI() {
    document.getElementById("previewFront").classList.add("click-enabled");
    // 칸 크기 고정: 빈 책상 모드에서도 ② 버튼을 숨기지 않고 비활성화만 (레이아웃 흔들림 방지)
    const keepBtn = document.getElementById("stepKeepBtn");
    if (keepBtn) keepBtn.disabled = isEmptyDeskMode();
    // 빈 책상 모드면 keep 무효 → 초기화하고 corners로
    if (isEmptyDeskMode()) {
        clearDeskKeep();
        if (markMode === "keep") markMode = "corners";
    }
    refreshStepButtons();
    applyMarkModeBoxes();
    updateHint();
}

function setMarkMode(mode) {
    if (mode === "keep" && isEmptyDeskMode()) return;  // 빈 책상은 keep 단계 없음
    markMode = mode;
    refreshStepButtons();
    applyMarkModeBoxes();
    updateHint();
}

function refreshStepButtons() {
    const cb = document.getElementById("stepCornerBtn");
    const kb = document.getElementById("stepKeepBtn");
    if (cb) cb.className = "btn " + (markMode === "corners" ? "btn-primary" : "btn-outline-primary");
    if (kb) kb.className = "btn " + (markMode === "keep" ? "btn-success" : "btn-outline-success");
}

// 호환용: 새 사진 선택/촬영 시 모두 초기화
function clearDeskClick() { clearDeskCorners(); clearDeskKeep(); }

function clearDeskCorners() {
    deskCornerPts = [];
    document.getElementById("deskCorners").value = "";
    document.querySelectorAll(".desk-corner-marker").forEach(m => m.remove());
    updateHint();
}

function clearDeskKeep() {
    deskKeepPts = [];
    const kp = document.getElementById("keepPoints");
    if (kp) kp.value = "";
    document.querySelectorAll(".desk-keep-marker").forEach(m => m.remove());
    // 검출 박스/선택 초기화 (새 사진이면 다음 keep 진입 시 재검출)
    detectedBoxes = [];
    keptBoxIdx = new Set();
    detectState = "idle";
    document.querySelectorAll(".desk-obj-box").forEach(b => b.remove());
    updateHint();
}

// 현재 단계의 마킹만 초기화
function resetCurrentMode() {
    if (markMode === "keep") {
        if (detectedBoxes.length > 0) {
            // 검출 박스 모드: 선택만 해제 (검출 결과는 유지)
            keptBoxIdx.clear();
            document.querySelectorAll(".desk-obj-box.kept").forEach(b => b.classList.remove("kept"));
            syncKeepPointsFromBoxes();
            updateHint();
        } else {
            clearDeskKeep();  // 폴백(직접 탭) 좌표 초기화
        }
    } else {
        clearDeskCorners();
    }
}

// ── 남길 제품: 검출 박스 클릭 선택 ──────────────────────
function setBoxesVisible(show) {
    document.querySelectorAll(".desk-obj-box").forEach(b => {
        b.style.display = show ? "block" : "none";
    });
}

function syncKeepPointsFromBoxes() {
    const pts = [];
    keptBoxIdx.forEach(i => {
        const b = detectedBoxes[i];
        if (b) pts.push([
            parseFloat(((b.x1 + b.x2) / 2).toFixed(4)),
            parseFloat(((b.y1 + b.y2) / 2).toFixed(4)),
        ]);
    });
    const kp = document.getElementById("keepPoints");
    if (kp) kp.value = pts.length ? JSON.stringify(pts) : "";
}

function renderObjectBoxes() {
    const wrap = document.querySelector(".desk-click-wrapper");
    document.querySelectorAll(".desk-obj-box").forEach(b => b.remove());
    detectedBoxes.forEach((b, i) => {
        const div = document.createElement("div");
        div.className = "desk-obj-box" + (keptBoxIdx.has(i) ? " kept" : "");
        div.style.left   = (b.x1 * 100) + "%";
        div.style.top    = (b.y1 * 100) + "%";
        div.style.width  = ((b.x2 - b.x1) * 100) + "%";
        div.style.height = ((b.y2 - b.y1) * 100) + "%";
        div.style.display = (markMode === "keep") ? "block" : "none";
        div.innerHTML = '<span class="obj-tag">남김</span>';
        div.addEventListener("click", (ev) => {
            ev.stopPropagation();
            toggleObjBox(i, div);
        });
        wrap.appendChild(div);
    });
}

function toggleObjBox(i, el) {
    if (keptBoxIdx.has(i)) { keptBoxIdx.delete(i); el.classList.remove("kept"); }
    else { keptBoxIdx.add(i); el.classList.add("kept"); }
    syncKeepPointsFromBoxes();
    updateHint();
}

async function ensureDetection() {
    if (isEmptyDeskMode()) return;
    // loading/done/none은 재검출 안 함. error는 keep 재진입 시 재시도.
    if (detectState === "loading" || detectState === "done" || detectState === "none") return;
    const frontData = document.getElementById("frontImageData").value;
    if (!frontData) return;  // 사진 없으면 검출 불가
    detectState = "loading";
    updateHint();
    try {
        const resp = await fetch("${pageContext.request.contextPath}/api/detect-objects", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_base64: frontData }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || ("HTTP " + resp.status));
        detectedBoxes = Array.isArray(data.objects) ? data.objects : [];
        keptBoxIdx = new Set();
        detectState = detectedBoxes.length ? "done" : "none";
        renderObjectBoxes();
        setBoxesVisible(markMode === "keep");
    } catch (e) {
        console.warn("[detect] 실패:", e);
        detectedBoxes = [];
        detectState = "error";
    }
    syncKeepPointsFromBoxes();
    updateHint();
}

// keep 모드 진입 시 검출 실행 + 박스 표시, 그 외엔 박스 숨김
function applyMarkModeBoxes() {
    const inKeep = (markMode === "keep") && !isEmptyDeskMode();
    setBoxesVisible(inKeep);
    if (inKeep) ensureDetection();
}

function updateHint() {
    const hint = document.getElementById("deskClickHint");
    const resetBtn = document.getElementById("deskCornersReset");
    hint.style.display = "block";
    if (markMode === "keep") {
        if (detectState === "loading") {
            hint.innerHTML = "② 책상 위 제품을 검출하는 중입니다… 잠시만 기다려주세요.";
            if (resetBtn) resetBtn.style.display = "none";
        } else if (detectState === "done" && detectedBoxes.length > 0) {
            hint.innerHTML = "② 남길 제품을 <b>클릭</b>하세요. 선택한 제품은 그대로 유지되고 나머지는 교체됩니다. "
                           + "(검출 " + detectedBoxes.length + "개, 선택 " + keptBoxIdx.size + "개)";
            if (resetBtn) {
                resetBtn.textContent = "선택 초기화";
                resetBtn.style.display = keptBoxIdx.size ? "inline-block" : "none";
            }
        } else {
            // 검출 0개/실패 → 미리보기 직접 탭 폴백
            const why = detectState === "error" ? "검출에 실패했습니다. " : "검출된 제품이 없습니다. ";
            hint.innerHTML = "② " + why + "남길 위치를 미리보기에서 직접 탭하세요. 안 찍으면 모두 교체됩니다. "
                           + "(현재 " + deskKeepPts.length + "개)";
            if (resetBtn) {
                resetBtn.textContent = "남길 제품 초기화";
                resetBtn.style.display = deskKeepPts.length ? "inline-block" : "none";
            }
        }
    } else {
        const n = deskCornerPts.length;
        if (n < 4) {
            hint.innerHTML = "① 책상 <b>윗면</b> 네 모서리를 <b>좌측뒤 → 우측뒤 → 우측앞 → 좌측앞</b> 순서로 탭하세요. "
                           + "(" + n + "/4, 다음: " + CORNER_LABELS[n] + ")";
        } else {
            hint.innerHTML = "① 책상 4점 완료. 다시 찍으려면 아래 버튼, 또는 ②로 넘어가세요.";
        }
        if (resetBtn) {
            resetBtn.textContent = "책상 모서리 다시 찍기";
            resetBtn.style.display = n ? "inline-block" : "none";
        }
    }
}

function handleDeskClick(ev) {
    const img = ev.currentTarget;
    const rect = img.getBoundingClientRect();
    const x_norm = (ev.clientX - rect.left) / rect.width;
    const y_norm = (ev.clientY - rect.top)  / rect.height;
    if (x_norm < 0 || x_norm > 1 || y_norm < 0 || y_norm > 1) return;
    const wrap = document.querySelector(".desk-click-wrapper");

    if (markMode === "keep") {
        if (isEmptyDeskMode()) return;
        // 검출 박스가 있으면 박스 클릭으로만 선택 — 빈 영역 탭은 무시 (폴백은 검출 0/실패 시)
        if (detectedBoxes.length > 0) return;
        deskKeepPts.push([parseFloat(x_norm.toFixed(4)), parseFloat(y_norm.toFixed(4))]);
        const m = document.createElement("div");
        m.className = "desk-keep-marker";
        m.style.left = (x_norm * 100) + "%";
        m.style.top  = (y_norm * 100) + "%";
        wrap.appendChild(m);
        document.getElementById("keepPoints").value = JSON.stringify(deskKeepPts);
        updateHint();
        return;
    }

    // corners 모드
    if (deskCornerPts.length >= 4) return;
    deskCornerPts.push([parseFloat(x_norm.toFixed(4)), parseFloat(y_norm.toFixed(4))]);
    const m = document.createElement("div");
    m.className = "desk-corner-marker";
    m.style.left = (x_norm * 100) + "%";
    m.style.top  = (y_norm * 100) + "%";
    m.textContent = deskCornerPts.length;
    wrap.appendChild(m);
    if (deskCornerPts.length === 4) {
        document.getElementById("deskCorners").value = JSON.stringify(deskCornerPts);
        if (!isEmptyDeskMode()) setMarkMode("keep");  // own_desk면 자동 ②로
    }
    updateHint();
}

// 모드 라디오 변경 시 클릭 UI 갱신
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("deskModeEmpty").addEventListener("change", updateClickUI);
    document.getElementById("deskModeOccupied").addEventListener("change", updateClickUI);
    updateClickUI();
});

// 폼 submit 직전: 필수 입력 검증 (top-view, 정면 사진, 빈 책상 모드면 클릭)
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("customizeForm").addEventListener("submit", (ev) => {
        const frontData = document.getElementById("frontImageData").value;
        if (!frontData) {
            ev.preventDefault();
            alert("정면 책상 사진을 업로드해주세요.");
            document.getElementById("previewFront").scrollIntoView({behavior: "smooth"});
            return;
        }
        const topData = document.getElementById("topImageData").value;
        if (!topData) {
            ev.preventDefault();
            alert("위에서 본 책상 사진(top-view)도 필수입니다. 업로드해주세요.");
            document.getElementById("previewTop").scrollIntoView({behavior: "smooth"});
            return;
        }
        if (deskCornerPts.length < 4) {
            ev.preventDefault();
            alert("책상 윗면 네 모서리를 모두 클릭해주세요. (현재 " + deskCornerPts.length + "/4)");
            document.getElementById("previewFront").scrollIntoView({behavior: "smooth"});
        }
    });
});

// PC/모바일 판별 — 데스크톱이면 카메라 버튼 숨기고 업로드 안내 (PC는 책상 촬영에 부적합)
document.addEventListener("DOMContentLoaded", () => {
    const ua = navigator.userAgent;
    const uaMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile/i.test(ua);
    // 터치 전용(hover 불가 + coarse) = 진짜 폰/태블릿. 터치스크린 PC(트랙패드/마우스 hover 가능)는 제외.
    const touchOnly = window.matchMedia && window.matchMedia("(hover: none) and (pointer: coarse)").matches;
    const isMobile = uaMobile || touchOnly;
    if (!isMobile) {
        document.querySelectorAll(".camera-controls").forEach(el => {
            // Bootstrap .d-flex가 display:flex !important라, 인라인도 !important로 덮어써야 숨겨짐
            el.style.setProperty("display", "none", "important");
            const hint = document.createElement("small");
            hint.className = "text-muted d-block mt-2";
            hint.innerHTML = "PC에서는 위 <b>사진 업로드</b> 버튼으로 사진을 올려주세요.";
            el.parentNode.insertBefore(hint, el.nextSibling);
        });
    }
});

// 숫자(정수)만 입력 — budget/width/depth. e/E/+/-/. 등 차단 + 붙여넣기 정리
document.addEventListener("DOMContentLoaded", () => {
    const numInputs = document.querySelectorAll(
        'input[name="width"], input[name="depth"], input[name="budget"]');
    const NAV_KEYS = ["Backspace","Delete","Tab","Escape","Enter",
                      "ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End"];
    numInputs.forEach(el => {
        el.addEventListener("keydown", (ev) => {
            if (ev.ctrlKey || ev.metaKey || NAV_KEYS.includes(ev.key)) return;
            if (!/^[0-9]$/.test(ev.key)) ev.preventDefault();
        });
        el.addEventListener("input", () => {
            const cleaned = el.value.replace(/[^0-9]/g, "");
            if (el.value !== cleaned) el.value = cleaned;
        });
    });
});

let frontStream = null;

//OPEN CAMERA
async function startFrontCamera(){
    try{
        frontStream = await navigator.mediaDevices.getUserMedia({
            video:{
                facingMode:"environment",
                width:  { ideal: 1920 },
                height: { ideal: 1080 }
            }
        });
        const video = document.getElementById("cameraFront");
        video.srcObject = frontStream;
        video.addEventListener("loadedmetadata", () => {
            console.log("[camera] front stream:", video.videoWidth + "×" + video.videoHeight);
        }, { once: true });
        video.style.display = "block";
        document.getElementById("frontGuide").style.display = "block";
        await video.play();
    }catch(error){
        alert("Camera Error: " + error);
        console.log(error);
    }
}

//TAKE PHOTO
function takeFrontPhoto(){
    capturePhoto("cameraFront", "canvasFront", "previewFront", "frontImageData", closeFrontCamera);
    clearDeskClick();  // 새 사진 → 클릭 좌표 초기화
}

//CLOSE CAMERA
function closeFrontCamera(){
    if(frontStream){
        frontStream.getTracks().forEach(track => track.stop());
        frontStream = null;
    }
    document.getElementById("cameraFront").style.display = "none";
    document.getElementById("frontGuide").style.display = "none";
}

//FILE PREVIEW
function previewFrontFile(_input){
    handleFileSelected("fileFront", "previewFront", "frontImageData");
    clearDeskClick();  // 새 사진 → 클릭 좌표 초기화
}

let topStream = null;

//OPEN TOP CAMERA
async function startTopCamera(){
    try{
        topStream = await navigator.mediaDevices.getUserMedia({
            video:{
                facingMode:"environment",
                width:  { ideal: 1920 },
                height: { ideal: 1080 }
            }
        });
        const video = document.getElementById("cameraTop");
        video.srcObject = topStream;
        video.addEventListener("loadedmetadata", () => {
            console.log("[camera] top stream:", video.videoWidth + "×" + video.videoHeight);
        }, { once: true });
        video.style.display = "block";
        // 입력한 책상 가로:세로(width:depth)를 top-view 가이드 사각형 비율에 반영
        const _w = parseFloat(document.querySelector('input[name="width"]')?.value);
        const _d = parseFloat(document.querySelector('input[name="depth"]')?.value);
        const _rect = document.querySelector("#topGuide .guide-rect");
        if (_rect && _w > 0 && _d > 0) {
            _rect.style.setProperty("--desk-ar", (_w / _d).toFixed(3));
        }
        document.getElementById("topGuide").style.display = "block";
        await video.play();
    }catch(error){
        alert("Camera Error: " + error);
        console.log(error);
    }
}

//TAKE TOP PHOTO
function takeTopPhoto(){
    capturePhoto("cameraTop", "canvasTop", "previewTop", "topImageData", closeTopCamera);
}

//CLOSE TOP CAMERA
function closeTopCamera(){
    if(topStream){
        topStream.getTracks().forEach(track => track.stop());
        topStream = null;
    }
    document.getElementById("cameraTop").style.display = "none";
    document.getElementById("topGuide").style.display = "none";
}

//TOP FILE PREVIEW
function previewTopFile(_input){
    handleFileSelected("fileTop", "previewTop", "topImageData");
}
</script>

<jsp:include page="layout/footer.jsp" />
