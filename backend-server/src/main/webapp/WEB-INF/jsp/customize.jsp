<!DOCTYPE html>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="jakarta.tags.core" %>
<jsp:include page="layout/header.jsp" />


<main>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="text-center mb-5">
                    <h2 class="fw-bold">Create Your Perfect Desk</h2>
                    <p class="text-muted">Follow the steps below to transform your workspace</p>
                </div>

                <c:if test="${not empty error}">
                    <div class="alert alert-danger alert-dismissible fade show" role="alert">
                        <strong>오류:</strong> <c:out value="${error}" />
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                </c:if>

                <style>
                    /* 미리보기 이미지 화면 폭/높이 cap — PC 큰 모니터에서 사진이 폼을 뚫고 나가는 거 방지 */
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
                    <div class="card mb-4">
                        <div class="card-body p-4">
                            <div class="d-flex align-items-center mb-3">
                                <div class="step-icon me-3" style="width: 40px; height: 40px; font-size: 1rem;">0</div>
                                <h5 class="mb-0">책상 상태를 선택하세요</h5>
                            </div>
                            <p class="text-muted small mb-3">현재 책상 위에 물건이 있는지에 따라 처리 방식이 달라집니다.</p>
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <input type="radio" class="btn-check" name="deskMode" id="deskModeEmpty"
                                           value="add" autocomplete="off">
                                    <label class="btn btn-outline-primary w-100 py-3" for="deskModeEmpty">
                                        <div class="fw-bold mb-1">🪑 비어있는 책상</div>
                                        <small class="text-muted">기존 물체 없이 바로 제품 배치</small>
                                    </label>
                                </div>
                                <div class="col-md-6">
                                    <input type="radio" class="btn-check" name="deskMode" id="deskModeOccupied"
                                           value="own_desk" autocomplete="off" checked>
                                    <label class="btn btn-outline-primary w-100 py-3" for="deskModeOccupied">
                                        <div class="fw-bold mb-1">🧹 제품들이 놓여있는 책상</div>
                                        <small class="text-muted">기존 물체 자동 제거 후 새로 배치</small>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="card mb-4">
                        <div class="card-body p-4">
                            <div class="d-flex align-items-center mb-3">
                                <div class="step-icon me-3" style="width: 40px; height: 40px; font-size: 1rem;">1</div>
                                <h5 class="mb-0">Upload Your Desk Photo</h5>
                            </div>
                            <!-- Imagen 1 -->
                            <div class="border rounded-4 p-4 text-center mb-4">

                                <!-- Imagen de ejemplo -->
                                <img src="img/1image.webp"
                                     class="img-fluid rounded-3 mb-2"
                                     style="max-height: 200px;">

                                <small class="text-muted d-block mb-3">
                                    이미지에 보이는 것처럼 책상 앞면을 사진으로 찍어 주세요.
                                </small>

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

                                        Upload Image

                                    </button>

                                <!-- CAMERA AREA -->
                                <div class="camera-container mb-3">

                                    <video id="cameraFront"
                                           autoplay
                                           playsinline
                                           class="camera-video"
                                           style="display:none;">
                                    </video>

                                    <img src="img/front-guide.png"
                                         id="frontGuide"
                                         class="guide-overlay"
                                         style="display:none;">

                                </div>

                                <!-- CANVAS -->
                                <canvas id="canvasFront" style="display:none;"></canvas>

                                <!-- HIDDEN INPUT -->
                                <input type="hidden"
                                       name="frontImageData"
                                       id="frontImageData">

                                <!-- BUTTONS -->
                                <div class="d-flex justify-content-center gap-2 flex-wrap">

                                    <!-- 1 -->
                                    <button type="button"
                                            class="btn btn-primary"
                                            onclick="startFrontCamera()">

                                        Open Camera

                                    </button>

                                    <!-- 2 -->
                                    <button type="button"
                                            class="btn btn-success"
                                            onclick="takeFrontPhoto()">

                                        Take Photo

                                    </button>

                                    <!-- 3 -->
                                    <button type="button"
                                            class="btn btn-danger"
                                            onclick="closeFrontCamera()">

                                        Close Camera

                                    </button>
                                </div>

                                <section class="py-4 bg-light">
                                    <div class="container">

                                        <div class="row justify-content-center">

                                            <div class="col-md-8 text-center">

                                                <h4 class="mb-3">Enter your desk dimensions</h4>

                                                <div class="row g-3">

                                                    <div class="col-md-6">
                                                        <input type="number" class="form-control" name="width"
                                                               placeholder="Width (cm) *" required min="1">
                                                    </div>

                                                    <div class="col-md-6">
                                                        <input type="number" class="form-control" name="depth"
                                                               placeholder="Depth (cm) *" required min="1">
                                                    </div>

                                                </div>

                                            </div>

                                        </div>

                                    </div>
                                </section>
                            </div>

                            <!-- Imagen 2 -->
                            <div class="border rounded-4 p-4 text-center">

                                <!-- Imagen de ejemplo -->
                                <img src="img/2imagen.jpeg"
                                     class="img-fluid rounded-3 mb-3"
                                     style="max-height: 200px;">

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

                                        Upload Image

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
                                    <img src="img/top-guide.png"
                                         id="topGuide"
                                         class="guide-overlay"
                                         style="display:none;">

                                </div>

                                <!-- CANVAS -->
                                <canvas id="canvasTop" style="display:none;"></canvas>

                                <!-- HIDDEN INPUT -->
                                <input type="hidden"
                                       name="topImageData"
                                       id="topImageData">

                                <!-- BUTTONS -->
                                <div class="d-flex justify-content-center gap-2 flex-wrap">

                                    <!-- 1 -->
                                    <button type="button"
                                            class="btn btn-primary"
                                            onclick="startTopCamera()">

                                        Open Camera

                                    </button>

                                    <!-- 2 -->
                                    <button type="button"
                                            class="btn btn-success"
                                            onclick="takeTopPhoto()">

                                        Take Photo

                                    </button>

                                    <!-- 3 -->
                                    <button type="button"
                                            class="btn btn-danger"
                                            onclick="closeTopCamera()">

                                        Close Camera

                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card mb-4">
                        <div class="card-body p-4">
                            <div class="d-flex align-items-center mb-4">
                                <div class="step-icon me-3" style="width: 40px; height: 40px; font-size: 1rem;">2</div>
                                <h5 class="mb-0">Choose Your Style</h5>
                            </div>

                            <!-- Custom ComboBox Dropdown using <select> -->
                            <div class="form-group">
                                <label for="styleSelect" class="form-label">Select Style</label>
                                <select class="form-select" id="styleSelect" name="style" required>
                                    <option value="" selected disabled>Please choose a style for your desk</option>
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
                                <h5 class="mb-0">Enter Your Budget</h5>
                            </div>

                            <div class="mb-3">
                                <label for="budget" class="form-label">Budget (in WON)</label>
                                <input type="number" class="form-control" id="budget" name="budget" placeholder="Enter your budget" required>
                            </div>

                        </div>
                    </div>


                    <div class="text-center">
                        <button type="submit" class="btn btn-primary btn-lg px-5" id="submitBtn">
                            <span id="btnText">Generate My Desk</span>
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
   공통 헬퍼 — 이미지 리사이즈 (최대 변 1280px) + JPEG 변환
   카메라 캡처 또는 파일 업로드 시 ai-server 부담 + 네트워크 페이로드 줄임
========================= */
const MAX_IMAGE_DIM = 1280;
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

function updateClickUI() {
    const preview = document.getElementById("previewFront");
    const hint    = document.getElementById("deskClickHint");
    if (isEmptyDeskMode()) {
        preview.classList.add("click-enabled");
        hint.style.display = "block";
    } else {
        preview.classList.remove("click-enabled");
        hint.style.display = "none";
        clearDeskClick();
    }
}

function clearDeskClick() {
    document.getElementById("deskClickMarker").style.display = "none";
    document.getElementById("deskClickX").value = "";
    document.getElementById("deskClickY").value = "";
}

function handleDeskClick(ev) {
    if (!isEmptyDeskMode()) return;
    const img = ev.currentTarget;
    const rect = img.getBoundingClientRect();
    const x_norm = (ev.clientX - rect.left) / rect.width;
    const y_norm = (ev.clientY - rect.top)  / rect.height;
    if (x_norm < 0 || x_norm > 1 || y_norm < 0 || y_norm > 1) return;

    document.getElementById("deskClickX").value = x_norm.toFixed(4);
    document.getElementById("deskClickY").value = y_norm.toFixed(4);

    const marker = document.getElementById("deskClickMarker");
    marker.style.left = (x_norm * 100) + "%";
    marker.style.top  = (y_norm * 100) + "%";
    marker.style.display = "block";
}

// 모드 라디오 변경 시 클릭 UI on/off
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
        if (isEmptyDeskMode()) {
            const cx = document.getElementById("deskClickX").value;
            if (!cx) {
                ev.preventDefault();
                alert("빈 책상 모드에서는 책상 윗면을 한 번 클릭해주세요.");
                document.getElementById("previewFront").scrollIntoView({behavior: "smooth"});
            }
        }
    });
});

let frontStream = null;

/* =========================
   OPEN CAMERA
========================= */

async function startFrontCamera(){

    try{

        frontStream = await navigator.mediaDevices.getUserMedia({
            video:{
                facingMode:"environment"
            }
        });

        const video = document.getElementById("cameraFront");

        video.srcObject = frontStream;

        video.style.display = "block";

        document.getElementById("frontGuide").style.display = "block";

        await video.play();

    }catch(error){

        alert("Camera Error: " + error);

        console.log(error);
    }
}

/* =========================
   TAKE PHOTO
========================= */

function takeFrontPhoto(){
    capturePhoto("cameraFront", "canvasFront", "previewFront", "frontImageData", closeFrontCamera);
    clearDeskClick();  // 새 사진 → 클릭 좌표 초기화
}

/* =========================
   CLOSE CAMERA
========================= */

function closeFrontCamera(){

    if(frontStream){

        frontStream.getTracks().forEach(track => track.stop());

        frontStream = null;
    }

    document.getElementById("cameraFront").style.display = "none";

    document.getElementById("frontGuide").style.display = "none";
}

/* =========================
   FILE PREVIEW
========================= */

function previewFrontFile(_input){
    handleFileSelected("fileFront", "previewFront", "frontImageData");
    clearDeskClick();  // 새 사진 → 클릭 좌표 초기화
}

let topStream = null;

/* =========================
   OPEN TOP CAMERA
========================= */

async function startTopCamera(){

    try{

        topStream = await navigator.mediaDevices.getUserMedia({
            video:{
                facingMode:"environment"
            }
        });

        const video = document.getElementById("cameraTop");

        video.srcObject = topStream;

        video.style.display = "block";

        document.getElementById("topGuide").style.display = "block";

        await video.play();

    }catch(error){

        alert("Camera Error: " + error);

        console.log(error);
    }
}

/* =========================
   TAKE TOP PHOTO
========================= */

function takeTopPhoto(){
    capturePhoto("cameraTop", "canvasTop", "previewTop", "topImageData", closeTopCamera);
}

/* =========================
   CLOSE TOP CAMERA
========================= */

function closeTopCamera(){

    if(topStream){

        topStream.getTracks().forEach(track => track.stop());

        topStream = null;
    }

    document.getElementById("cameraTop").style.display = "none";

    document.getElementById("topGuide").style.display = "none";
}

/* =========================
   TOP FILE PREVIEW
========================= */

function previewTopFile(_input){
    handleFileSelected("fileTop", "previewTop", "topImageData");
}
</script>

<jsp:include page="layout/footer.jsp" />
