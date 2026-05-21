<!DOCTYPE html>
<jsp:include page="layout/header.jsp" />
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>


<main>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="text-center mb-5">
                    <h2 class="fw-bold">Create Your Perfect Desk</h2>
                    <p class="text-muted">Follow the steps below to transform your workspace</p>
                </div>

                <form id="customizeForm" action="api/customize" method="POST" enctype="multipart/form-data">
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

                                <!-- PREVIEW -->
                                <img id="previewFront"
                                     class="preview-image mb-3"
                                     src="https://placehold.co/400x250/e2e8f0/64748b?text=Front+Photo">

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

                                                    <div class="col-md-4">
                                                        <input type="number" class="form-control" name="width" placeholder="Width (cm)">
                                                    </div>

                                                    <div class="col-md-4">
                                                        <input type="number" class="form-control" name="depth" placeholder="Depth (cm)">
                                                    </div>

                                                    <div class="col-md-4">
                                                        <input type="number" class="form-control" name="height" placeholder="Height (cm)">
                                                    </div>

                                                    <div class="col-12">
                                                        <button type="button" class="btn btn-primary px-4">
                                                            Apply Dimensions
                                                        </button>
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
                                    <!-- Placeholder option that will be shown initially -->
                                    <option value="" selected disabled>Please choose a style for your desk</option>
                                    <option value="gamer">🎮 Gamer</option>
                                    <option value="office">💼 Office</option>
                                    <option value="minimalist">✨ Minimalist</option>
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
function previewFile(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('previewImage').src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
    }
}

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

    const video = document.getElementById("cameraFront");

    const canvas = document.getElementById("canvasFront");

    const preview = document.getElementById("previewFront");

    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL("image/png");

    document.getElementById("frontImageData").value = imageData;
    console.log(document.getElementById("frontImageData").value);
    preview.src = imageData;

    closeFrontCamera();

    alert("Front photo captured!");
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

function previewFrontFile(input){

    if(input.files && input.files[0]){

        const reader = new FileReader();

        reader.onload = function(e){

            document.getElementById("previewFront").src = e.target.result;
        };

        reader.readAsDataURL(input.files[0]);
    }
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

    const video = document.getElementById("cameraTop");

    const canvas = document.getElementById("canvasTop");

    const preview = document.getElementById("previewTop");

    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL("image/png");

    document.getElementById("topImageData").value = imageData;

    preview.src = imageData;

    closeTopCamera();

    alert("Top photo captured!");
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

function previewTopFile(input){

    if(input.files && input.files[0]){

        const reader = new FileReader();

        reader.onload = function(e){

            document.getElementById("previewTop").src = e.target.result;
        };

        reader.readAsDataURL(input.files[0]);
    }
}
</script>

<jsp:include page="layout/footer.jsp" />
