<jsp:include page="WEB-INF/jsp/layout/header.jsp" />
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>


<main>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="text-center mb-5">
                    <h2 class="fw-bold">Create Your Perfect Desk</h2>
                    <p class="text-muted">Follow the steps below to transform your workspace</p>
                </div>

                <form id="customizeForm" action="/api/customize" method="POST" enctype="multipart/form-data">
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

                                <!-- Preview -->
                                <img id="preview1"
                                     src="https://placehold.co/400x250/e2e8f0/64748b?text=Your+Image"
                                     class="img-fluid rounded-3 mb-3"
                                     style="max-height: 250px;">

                                <!-- Input oculto -->
                                <input type="file"
                                       id="file1"
                                       accept="image/*"
                                       capture="environment"
                                       style="display:none"
                                       onchange="document.getElementById('preview1').src = window.URL.createObjectURL(this.files[0])">

                                <button type="button"
                                        class="btn btn-primary"
                                        onclick="document.getElementById('file1').click()">
                                    Upload Front Image
                                </button>

                            </div>

                            <!-- Imagen 2 -->
                            <div class="border rounded-4 p-4 text-center">

                                <!-- Imagen de ejemplo -->
                                <img src="img/2imagen.jpeg"
                                     class="img-fluid rounded-3 mb-2"
                                     style="max-height: 200px;">

                                <small class="text-muted d-block mb-3">
                                    이미지에 보이는 것처럼 책상의 위에서 아래까지 사진을 찍어 주세요.
                                </small>

                                <!-- Preview -->
                                <img id="preview2"
                                     src="https://placehold.co/400x250/e2e8f0/64748b?text=Your+Image"
                                     class="img-fluid rounded-3 mb-3"
                                     style="max-height: 250px;">

                                <input type="file"
                                       id="file2"
                                       accept="image/*"
                                       capture="environment"
                                       style="display:none"
                                       onchange="document.getElementById('preview2').src = window.URL.createObjectURL(this.files[0])">

                                <button type="button"
                                        class="btn btn-primary"
                                        onclick="document.getElementById('file2').click()">
                                    Upload Side Image
                                </button>

                            </div>
                        </div>
                    </div>

                    <div class="card mb-4">
                        <div class="card-body p-4">

                            <div class="d-flex align-items-center mb-4">
                                <div class="step-icon me-3" style="width: 40px; height: 40px; font-size: 1rem;">2</div>
                                <h5 class="mb-0">Choose Your Style</h5>
                            </div>

                            <!-- Dropdown -->
                            <div class="dropdown w-100">

                                <button class="btn btn-outline-primary dropdown-toggle w-100 p-3"
                                        type="button"
                                        data-bs-toggle="dropdown"
                                        aria-expanded="false"
                                        id="styleButton">
                                    Select Style
                                </button>

                                <ul class="dropdown-menu w-100 shadow">

                                    <li>
                                        <a class="dropdown-item d-flex align-items-center gap-2"
                                           href="javascript:void(0)"
                                           onclick="selectStyle('gamer', 'Gamer')">
                                            🎮 Gamer
                                        </a>
                                    </li>

                                    <li>
                                        <a class="dropdown-item d-flex align-items-center gap-2"
                                           href="javascript:void(0)"
                                           onclick="selectStyle('office', 'Office')">
                                            💼 Office
                                        </a>
                                    </li>

                                    <li>
                                        <a class="dropdown-item d-flex align-items-center gap-2"
                                           href="javascript:void(0)"
                                           onclick="selectStyle('minimalist', 'Minimalist')">
                                            ✨ Minimalist
                                        </a>
                                    </li>

                                </ul>
                            </div>

                            <!-- Hidden input para enviar el valor -->
                            <input type="hidden" name="style" id="styleInput" required>

                        </div>
                    </div>
                    <script>
                    function selectStyle(value, label) {
                        // actualizar botón visible
                        document.getElementById('styleButton').innerText = label;

                        // guardar valor real para el form
                        document.getElementById('styleInput').value = value;
                    }
                    </script>



                    <div class="card mb-4">
                        <div class="card-body p-4">
                            <div class="d-flex align-items-center mb-3">
                                <div class="step-icon me-3" style="width: 40px; height: 40px; font-size: 1rem;">3</div>
                                <h5 class="mb-0">Add Details (Optional)</h5>
                            </div>
                            <textarea class="form-control" id="prompt" name="prompt" rows="3"
                                      placeholder="Tell us more about your preferences... e.g., I prefer wooden desk, need more storage, like green plants..."></textarea>
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

document.getElementById('dropZone').addEventListener('dragover', function(e) {
    e.preventDefault();
    this.classList.add('border-primary', 'bg-primary', 'bg-opacity-10');
});

document.getElementById('dropZone').addEventListener('dragleave', function(e) {
    e.preventDefault();
    this.classList.remove('border-primary', 'bg-primary', 'bg-opacity-10');
});

document.getElementById('dropZone').addEventListener('drop', function(e) {
    e.preventDefault();
    this.classList.remove('border-primary', 'bg-primary', 'bg-opacity-10');
    const fileInput = document.getElementById('deskImage');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        previewFile(fileInput);
    }
});

document.getElementById('customizeForm').addEventListener('submit', function() {
    document.getElementById('submitBtn').disabled = true;
    document.getElementById('btnText').textContent = 'Generating...';
    document.getElementById('loadingSpinner').classList.remove('d-none');
});
</script>

<jsp:include page="WEB-INF/jsp/layout/footer.jsp" />
