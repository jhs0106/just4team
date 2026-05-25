<!DOCTYPE html>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<jsp:include page="layout/header.jsp" />


<main>
    <div class="container py-5">
        <div class="text-center mb-5">
            <h2 class="fw-bold">Your Styled Desk</h2>
            <p class="text-muted">Here's your AI-generated workspace with recommended products</p>
        </div>

        <div class="row justify-content-center mb-5">
            <div class="col-lg-10">
                <div class="card">
                    <!-- 진행 상태 패널 (생성 중) -->
                    <div id="progressPanel" class="card-body text-center py-5">
                        <div class="spinner-border text-primary mb-3" role="status" style="width:3rem;height:3rem;">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <h4 id="progressTitle" class="mb-2">AI가 책상을 디자인하는 중입니다</h4>
                        <p id="progressDetail" class="text-muted mb-1">job_id: <span id="jobIdLabel">-</span></p>
                        <p id="progressStatus" class="text-muted mb-0">상태: 시작 대기 중...</p>
                        <p id="progressElapsed" class="text-muted small mt-2">경과: 0초</p>
                    </div>

                    <!-- 결과 패널 (완료 시 표시) -->
                    <div id="resultPanel" style="display:none;">
                        <img id="resultImage" src="" alt="Generated desk image" class="card-img-top" style="border-radius: 24px 24px 0 0;">
                        <div class="card-body text-center py-4 bg-white">
                            <span class="badge bg-primary px-3 py-2 rounded-pill">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-wand-sparkles-icon lucide-wand-sparkles"><path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72"/><path d="m14 7 3 3"/><path d="M5 6v4"/><path d="M19 14v4"/><path d="M10 2v2"/><path d="M7 8H3"/><path d="M21 16h-4"/><path d="M11 3H9"/></svg>
                                AI Generated
                            </span>
                        </div>
                    </div>

                    <!-- 에러 패널 -->
                    <div id="errorPanel" class="card-body text-center py-5" style="display:none;">
                        <div class="text-danger mb-3" style="font-size:3rem;">⚠</div>
                        <h4 class="mb-2">생성 실패</h4>
                        <p id="errorMessage" class="text-muted"></p>
                        <a href="${pageContext.request.contextPath}/customize" class="btn btn-primary mt-3">다시 시도</a>
                    </div>
                </div>
            </div>
        </div>

        <div id="retryBlock" class="text-center mt-5" style="display:none;">
            <a href="${pageContext.request.contextPath}/customize" class="btn btn-primary btn-lg px-5">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-arrow-repeat me-2" viewBox="0 0 16 16">
                    <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41zm-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9z"/>
                    <path fill-rule="evenodd" d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5.002 5.002 0 0 0 8 3zM3.1 9a5.002 5.002 0 0 0 8.757 2.182.5.5 0 1 1 .771.636A6.002 6.002 0 0 1 2.083 9H3.1z"/>
                </svg>
                Try Another Style
            </a>
        </div>
    </div>
</main>

<script>
    (function () {
        const jobId = "<c:out value='${jobId}' />";
        const jobIdLabel = document.getElementById("jobIdLabel");
        const progressPanel = document.getElementById("progressPanel");
        const resultPanel = document.getElementById("resultPanel");
        const errorPanel = document.getElementById("errorPanel");
        const retryBlock = document.getElementById("retryBlock");
        const progressStatus = document.getElementById("progressStatus");
        const progressElapsed = document.getElementById("progressElapsed");
        const resultImage = document.getElementById("resultImage");
        const errorMessage = document.getElementById("errorMessage");

        if (!jobId) {
            progressPanel.style.display = "none";
            errorPanel.style.display = "block";
            errorMessage.textContent = "jobId가 없습니다. /customize에서 다시 시작해주세요.";
            return;
        }
        jobIdLabel.textContent = jobId;

        const t0 = Date.now();
        const POLL_MS = 5000;        // 5초마다 폴링
        const MAX_ELAPSED = 15 * 60; // 15분 타임아웃

        function fmtElapsed() {
            return Math.floor((Date.now() - t0) / 1000);
        }

        async function poll() {
            const elapsed = fmtElapsed();
            progressElapsed.textContent = "경과: " + elapsed + "초";

            if (elapsed > MAX_ELAPSED) {
                showError("타임아웃 (" + MAX_ELAPSED + "초 초과)");
                return;
            }

            try {
                const url = "${pageContext.request.contextPath}/api/job-status/" + encodeURIComponent(jobId);
                const resp = await fetch(url, { cache: "no-store" });
                const data = await resp.json();

                const status = data.status || "unknown";
                const placed = data.num_placed || 0;
                const removed = data.num_removed || 0;
                progressStatus.textContent =
                    "상태: " + status + " (제거 " + removed + " / 배치 " + placed + "/5)";

                if (status === "done") {
                    if (!data.result_image) {
                        showError("status=done인데 result_image가 없습니다.");
                        return;
                    }
                    resultImage.src = "data:image/png;base64," + data.result_image;
                    progressPanel.style.display = "none";
                    resultPanel.style.display = "block";
                    retryBlock.style.display = "block";
                    return;
                }
                if (status === "failed" || status === "error") {
                    showError(data.error || "생성에 실패했습니다.");
                    return;
                }
            } catch (e) {
                console.warn("폴링 실패:", e);
            }

            setTimeout(poll, POLL_MS);
        }

        function showError(msg) {
            progressPanel.style.display = "none";
            errorPanel.style.display = "block";
            errorMessage.textContent = msg;
        }

        poll();
    })();
</script>

<jsp:include page="layout/footer.jsp" />
