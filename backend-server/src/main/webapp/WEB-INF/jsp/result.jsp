<!DOCTYPE html>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="jakarta.tags.core" %>
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

        <!-- 추천된 제품 리스트 (완료 시 표시) -->
        <div id="productsBlock" style="display:none;">
            <div class="mb-4 mt-5">
                <h4 class="fw-bold">Products in This Setup</h4>
                <p class="text-muted">생성된 책상에 사용된 제품들 — 클릭하면 구매 페이지로 이동합니다</p>
            </div>
            <div id="productsList" class="row g-4"></div>
        </div>

        <div id="retryBlock" class="text-center mt-5" style="display:none;">
            <a href="${pageContext.request.contextPath}/customize" class="btn btn-primary btn-lg px-5">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-arrow-repeat me-2" viewBox="0 0 16 16">
                    <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41zm-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9z"/>
                    <path fill-rule="evenodd" d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5.002 5.002 0 0 0 8 3zM3.1 9a5.002 5.002 0 0 0 8.757 2.182.5.5 0 1 1 .771.636A6.002 6.002 0 0 1 2.083 9H3.1z"/>
                </svg>
                다른 스타일 만들러 가기
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
                // 빈 책상 모드(add)에서는 제거 단계가 없으므로 "제거 N" 표시 생략
                const isEmptyMode = data.mode === "add";
                progressStatus.textContent = isEmptyMode
                    ? ("상태: " + status + " (배치 " + placed + "/5)")
                    : ("상태: " + status + " (제거 " + removed + " / 배치 " + placed + "/5)");

                if (status === "done") {
                    if (!data.result_image) {
                        showError("status=done인데 result_image가 없습니다.");
                        return;
                    }
                    resultImage.src = "data:image/png;base64," + data.result_image;
                    progressPanel.style.display = "none";
                    resultPanel.style.display = "block";
                    retryBlock.style.display = "block";
                    renderProducts(data.products || []);
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

        function escapeHtml(s) {
            if (s == null) return "";
            return String(s)
                .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }

        function fmtPrice(n) {
            if (n == null) return "";
            return Number(n).toLocaleString("ko-KR") + "원";
        }

        // ai-server 카테고리 → 한글 표시명
        const CAT_KO = {
            MONITOR: "모니터", KEYBOARD: "키보드", MOUSE: "마우스",
            MOUSEPAD: "마우스패드", SPEAKER: "스피커", DESK_LAMP: "데스크 램프",
            DESK_SHELF: "책상 선반", LAPTOP_STAND: "노트북 거치대",
            DECO: "데코", CLOCK: "시계", LIGHTING: "조명",
        };

        function renderProducts(products) {
            const productsBlock = document.getElementById("productsBlock");
            const list = document.getElementById("productsList");
            if (!products || products.length === 0) {
                productsBlock.style.display = "none";
                return;
            }
            list.innerHTML = "";
            for (const p of products) {
                const catLabel = escapeHtml(CAT_KO[p.category] || p.category || "");
                const name     = escapeHtml(p.name || "이름 없음");
                const imgSrc   = p.image_url ? escapeHtml(p.image_url)
                                             : "https://placehold.co/400x400/e2e8f0/64748b?text=No+Image";
                const price    = fmtPrice(p.price);
                const link     = p.product_url ? escapeHtml(p.product_url) : null;

                const card = document.createElement("div");
                card.className = "col-md-6 col-lg-3";
                card.innerHTML =
                    '<div class="card h-100 product-card">' +
                        '<img src="' + imgSrc + '" alt="' + name + '" class="card-img-top" ' +
                             'style="height:180px;object-fit:cover" ' +
                             'onerror="this.src=\'https://placehold.co/400x400/e2e8f0/64748b?text=No+Image\'">' +
                        '<div class="card-body">' +
                            '<span class="badge bg-light text-secondary mb-2">' + catLabel + '</span>' +
                            '<h6 class="card-title">' + name + '</h6>' +
                            (price ? '<p class="card-text text-primary fw-bold mb-0">' + price + '</p>' : '') +
                        '</div>' +
                        '<div class="card-footer bg-transparent border-0 pb-3">' +
                            (link
                                ? '<a href="' + link + '" target="_blank" rel="noopener noreferrer" ' +
                                  'class="btn btn-outline-primary btn-sm w-100">구매 페이지 보기</a>'
                                : '<button class="btn btn-outline-secondary btn-sm w-100" disabled>링크 없음</button>'
                            ) +
                        '</div>' +
                    '</div>';
                list.appendChild(card);
            }
            productsBlock.style.display = "block";
        }

        poll();
    })();
</script>

<jsp:include page="layout/footer.jsp" />
