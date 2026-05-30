<!DOCTYPE html>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<jsp:include page="layout/header.jsp" />

<main>
    <!-- 히어로 -->
    <section class="hero-section">
        <div class="container">
            <div class="row align-items-center g-5">
                <div class="col-lg-6">
                    <h1 class="display-4 fw-bold mb-3">
                        사진 한 장으로<br>
                        <span class="text-primary">나만의 데스크테리어</span>
                    </h1>
                    <p class="lead mb-4">
                        쓰던 책상 사진을 올리면 AI가 스타일과 예산에 맞춰
                        실제 구매 가능한 제품으로 책상을 꾸며 보여드립니다.
                    </p>
                    <div class="d-flex gap-2 flex-wrap">
                        <a href="${pageContext.request.contextPath}/customize"
                           class="btn btn-primary btn-lg px-5">시작하기</a>
                        <a href="#how" class="btn btn-outline-primary btn-lg px-4">이용 방법</a>
                    </div>
                </div>
                <div class="col-lg-6 text-center">
                    <img src="img/background1.webp" alt="데스크테리어 예시"
                         class="img-fluid hero-image shadow">
                </div>
            </div>
        </div>
    </section>

    <!-- 이용 방법 -->
    <section id="how" class="steps-section bg-white">
        <div class="container">
            <h2 class="text-center fw-bold mb-5">이용 방법</h2>
            <div class="row g-4">
                <div class="col-md-4 text-center">
                    <div class="step-icon">1</div>
                    <h5 class="mt-3">책상 사진 업로드</h5>
                    <p class="text-muted">정면·위에서 본 책상 사진을 올리고 크기를 입력하세요.</p>
                </div>
                <div class="col-md-4 text-center">
                    <div class="step-icon">2</div>
                    <h5 class="mt-3">스타일·예산 선택</h5>
                    <p class="text-muted">원하는 스타일과 예산을 정하면 맞춤 제품을 추천합니다.</p>
                </div>
                <div class="col-md-4 text-center">
                    <div class="step-icon">3</div>
                    <h5 class="mt-3">AI 결과 확인</h5>
                    <p class="text-muted">꾸며진 책상 이미지와 사용된 제품·구매 링크를 받아보세요.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- 스타일 -->
    <section id="styles" class="py-5">
        <div class="container">
            <h2 class="text-center fw-bold mb-5">스타일</h2>
            <div class="row g-4">
                <div class="col-6 col-md-3">
                    <div class="card h-100 text-center">
                        <div class="card-body p-4">
                            <div class="step-icon mb-3"
                                 style="background: linear-gradient(135deg,#f8fafc 0%,#e2e8f0 100%); color:#475569;">화</div>
                            <h5 class="mb-1">화이트</h5>
                            <p class="text-muted small mb-0">밝고 깔끔한 미니멀</p>
                        </div>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card h-100 text-center">
                        <div class="card-body p-4">
                            <div class="step-icon mb-3"
                                 style="background: linear-gradient(135deg,#1f2937 0%,#4b5563 100%);">블</div>
                            <h5 class="mb-1">블랙</h5>
                            <p class="text-muted small mb-0">차분하고 모던한 무드</p>
                        </div>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card h-100 text-center">
                        <div class="card-body p-4">
                            <div class="step-icon mb-3"
                                 style="background: linear-gradient(135deg,#7c3aed 0%,#ec4899 100%);">게</div>
                            <h5 class="mb-1">게이밍</h5>
                            <p class="text-muted small mb-0">RGB 감성의 게이밍 셋업</p>
                        </div>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="card h-100 text-center">
                        <div class="card-body p-4">
                            <div class="step-icon mb-3"
                                 style="background: linear-gradient(135deg,#b45309 0%,#92400e 100%);">우</div>
                            <h5 class="mb-1">우드</h5>
                            <p class="text-muted small mb-0">따뜻한 우드 톤의 아늑함</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- 최종 CTA -->
    <section class="py-5">
        <div class="container">
            <div class="card text-center text-white" style="background: var(--bg-gradient);">
                <div class="card-body py-5">
                    <h2 class="fw-bold mb-3 text-white">지금 바로 내 책상을 꾸며보세요</h2>
                    <p class="mb-4">사진만 있으면 1분 만에 시작할 수 있습니다.</p>
                    <a href="${pageContext.request.contextPath}/customize"
                       class="btn btn-light btn-lg px-5">시작하기</a>
                </div>
            </div>
        </div>
    </section>
</main>

<jsp:include page="layout/footer.jsp" />
