<!DOCTYPE html>
<jsp:include page="layout/header.jsp" />
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>

<main>
    <section class="hero-section">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-6">
                    <h1 class="display-4 fw-bold mb-4">Design Your <span class="text-primary">데스크테리어</span></h1>
                    <p class="lead text-muted mb-4">
                        Upload a photo of your current desk and let AI transform it into your perfect setup.
                        Choose from Gamer, Office, or Minimalist styles with product recommendations.
                    </p>
                    <a href="customize" class="btn btn-primary btn-lg px-5">
                        Start Creating
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-arrow-right ms-2" viewBox="0 0 16 16">
                            <path fill-rule="evenodd" d="M1 8a.5.5 0 0 1 .5-.5h11.793l-3.147-3.146a.5.5 0 0 1 .708-.708l4 4a.5.5 0 0 1 0 .708l-4 4a.5.5 0 0 1-.708-.708L13.293 8.5H1.5A.5.5 0 0 1 1 8z"/>
                        </svg>
                    </a>
                </div>
                <div class="col-lg-6 text-center mt-5 mt-lg-0">
                    <img src="https://placehold.co/600x450/667eea/ffffff?text=AI+Desk+Styling"
                         alt="Desk preview" class="img-fluid hero-image shadow">
                </div>
            </div>
        </div>
    </section>

    <section class="steps-section bg-white py-5">
        <div class="container">
            <div class="row g-4">
                <div class="col-md-4 text-center">
                    <div class="step-icon">1</div>
                    <h5 class="mt-3">Upload Your Desk</h5>
                    <p class="text-muted">Take a photo of your current desk setup</p>
                </div>
                <div class="col-md-4 text-center">
                    <div class="step-icon">2</div>
                    <h5 class="mt-3">Choose Your Style</h5>
                    <p class="text-muted">Gamer, Office, or Minimalist</p>
                </div>
                <div class="col-md-4 text-center">
                    <div class="step-icon">3</div>
                    <h5 class="mt-3">Get AI Results</h5>
                    <p class="text-muted">See styled desk with product links</p>
                </div>
            </div>
        </div>
    </section>

    <section class="py-5">
        <div class="container">
            <div class="row g-4">
                <div class="col-md-4">
                    <div class="card h-100 text-center">
                        <div class="card-body p-4">
                            <div class="step-icon mb-3" style="background: linear-gradient(135deg, #1a1a2e 0%, #4a4a6a 100%);">G</div>
                            <h5>Gamer</h5>
                            <p class="text-muted mb-0">RGB lighting, gaming gear, and epic setups</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100 text-center">
                        <div class="card-body p-4">
                            <div class="step-icon mb-3" style="background: linear-gradient(135deg, #495057 0%, #868e96 100%);">O</div>
                            <h5>Office</h5>
                            <p class="text-muted mb-0">Professional, ergonomic, productivity-focused</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card h-100 text-center">
                        <div class="card-body p-4">
                            <div class="step-icon mb-3" style="background: linear-gradient(135deg, #f8f9fa 0%, #dee2e6 100%); color: #495057;">M</div>
                            <h5>Minimalist</h5>
                            <p class="text-muted mb-0">Clean, simple, and clutter-free spaces</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
</main>

<jsp:include page="layout/footer.jsp" />
