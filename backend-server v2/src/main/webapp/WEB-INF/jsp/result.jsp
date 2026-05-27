<!DOCTYPE html>
<jsp:include page="layout/header.jsp" />
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>


<main>
    <div class="container py-5">
        <div class="text-center mb-5">
            <h2 class="fw-bold">Your Styled Desk</h2>
            <p class="text-muted">Here's your AI-generated workspace with recommended products</p>
        </div>

        <div class="row justify-content-center mb-5">
            <div class="col-lg-10">
                <div class="card">
                    <img src="${generatedImageUrl}" alt="Generated desk image" class="card-img-top" style="border-radius: 24px 24px 0 0;">
                    <div class="card-body text-center py-4 bg-white">
                        <span class="badge bg-primary px-3 py-2 rounded-pill">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-wand-sparkles-icon lucide-wand-sparkles"><path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72"/><path d="m14 7 3 3"/><path d="M5 6v4"/><path d="M19 14v4"/><path d="M10 2v2"/><path d="M7 8H3"/><path d="M21 16h-4"/><path d="M11 3H9"/></svg>
                            AI Generated
                        </span>
                    </div>
                </div>
            </div>
        </div>

        <div class="mb-4">
            <h4 class="fw-bold">Products in This Setup</h4>
            <p class="text-muted">Shop the items featured in your styled desk</p>
        </div>
        <div class="row g-4">
            <c:forEach var="product" items="${products}">
                <div class="col-md-6 col-lg-3">
                    <div class="card h-100 product-card">
                        <img src="${product.imageUrl}" alt="${product.name}" class="card-img-top product-card-img" style="height: 180px; object-fit: cover;">
                        <div class="card-body">
                            <h6 class="card-title">${product.name}</h6>
                            <p class="card-text text-muted small">${product.description}</p>
                        </div>
                        <div class="card-footer bg-transparent border-0 pb-3">
                            <a href="${product.purchaseLink}" target="_blank" class="btn btn-outline-primary btn-sm w-100">
                                View Product
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-square-arrow-out-up-right-icon lucide-square-arrow-out-up-right"><path d="M21 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h6"/><path d="m21 3-9 9"/><path d="M15 3h6v6"/></svg>
                            </a>
                        </div>
                    </div>
                </div>
            </c:forEach>
        </div>

        <div class="text-center mt-5">
            <a href="/customize" class="btn btn-primary btn-lg px-5">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-arrow-repeat me-2" viewBox="0 0 16 16">
                    <path d="M11.534 7h3.932a.25.25 0 0 1 .192.41l-1.966 2.36a.25.25 0 0 1-.384 0l-1.966-2.36a.25.25 0 0 1 .192-.41zm-11 2h3.932a.25.25 0 0 0 .192-.41L2.692 6.23a.25.25 0 0 0-.384 0L.342 8.59A.25.25 0 0 0 .534 9z"/>
                    <path fill-rule="evenodd" d="M8 3c-1.552 0-2.94.707-3.857 1.818a.5.5 0 1 1-.771-.636A6.002 6.002 0 0 1 13.917 7H12.9A5.002 5.002 0 0 0 8 3zM3.1 9a5.002 5.002 0 0 0 8.757 2.182.5.5 0 1 1 .771.636A6.002 6.002 0 0 1 2.083 9H3.1z"/>
                </svg>
                Try Another Style
            </a>
        </div>
    </div>
</main>

<jsp:include page="layout/footer.jsp" />
