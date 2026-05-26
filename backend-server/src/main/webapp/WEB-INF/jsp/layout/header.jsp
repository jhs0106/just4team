<!DOCTYPE html>
<html lang="en">
<head>

    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>데스크테리어 - AI Desk Styling</title>
<!-- Google Fonts -->
    <link rel="preconnect"
          href="https://fonts.googleapis.com">

    <link rel="preconnect"
          href="https://fonts.gstatic.com"
          crossorigin>

    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
          rel="stylesheet">

    <!-- Bootstrap Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"
          rel="stylesheet">

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css"
          rel="stylesheet"
          integrity="sha384-sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB"
          crossorigin="anonymous">

    <!-- Custom CSS -->
    <link href="${pageContext.request.contextPath}/css/custom.css"
          rel="stylesheet">

</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-light bg-white border-bottom shadow-sm fixed-top">
        <div class="container">
            <a class="navbar-brand fw-bold text-primary" href="/">데스크테리어 시뮬레이션</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a href="/#about" class="nav-link active">About</a></li>
                    <li class="nav-item"><a href="/#services" class="nav-link">Services</a></li>
                    <li class="nav-item"><a href="/#portfolio" class="nav-link">reviews</a></li>
                    <li class="nav-item"><a href="/#contact" class="nav-link">contact</a></li>
                    <li class="nav-item"><a href="/home" class="nav-link">TRY</a></li>
                    </li>
                </ul>
                <%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
                <% request.setCharacterEncoding("UTF-8"); %>
                <c:import url="/Header.jsp" charEncoding="UTF-8"/>
            </div>
        </div>
    </nav>