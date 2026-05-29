<!DOCTYPE html>
<head>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
    <title>Mi Página Funciona</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"
      rel="stylesheet">

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB"
      crossorigin="anonymous">
<meta charset="UTF-8">
</head>

<body>
<!-- Aquí llamas al header -->
<jsp:include page="layout/header.jsp" />


<header class="vh-100 position-relative d-flex align-items-center text-white"
        style="background: url('img/background1.webp') center/cover no-repeat;">

    <!-- 🔵 Overlay azul tecnológico -->
    <div class="position-absolute top-0 start-0 w-100 h-100"
         style="background: rgba(0, 47, 167, 0.45);">
    </div>

    <!-- Contenido -->
    <div class="container text-center position-relative">
        <h1 class="fw-bold">
            Your Desk, Your Essence: The Space Where Comfort Meets Productivity
        </h1>
        <p class="lead mt-3">
            Don’t just imagine the workspace you’ve always dreamed of. Make it real with a design that fits your rhythm and your life.!
        </p>
        <a href="#about" class="btn btn-light mt-3 px-4">
            Find Out More
        </a>
    </div>
</header>


<section class="py-5 bg-primary text-white" id="about">
    <div class="container text-center">
        <h2 class="mb-4">
            We've got what you need!
        </h2>
        <p class="lead">
            We create the perfect place for you to feel comfortable while you work or play.
            We merge the essence of your personality with the functionality your daily life requires.
            We give you a visual map of your future sanctuary, designed to inspire your productivity.
            Because your desk is not just furniture; it is the space where your dreams come to life.
        </p>
        <a href="#services" class="btn btn-light mt-3 px-4">
            Get Started
        </a>
    </div>
</section>



<section class="py-5" id="services">
    <div class="container">
        <h2 class="text-center mb-5">At Your Service</h2>
        <div class="row g-4">
            <!-- Diamante -->
            <div class="col-md-3 text-center">
                <div class="mb-3">
                    <i class="bi bi-gem fs-1 text-primary"></i>
                </div>
                <h5>Total Realism</h5>
                <p class="text-muted">
                    We use real products so what you see is exactly what you can buy.
                </p>
            </div>

            <!-- Computador -->
            <div class="col-md-3 text-center">
                <div class="mb-3">
                    <i class="bi bi-pc-display fs-1 text-primary"></i>
                </div>
                <h5>Open Design</h5>
                <p class="text-muted">
                    A professional space shouldn't be a luxury, but a standard for everyone.
                </p>
            </div>

            <!-- Mundo / Tierra -->
            <div class="col-md-3 text-center">
                <div class="mb-3">
                    <i class="bi bi-globe2 fs-1 text-primary"></i>
                </div>
                <h5>Pure Comfort</h5>
                <p class="text-muted">
                    We design spaces where comfort flows naturally.
                </p>
            </div>

            <!-- Corazón -->
            <div class="col-md-3 text-center">
                <div class="mb-3">
                    <i class="bi bi-heart-fill fs-1 text-primary"></i>
                </div>
                <h5>Built for You</h5>
                <p class="text-muted">
                    Your well-being is our only goal.
                </p>
            </div>
        </div>
    </div>
</section>



<section id="portfolio" class="py-5">
    <div class="container-fluid px-0">
        <div class="row g-0">
            <!-- Item 1 -->
            <div class="col-lg-4 col-sm-6">
                <div class="position-relative overflow-hidden">
                    <img src="img/fullsize/1.jpg"
                         class="img-fluid w-100"
                         alt="Gamer">
                    <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column justify-content-center align-items-center text-white"
                         style="background: rgba(0, 47, 167, 0.75); opacity: 0; transition: 0.3s;"
                         onmouseover="this.style.opacity='1'"
                         onmouseout="this.style.opacity='0'">

                        <div class="text-white-50">Category</div>
                        <h5>Gamer</h5>
                    </div>
                </div>
            </div>

            <!-- Item 2 -->
            <div class="col-lg-4 col-sm-6">
                <div class="position-relative overflow-hidden">
                    <img src="img/fullsize/22.jpg"
                         class="img-fluid w-100"
                         alt="Gamer">
                    <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column justify-content-center align-items-center text-white"
                         style="background: rgba(0, 47, 167, 0.75); opacity: 0; transition: 0.3s;"
                         onmouseover="this.style.opacity='1'"
                         onmouseout="this.style.opacity='0'">
                        <div class="text-white-50">Category</div>
                        <h5>Gamer</h5>
                    </div>
                </div>
            </div>

            <!-- Item 3 -->
            <div class="col-lg-4 col-sm-6">
                <div class="position-relative overflow-hidden">
                    <img src="img/fullsize/33.jpg"
                         class="img-fluid w-100"
                         alt="Office">
                    <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column justify-content-center align-items-center text-white"
                         style="background: rgba(0, 47, 167, 0.75); opacity: 0; transition: 0.3s;"
                         onmouseover="this.style.opacity='1'"
                         onmouseout="this.style.opacity='0'">
                        <div class="text-white-50">Category</div>
                        <h5>Office</h5>
                    </div>
                </div>
            </div>

            <!-- Item 4 -->
            <div class="col-lg-4 col-sm-6">
                <div class="position-relative overflow-hidden">
                    <img src="img/fullsize/44.jpg"
                         class="img-fluid w-100"
                         alt="Office">
                    <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column justify-content-center align-items-center text-white"
                         style="background: rgba(0, 47, 167, 0.75); opacity: 0; transition: 0.3s;"
                         onmouseover="this.style.opacity='1'"
                         onmouseout="this.style.opacity='0'">
                        <div class="text-white-50">Category</div>
                        <h5>Office</h5>
                    </div>
                </div>
            </div>

            <!-- Item 5 -->
            <div class="col-lg-4 col-sm-6">
                <div class="position-relative overflow-hidden">
                    <img src="img/fullsize/55.jpg"
                         class="img-fluid w-100"
                         alt="Minimalist">
                    <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column justify-content-center align-items-center text-white"
                         style="background: rgba(0, 47, 167, 0.75); opacity: 0; transition: 0.3s;"
                         onmouseover="this.style.opacity='1'"
                         onmouseout="this.style.opacity='0'">
                        <div class="text-white-50">Category</div>
                        <h5>Minimalist</h5>
                    </div>
                </div>
            </div>

            <!-- Item 6 -->
            <div class="col-lg-4 col-sm-6">
                <div class="position-relative overflow-hidden">
                    <img src="img/fullsize/66.jpg"
                         class="img-fluid w-100"
                         alt="Minimalist">
                    <div class="position-absolute top-0 start-0 w-100 h-100 d-flex flex-column justify-content-center align-items-center text-white"
                         style="background: rgba(0, 47, 167, 0.75); opacity: 0; transition: 0.3s;"
                         onmouseover="this.style.opacity='1'"
                         onmouseout="this.style.opacity='0'">
                        <div class="text-white-50">Category</div>
                        <h5>Minimalist</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>


<section class="py-5 bg-dark text-white">
    <div class="container text-center">
        <h2 class="mb-4">
            You want to try your first test!
        </h2>
        <a href="/home" class="btn btn-light px-4 py-2">
            TRY!
        </a>
    </div>
</section>



<section id="contact" class="py-5 bg-light">
    <div class="container">
        <div class="text-center mb-4">
            <h2>Let's Get In Touch</h2>
        </div>
        <div class="row justify-content-center">
            <div class="col-md-6">
                <form method="post" action="ContactServlet">
                    <input class="form-control mb-3" name="name" placeholder="Full name">
                    <input class="form-control mb-3" name="email" placeholder="Email">
                    <input class="form-control mb-3" name="phone" placeholder="Phone">
                    <textarea class="form-control mb-3" name="message" rows="5"
                              placeholder="Message"></textarea>
                    <button class="btn btn-primary w-100">
                        Submit
                    </button>
                </form>
            </div>
        </div>
    </div>
</section>
<jsp:include page="layout/footer.jsp" />
<script src="<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js" integrity="sha384-FKyoEForCGlyvwx9Hj09JcYn3nv7wiPVlz7YYwJrWVcXK/BmnVDxM+D2scQbITxI" crossorigin="anonymous"></script>"
</body>
