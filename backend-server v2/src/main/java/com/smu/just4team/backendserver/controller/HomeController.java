package com.smu.just4team.backendserver.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping({"/", "/index"})
    public String indexPage() {
        return "index"; // Esto buscará /WEB-INF/jsp/index.jsp
    }

    @GetMapping("/home")
    public String home() {
        return "home"; // Esto buscará /WEB-INF/jsp/home.jsp
    }

//    @GetMapping("/customize")
//    public String customizePage() {
//        return "customize"; // Esto buscará /WEB-INF/jsp/customize.jsp
//    }

//    @GetMapping("/result")
//    public String resultPage() {
//        return "result"; // Esto buscará /WEB-INF/jsp/result.jsp
//    }
}