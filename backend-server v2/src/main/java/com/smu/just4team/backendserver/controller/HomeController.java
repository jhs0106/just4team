package com.smu.just4team.backendserver.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping({"/", "/index"})
    public String indexPage() {
        return "index"; // Esto buscará /WEB-INF/jsp/index.jsp
    }

    // 단일 랜딩으로 통합 — 기존 /home 링크는 메인('/')으로 리다이렉트
    @GetMapping("/home")
    public String home() {
        return "redirect:/";
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