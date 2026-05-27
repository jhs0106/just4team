package com.smu.just4team.backendserver.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class ResultController {

    @GetMapping("/result")
    public String showResultPage() {
        return "result"; // Esto busca /WEB-INF/jsp/result.jsp
    }
}