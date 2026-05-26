package com.smu.just4team.backendserver.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class ResultController {

    @GetMapping("/result")
    public String showResultPage(
            @RequestParam(value = "jobId", required = false) String jobId,
            Model model
    ) {
        model.addAttribute("jobId", jobId);
        return "result";
    }
}
