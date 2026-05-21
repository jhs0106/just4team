package com.smu.just4team.backendserver.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class CustomizeController {

    @GetMapping("/customize")
    public String showForm() {

        return "customize";
    }

    @PostMapping("/api/customize")
    public String handleForm(

            @RequestParam("width") String width,
            @RequestParam("depth") String depth,
            @RequestParam("height") String height,
            @RequestParam("style") String style,
            @RequestParam(value = "prompt", required = false) String prompt,
            @RequestParam("budget") String budget,

            // FILE UPLOADS
            @RequestParam(value = "frontFile", required = false)
            MultipartFile frontFile,

            @RequestParam(value = "topFile", required = false)
            MultipartFile topFile,

            // CAMERA BASE64
            @RequestParam(value = "frontImageData", required = false)
            String frontImageData,

            @RequestParam(value = "topImageData", required = false)
            String topImageData,

            Model model

    ) {

        System.out.println("===== FORM DATA =====");

        System.out.println("Width: " + width);
        System.out.println("Depth: " + depth);
        System.out.println("Height: " + height);

        System.out.println("Style: " + style);

        System.out.println("Prompt: " + prompt);

        System.out.println("Budget: " + budget);

        System.out.println("Front length: " +
                (frontImageData != null ? frontImageData.length() : 0));

        System.out.println("Top length: " +
                (topImageData != null ? topImageData.length() : 0));

        /* =========================
           FILE UPLOADS
        ========================= */

        if(frontFile != null && !frontFile.isEmpty()) {

            System.out.println("Front File Upload: "
                    + frontFile.getOriginalFilename());
        }

        if(topFile != null && !topFile.isEmpty()) {

            System.out.println("Top File Upload: "
                    + topFile.getOriginalFilename());
        }

        /* =========================
           CAMERA IMAGES
        ========================= */

        if(frontImageData != null &&
                !frontImageData.isEmpty()) {

            System.out.println("Front Camera Image Received");
        }

        if(topImageData != null &&
                !topImageData.isEmpty()) {

            System.out.println("Top Camera Image Received");
        }

        /* =========================
           MODEL
        ========================= */

        model.addAttribute("width", width);

        model.addAttribute("depth", depth);

        model.addAttribute("height", height);

        model.addAttribute("style", style);

        model.addAttribute("prompt", prompt);

        model.addAttribute("budget", budget);

        return "customize";
    }
}