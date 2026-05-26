package com.smu.just4team.backendserver.controller;

import com.smu.just4team.backendserver.service.AiServerClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.Base64;
import java.util.Map;

@Controller
public class CustomizeController {

    @Autowired
    private AiServerClient aiServerClient;

    // 사용자 화면 표시 한글 ↔ ai-server theme enum 매핑.
    // ai-server RecommendAndGenerateRequest.theme: white | black | gaming | wood
    private static final Map<String, String> STYLE_MAP = Map.ofEntries(
            Map.entry("화이트", "white"),
            Map.entry("white",  "white"),
            Map.entry("블랙",   "black"),
            Map.entry("black",  "black"),
            Map.entry("게이밍", "gaming"),
            Map.entry("gaming", "gaming"),
            Map.entry("우드",   "wood"),
            Map.entry("wood",   "wood")
    );

    @GetMapping("/customize")
    public String showForm() {
        return "customize";
    }

    @PostMapping("/api/customize")
    public String handleForm(
            @RequestParam(value = "width",  required = false) String width,
            @RequestParam(value = "depth",  required = false) String depth,
            @RequestParam(value = "height", required = false) String height,
            @RequestParam("style")  String style,
            @RequestParam(value = "prompt", required = false) String prompt,
            @RequestParam("budget") String budget,
            @RequestParam(value = "deskMode", required = false, defaultValue = "own_desk") String deskMode,
            @RequestParam(value = "deskClickX", required = false) String deskClickXStr,
            @RequestParam(value = "deskClickY", required = false) String deskClickYStr,

            @RequestParam(value = "frontFile", required = false) MultipartFile frontFile,
            @RequestParam(value = "topFile",   required = false) MultipartFile topFile,
            @RequestParam(value = "frontImageData", required = false) String frontImageData,
            @RequestParam(value = "topImageData",   required = false) String topImageData,

            RedirectAttributes redirectAttributes
    ) {
        try {
            // 1. front 이미지 base64 확보 (파일 우선, 없으면 카메라 캡처)
            String frontB64 = extractBase64(frontFile, frontImageData);
            if (frontB64 == null) {
                redirectAttributes.addFlashAttribute("error", "정면 책상 사진이 필요합니다.");
                return "redirect:/customize";
            }
            // top-view 사진도 필수 — 사용자 책상의 실제 top-view 없으면 placement 불가
            String topB64 = extractBase64(topFile, topImageData);
            if (topB64 == null) {
                redirectAttributes.addFlashAttribute("error", "위에서 본 책상 사진(top-view)도 필수입니다.");
                return "redirect:/customize";
            }

            // 2. style → ai-server theme enum 변환
            String key = style == null ? "" : style.trim().toLowerCase();
            String theme = STYLE_MAP.get(key);
            if (theme == null) theme = STYLE_MAP.getOrDefault(style, "white");

            // 3. budget 파싱
            int budgetInt;
            try {
                budgetInt = Integer.parseInt(budget.replaceAll("[^0-9]", ""));
            } catch (Exception e) {
                redirectAttributes.addFlashAttribute("error", "예산은 숫자로 입력해주세요.");
                return "redirect:/customize";
            }

            // 4. cm → mm 변환 (JSP 입력은 cm). width/depth 필수 — 사진만으론 실측 mm 불가
            Integer widthMm = parseCmToMm(width);
            Integer depthMm = parseCmToMm(depth);
            if (widthMm == null || depthMm == null) {
                redirectAttributes.addFlashAttribute("error",
                        "책상 가로(width)와 깊이(depth)를 cm 단위로 입력해주세요.");
                return "redirect:/customize";
            }

            // 5. deskMode 정규화 — ai-server RemoveMode enum과 일치하는 값만 허용
            String mode = deskMode == null ? "own_desk" : deskMode.trim();
            if (!mode.equals("add") && !mode.equals("own_desk")
                && !mode.equals("replace") && !mode.equals("empty_desk")) {
                mode = "own_desk";
            }

            // 6. 빈 책상 모드면 클릭 좌표 파싱 (없거나 잘못된 형식이면 null → ai-server가 default DINO 사용)
            Double clickX = null, clickY = null;
            if ("add".equals(mode)) {
                try {
                    if (deskClickXStr != null && !deskClickXStr.isBlank()
                     && deskClickYStr != null && !deskClickYStr.isBlank()) {
                        double cx = Double.parseDouble(deskClickXStr);
                        double cy = Double.parseDouble(deskClickYStr);
                        if (cx >= 0.0 && cx <= 1.0 && cy >= 0.0 && cy <= 1.0) {
                            clickX = cx; clickY = cy;
                        }
                    }
                } catch (NumberFormatException ignore) {}
            }

            // 7. ai-server 호출 → job_id
            String jobId = aiServerClient.submitRecommendAndGenerate(
                    theme, budgetInt, frontB64, topB64, widthMm, depthMm, mode, clickX, clickY
            );

            return "redirect:/result?jobId=" + jobId;

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error",
                    "AI 서버 호출 실패: " + e.getMessage());
            return "redirect:/customize";
        }
    }

    private String extractBase64(MultipartFile file, String dataUrl) throws Exception {
        if (file != null && !file.isEmpty()) {
            return Base64.getEncoder().encodeToString(file.getBytes());
        }
        if (dataUrl != null && !dataUrl.isEmpty()) {
            // "data:image/png;base64,XXX" prefix 제거
            int comma = dataUrl.indexOf(',');
            return comma >= 0 ? dataUrl.substring(comma + 1) : dataUrl;
        }
        return null;
    }

    private Integer parseCmToMm(String cm) {
        if (cm == null || cm.isBlank()) return null;
        try {
            return Integer.parseInt(cm.trim()) * 10;
        } catch (NumberFormatException e) {
            return null;
        }
    }
}
