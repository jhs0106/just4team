package com.smu.just4team.backendserver.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.RestTemplate;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.http.client.SimpleClientHttpRequestFactory;

// ai-server (FastAPI uvicorn :8000) HTTP 호출 헬퍼.
// /recommend-and-generate 호출 → job_id 받기 / /jobs/{id} 폴링 프록시.
@Service
public class AiServerClient {

    private final RestTemplate restTemplate;
    private final ObjectMapper mapper = new ObjectMapper();

    @Value("${ai.server.base-url:http://127.0.0.1:8000}")
    private String baseUrl;

    @Value("${ai.server.submit-timeout-ms:30000}")
    private int submitTimeoutMs;

    public AiServerClient() {
        SimpleClientHttpRequestFactory f = new SimpleClientHttpRequestFactory();
        f.setConnectTimeout(Duration.ofSeconds(10));
        f.setReadTimeout(Duration.ofSeconds(30));
        this.restTemplate = new RestTemplate(f);
    }

    // POST /recommend-and-generate → job_id 반환.
    // deskWidthMm / deskDepthMm: 필수 (Controller에서 검증됨). 사용자 입력 그대로 전달, default 폴백 X.
    // topImageBase64: 필수 (Controller에서 검증됨).
    // modeNullable: add | own_desk | replace | empty_desk
    // deskClickX/Y: 빈 책상 모드(add) 보조 입력 (옵션)
    public String submitRecommendAndGenerate(
            String theme,
            int budget,
            String frontImageBase64,
            String topImageBase64,
            int deskWidthMm,
            int deskDepthMm,
            String modeNullable,
            Double deskClickXNullable,
            Double deskClickYNullable,
            String deskCornersJsonNullable,
            String keepPointsJsonNullable
    ) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("theme",                 theme);
        body.put("budget",                budget);
        body.put("image_base64",          frontImageBase64);
        body.put("desk_width_mm",         deskWidthMm);
        body.put("desk_depth_mm",         deskDepthMm);
        body.put("top_view_image_base64", topImageBase64);
        if (modeNullable != null && !modeNullable.isEmpty()) {
            body.put("mode", modeNullable);
        }
        if (deskClickXNullable != null && deskClickYNullable != null) {
            body.put("desk_click_x", deskClickXNullable);
            body.put("desk_click_y", deskClickYNullable);
        }
        if (deskCornersJsonNullable != null && !deskCornersJsonNullable.isBlank()) {
            // JSP가 보낸 "[[x,y],...]" (0~1 정규화 4점) → desk_corners 배열로 전달
            try {
                body.put("desk_corners", mapper.readValue(deskCornersJsonNullable, java.util.List.class));
            } catch (Exception ignore) {}
        }
        if (keepPointsJsonNullable != null && !keepPointsJsonNullable.isBlank()) {
            // 남길 기존 제품 탭 좌표 "[[x,y],...]" → keep_points (own_desk 선택 제거용)
            try {
                body.put("keep_points", mapper.readValue(keepPointsJsonNullable, java.util.List.class));
            } catch (Exception ignore) {}
        }

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> req = new HttpEntity<>(body, headers);

        String url = baseUrl + "/recommend-and-generate";
        try {
            ResponseEntity<String> resp = restTemplate.exchange(url, HttpMethod.POST, req, String.class);
            if (!resp.getStatusCode().is2xxSuccessful() || resp.getBody() == null) {
                throw new IllegalStateException("ai-server submit failed: " + resp.getStatusCode());
            }
            JsonNode json = mapper.readTree(resp.getBody());
            String jobId = json.path("job_id").asText(null);
            if (jobId == null || jobId.isEmpty()) {
                throw new IllegalStateException("ai-server submit: job_id 없음 (body=" + resp.getBody() + ")");
            }
            return jobId;
        } catch (HttpServerErrorException e) {
            throw new IllegalStateException("ai-server 5xx: " + e.getResponseBodyAsString(), e);
        } catch (Exception e) {
            throw new IllegalStateException("ai-server submit 실패: " + e.getMessage(), e);
        }
    }

    // GET /jobs/{jobId} raw JSON 반환 (Spring → 브라우저 폴링 프록시용).
    public String fetchJobStatusRaw(String jobId) {
        String url = baseUrl + "/jobs/" + URLEncoder.encode(jobId, StandardCharsets.UTF_8);
        ResponseEntity<String> resp = restTemplate.getForEntity(url, String.class);
        if (!resp.getStatusCode().is2xxSuccessful() || resp.getBody() == null) {
            throw new IllegalStateException("ai-server jobs fetch failed: " + resp.getStatusCode());
        }
        return resp.getBody();
    }

    // POST /detect-objects → 정면 사진 검출 박스 raw JSON 반환 (남길 제품 클릭 선택용).
    public String detectObjects(String imageBase64) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("image_base64", imageBase64);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> req = new HttpEntity<>(body, headers);

        String url = baseUrl + "/detect-objects";
        ResponseEntity<String> resp = restTemplate.exchange(url, HttpMethod.POST, req, String.class);
        if (!resp.getStatusCode().is2xxSuccessful() || resp.getBody() == null) {
            throw new IllegalStateException("ai-server detect failed: " + resp.getStatusCode());
        }
        return resp.getBody();
    }
}
