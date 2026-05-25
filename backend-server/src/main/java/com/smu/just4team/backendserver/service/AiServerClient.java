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

    @Value("${ai.server.default-desk-width-mm:1200}")
    private int defaultDeskWidthMm;

    @Value("${ai.server.default-desk-depth-mm:600}")
    private int defaultDeskDepthMm;

    @Value("${ai.server.submit-timeout-ms:30000}")
    private int submitTimeoutMs;

    public AiServerClient() {
        SimpleClientHttpRequestFactory f = new SimpleClientHttpRequestFactory();
        f.setConnectTimeout(Duration.ofSeconds(10));
        f.setReadTimeout(Duration.ofSeconds(30));
        this.restTemplate = new RestTemplate(f);
    }

    // POST /recommend-and-generate → job_id 반환.
    // deskWidthMm / deskDepthMm null이면 application.properties default 사용.
    public String submitRecommendAndGenerate(
            String theme,
            int budget,
            String frontImageBase64,
            String topImageBase64Nullable,
            Integer deskWidthMmNullable,
            Integer deskDepthMmNullable
    ) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("theme",        theme);
        body.put("budget",       budget);
        body.put("image_base64", frontImageBase64);
        body.put("desk_width_mm", deskWidthMmNullable != null ? deskWidthMmNullable : defaultDeskWidthMm);
        body.put("desk_depth_mm", deskDepthMmNullable != null ? deskDepthMmNullable : defaultDeskDepthMm);
        if (topImageBase64Nullable != null && !topImageBase64Nullable.isEmpty()) {
            body.put("top_view_image_base64", topImageBase64Nullable);
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
}
