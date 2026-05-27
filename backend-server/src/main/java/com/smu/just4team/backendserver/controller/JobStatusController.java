package com.smu.just4team.backendserver.controller;

import com.smu.just4team.backendserver.service.AiServerClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

// 브라우저(result.jsp)가 폴링하는 프록시 엔드포인트.
// ai-server /jobs/{id} JSON을 그대로 전달.
@RestController
public class JobStatusController {

    @Autowired
    private AiServerClient aiServerClient;

    @GetMapping(value = "/api/job-status/{jobId}", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> status(@PathVariable("jobId") String jobId) {
        try {
            String raw = aiServerClient.fetchJobStatusRaw(jobId);
            return ResponseEntity.ok(raw);
        } catch (Exception e) {
            String body = "{\"status\":\"error\",\"error\":\""
                    + e.getMessage().replace("\"", "'") + "\"}";
            return ResponseEntity.status(502).body(body);
        }
    }
}
