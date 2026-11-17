package com.lexnexus.backend.health;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/** 提供进程级 HTTP 健康检查，返回服务名称及当前可响应状态。 */
@RestController
public class HealthController {

    private final String serviceName;

    public HealthController(@Value("${spring.application.name}") String serviceName) {
        this.serviceName = serviceName;
    }

    /**
     * 返回当前服务的健康标识；UP 仅表示本接口能够响应请求。
     *
     * @return 服务名称与固定状态 UP
     */
    @GetMapping(path = "/api/v1/health", produces = MediaType.APPLICATION_JSON_VALUE)
    public HealthResponse health() {
        return new HealthResponse(serviceName, "UP");
    }

    /** 健康检查响应，service 来自 spring.application.name。 */
    public record HealthResponse(String service, String status) {
    }
}
