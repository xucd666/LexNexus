package com.lexnexus.backend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.net.URI;
import java.time.Duration;

/**
 * AI-Service 的连接配置，绑定 application.properties 及其环境变量占位符。
 *
 * @param baseUrl 服务基础地址，要求 HTTP(S) 协议及有效主机
 * @param connectTimeout 建立连接的正数超时
 * @param readTimeout 等待和读取响应的正数超时
 */
@ConfigurationProperties("ai-service")
public record AiServiceProperties(URI baseUrl, Duration connectTimeout, Duration readTimeout) {

    /**
     * 在配置绑定阶段检查地址和时长，提前暴露无效配置。
     *
     * @throws IllegalArgumentException 地址不符合要求，或超时为空、零、负数
     */
    public AiServiceProperties {
        if (baseUrl == null || baseUrl.getHost() == null
                || !("http".equalsIgnoreCase(baseUrl.getScheme()) || "https".equalsIgnoreCase(baseUrl.getScheme()))
                || baseUrl.getUserInfo() != null || baseUrl.getQuery() != null || baseUrl.getFragment() != null) {
            throw new IllegalArgumentException("ai-service.base-url must be an HTTP(S) URL without credentials, query, or fragment");
        }
        requirePositive(connectTimeout, "ai-service.connect-timeout");
        requirePositive(readTimeout, "ai-service.read-timeout");
    }

    private static void requirePositive(Duration timeout, String property) {
        if (timeout == null || timeout.isZero() || timeout.isNegative()) {
            throw new IllegalArgumentException(property + " must be a positive duration");
        }
    }
}
