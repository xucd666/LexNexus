package com.lexnexus.backend.dto.aiservice;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * 随 Agent 请求传输的上下文，身份和业务标识均由调用方提供。
 * 非空标识符使用 camelCase JSON 字段，空的可选字段在序列化时省略。
 *
 * @param requestId 必填请求标识，去除首尾空白后用于发送和响应核对
 * @param traceId 可选链路标识
 * @param userId 可选用户标识
 * @param sessionId 可选会话标识
 * @param taskId 可选任务标识
 * @param metadata 可选附加 JSON 数据，保留数字、布尔、对象、数组和 null 等类型
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record AgentContext(
        @JsonProperty("requestId") String requestId,
        @JsonProperty("traceId") String traceId,
        @JsonProperty("userId") String userId,
        @JsonProperty("sessionId") String sessionId,
        @JsonProperty("taskId") String taskId,
        Map<String, JsonNode> metadata
) {
    // 对齐 Python str.strip() 的空白范围，使发送的请求 ID 与 FastAPI 回显值一致。
    private static final Pattern EDGE_WHITESPACE = Pattern.compile("(?U)^[\\s\\x1c-\\x1f]+|[\\s\\x1c-\\x1f]+$");

    /**
     * 规范化请求标识；可选标识符的约束由上游 Pydantic 校验。
     *
     * @throws NullPointerException requestId 为 null
     * @throws IllegalArgumentException requestId 去除首尾空白后为空或超过 128 个 Unicode 码点
     */
    public AgentContext {
        Objects.requireNonNull(requestId, "requestId is required");
        requestId = EDGE_WHITESPACE.matcher(requestId).replaceAll("");
        if (requestId.isEmpty() || requestId.codePointCount(0, requestId.length()) > 128) {
            throw new IllegalArgumentException("requestId must contain 1 to 128 characters after trimming");
        }
    }
}
