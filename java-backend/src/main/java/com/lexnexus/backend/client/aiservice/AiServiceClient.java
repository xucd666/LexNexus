package com.lexnexus.backend.client.aiservice;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.lexnexus.backend.dto.aiservice.AgentContext;
import com.lexnexus.backend.dto.aiservice.AgentInvokeRequest;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse;
import com.lexnexus.backend.dto.aiservice.LegalQaInput;
import com.lexnexus.backend.exception.AiServiceException;
import com.lexnexus.backend.exception.AiServiceException.Kind;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;

import java.io.IOException;
import java.net.ConnectException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.net.http.HttpTimeoutException;
import java.nio.channels.UnresolvedAddressException;
import java.util.Objects;

/**
 * AI-Service 的同步 HTTP 调用组件，负责请求发送、响应校验及内部错误分类。
 * 当前支持 legal_qa，业务身份和调用上下文由上层提供，Agent 执行由 Python 服务负责。
 */
public class AiServiceClient {

    /** 单次响应正文的字节上限，成功响应与错误响应共用。 */
    static final int MAX_RESPONSE_BYTES = 2 * 1024 * 1024;

    private final RestClient restClient;
    private final AiServiceResponseDecoder decoder;

    /**
     * 接收已完成网络配置的客户端及 JSON 解析器。
     *
     * @param restClient 已设置 Base URL、连接及读取超时的 RestClient
     * @param objectMapper 用于构造独立响应读取器的 JSON 配置
     */
    public AiServiceClient(RestClient restClient, ObjectMapper objectMapper) {
        this.restClient = restClient;
        this.decoder = new AiServiceResponseDecoder(objectMapper);
    }

    /**
     * 调用 legal_qa 并核对响应中的 requestId，保留 Agent 返回的回答和引用。
     *
     * @param input 调用方提供的问题；问题内容约束由 AI-Service 校验
     * @param context 调用上下文，必须包含调用方提供的 requestId
     * @return 通过契约校验的响应，包含 requestId、回答和引用列表
     * @throws NullPointerException input 或 context 为 null
     * @throws AiServiceException 网络失败、超时、响应超限、上游 HTTP 错误或响应契约不匹配
     */
    public AgentInvokeResponse invokeLegalQa(LegalQaInput input, AgentContext context) {
        Objects.requireNonNull(input, "input is required");
        Objects.requireNonNull(context, "context is required");
        String requestId = context.requestId();

        try {
            return restClient.post()
                    .uri("/api/v1/agent/invoke")
                    .contentType(MediaType.APPLICATION_JSON)
                    .accept(MediaType.APPLICATION_JSON)
                    .body(new AgentInvokeRequest("legal_qa", input, context))
                    // exchange 交由本类处理全部状态码，以保留 Python 结构化错误。
                    .exchange((request, response) -> readResponse(response, requestId));
        } catch (ResourceAccessException failure) {
            throw networkFailure(failure, null, requestId);
        }
    }

    /** 读取完整响应后按状态码解析；已取得的 HTTP 状态也会保留在读取失败异常中。 */
    private AgentInvokeResponse readResponse(ClientHttpResponse response, String requestId) throws IOException {
        HttpStatusCode status = response.getStatusCode();
        byte[] body;
        try {
            // 多读取一个字节用于判定越界，按实际流量限制大小，兼容分块传输和缺失的 Content-Length。
            // exchange 结束时由 RestClient 关闭响应，包括超限和读取失败的分支。
            body = response.getBody().readNBytes(MAX_RESPONSE_BYTES + 1);
        } catch (IOException failure) {
            throw networkFailure(failure, status.value(), requestId);
        }
        if (body.length > MAX_RESPONSE_BYTES) {
            // 在任何 JSON 解析之前拒绝整个响应，防止截断后的成功体或错误体被误用。
            throw new AiServiceException(Kind.RESPONSE_TOO_LARGE, status.value(), requestId, null);
        }

        if (status.value() == HttpStatus.OK.value()) {
            return decoder.decodeSuccess(body, status.value(), requestId);
        }
        if (status.is2xxSuccessful()) {
            throw new AiServiceException(Kind.INVALID_RESPONSE, status.value(), requestId, null);
        }
        throw decoder.decodeHttpError(body, status.value(), requestId);
    }

    /**
     * 根据异常类型及原因链分类，明确的超时证据优先于连接失败。
     * Spring 在正文读取超时时可能仅关闭流并抛出 IOException，此时保留网络 I/O 后备分类。
     */
    private static AiServiceException networkFailure(Throwable failure, Integer status, String requestId) {
        Kind kind = Kind.NETWORK_IO_ERROR;
        String exceptionType = failure.getClass().getName();
        for (Throwable cause = failure; cause != null; cause = cause.getCause()) {
            if (cause instanceof HttpTimeoutException || cause instanceof SocketTimeoutException) {
                return new AiServiceException(Kind.TIMEOUT, status, requestId, null, cause.getClass().getName());
            }
            if (cause instanceof ConnectException || cause instanceof UnknownHostException
                    || cause instanceof UnresolvedAddressException) {
                kind = Kind.NETWORK_UNREACHABLE;
                exceptionType = cause.getClass().getName();
            }
        }
        // 仅保留异常类名，避免底层异常消息和原因链携带地址、请求或响应片段。
        return new AiServiceException(kind, status, requestId, null, exceptionType);
    }
}
