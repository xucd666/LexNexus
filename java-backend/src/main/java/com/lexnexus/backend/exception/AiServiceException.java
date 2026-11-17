package com.lexnexus.backend.exception;

import com.lexnexus.backend.dto.aiservice.AiServiceError;

/**
 * AI-Service 调用失败的内部异常，携带追踪信息及可用的上游错误。
 * 默认异常消息仅含分类，业务层可据此决定后续处理和对外响应。
 */
public class AiServiceException extends RuntimeException {

    /** 区分已知上游语义、网络故障和响应契约问题。 */
    public enum Kind {
        /** Agent 对问题等业务输入的校验失败。 */
        INVALID_AGENT_INPUT,
        /** Python 对请求外层结构或上下文的校验失败。 */
        INVALID_REQUEST,
        /** 上游注册表中未找到指定 Agent。 */
        AGENT_NOT_FOUND,
        /** 模型调用失败或未提供有效回答。 */
        LLM_UNAVAILABLE,
        /** 明确的连接失败或地址解析失败。 */
        NETWORK_UNREACHABLE,
        /** 连接中断、流读取失败或无法确定具体原因的网络 I/O 错误。 */
        NETWORK_IO_ERROR,
        /** 原因链中包含明确的连接或读取超时异常。 */
        TIMEOUT,
        /** 实际读取的响应正文超过客户端字节上限。 */
        RESPONSE_TOO_LARGE,
        /** 非成功 HTTP 响应无法匹配已知错误语义。 */
        UNEXPECTED_HTTP_RESPONSE,
        /** 成功响应的状态、正文或回显请求标识违反契约。 */
        INVALID_RESPONSE
    }

    private final Kind kind;
    private final Integer httpStatus;
    private final String requestId;
    private final AiServiceError upstreamError;
    private final String ioExceptionType;

    /**
     * 创建内部错误，同时将上游原文与默认异常消息分开保存。
     *
     * @param kind 内部错误分类
     * @param httpStatus 已取得的上游 HTTP 状态，网络失败时可能为 null
     * @param requestId 本次调用发送的请求标识
     * @param upstreamError 已识别的结构化错误，无法识别时为 null
     */
    public AiServiceException(Kind kind, Integer httpStatus, String requestId, AiServiceError upstreamError) {
        this(kind, httpStatus, requestId, upstreamError, null);
    }

    /**
     * 为网络错误附加非敏感的诊断类型，异常消息仅使用内部分类。
     *
     * @param kind 内部错误分类
     * @param httpStatus 已取得的上游 HTTP 状态，尚未取得时为 null
     * @param requestId 本次调用发送的请求标识
     * @param upstreamError 可识别的结构化上游错误
     * @param ioExceptionType 底层异常类名，禁止传入异常消息或正文
     */
    public AiServiceException(Kind kind, Integer httpStatus, String requestId,
                              AiServiceError upstreamError, String ioExceptionType) {
        super("AI-Service call failed: " + kind);
        this.kind = kind;
        this.httpStatus = httpStatus;
        this.requestId = requestId;
        this.upstreamError = upstreamError;
        this.ioExceptionType = ioExceptionType;
    }

    public Kind getKind() {
        return kind;
    }

    public Integer getHttpStatus() {
        return httpStatus;
    }

    public String getRequestId() {
        return requestId;
    }

    public AiServiceError getUpstreamError() {
        return upstreamError;
    }

    /** @return 网络诊断异常类名，非网络异常通常为 null */
    public String getIoExceptionType() {
        return ioExceptionType;
    }
}
