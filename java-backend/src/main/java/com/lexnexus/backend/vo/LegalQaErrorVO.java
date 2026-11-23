package com.lexnexus.backend.vo;

/**
 * 本地法律问答的错误响应，仅包含约定的错误码、安全消息和请求标识。
 *
 * @param code Java 法律问答错误码
 * @param message 面向调用方的固定消息
 * @param requestId 可用于追踪上游调用的标识，输入校验阶段未生成时为 null
 */
public record LegalQaErrorVO(String code, String message, String requestId) {
}
