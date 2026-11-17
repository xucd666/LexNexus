package com.lexnexus.backend.dto.aiservice;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.List;

/**
 * 可识别的上游结构化错误，供内部调用方诊断和后续业务层转换。
 *
 * @param code 上游错误码
 * @param message 上游消息原文，使用时需考虑其内容是否适合对外展示或记录
 * @param agentCode 可用的 Agent 标识，通常由 AGENT_NOT_FOUND 响应提供
 * @param details 符合契约的校验详情，缺失或无法提取时为空列表
 */
public record AiServiceError(String code, String message, String agentCode, List<Detail> details) {

    /**
     * Pydantic 字段校验详情。
     *
     * @param loc 字段路径，JsonNode 保留字符串字段名和整数索引的区别
     * @param type 校验错误类型
     */
    public record Detail(List<JsonNode> loc, String type) {
    }
}
