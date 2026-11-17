package com.lexnexus.backend.dto.aiservice;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * 对应 POST /api/v1/agent/invoke 的请求外层结构。
 *
 * @param agentCode Agent 代码，当前客户端固定传入 legal_qa
 * @param input Legal QA 问题输入
 * @param context 请求追踪及可选业务上下文
 */
public record AgentInvokeRequest(@JsonProperty("agentCode") String agentCode, LegalQaInput input, AgentContext context) {
}
