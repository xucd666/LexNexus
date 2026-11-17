package com.lexnexus.backend.dto.aiservice;

import java.util.List;

/**
 * 客户端解码后的 Agent 成功响应，由响应解码器完成字段类型与回显 ID 校验。
 *
 * @param requestId 已与发送值核对的请求标识
 * @param data Legal QA 的回答与引用
 */
public record AgentInvokeResponse(String requestId, LegalQaResult data) {

    /**
     * Agent 结果，回答原样保留。
     *
     * @param answer Agent 返回的回答
     * @param citations 引用列表，无引用时为空列表
     */
    public record LegalQaResult(String answer, List<Citation> citations) {
    }

    /**
     * 引用传输模型，各字段允许为空，snake_case JSON 名称由客户端解码器显式映射。
     *
     * @param documentId 来源文档标识，对应 document_id
     * @param kbChunkId 检索片段标识，对应 kb_chunk_id
     * @param lawName 法律或材料名称，对应 law_name
     * @param articleNo 条目编号，对应 article_no
     * @param quotedText 引用文本，对应 quoted_text
     * @param relevanceScore 相关度，对应 relevance_score
     */
    public record Citation(
            String documentId,
            String kbChunkId,
            String lawName,
            String articleNo,
            String quotedText,
            Double relevanceScore
    ) {
    }
}
