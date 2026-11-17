package com.lexnexus.backend.dto.aiservice;

import java.util.Objects;

/**
 * Legal QA 的问题输入；空白、长度等内容约束由 Python Agent 输入模型校验。
 *
 * @param question 调用方提供的问题文本
 */
public record LegalQaInput(String question) {

    /**
     * 确保请求包含问题字段。
     *
     * @throws NullPointerException question 为 null
     */
    public LegalQaInput {
        Objects.requireNonNull(question, "question is required");
    }
}
