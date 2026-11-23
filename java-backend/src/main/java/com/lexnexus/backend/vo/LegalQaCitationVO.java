package com.lexnexus.backend.vo;

/**
 * 本地法律问答返回的引用，保留 AI-Service 提供的字段及可空性。
 *
 * @param documentId 来源文档标识
 * @param kbChunkId 检索片段标识
 * @param lawName 法律或材料名称
 * @param articleNo 条目编号
 * @param quotedText 引用原文
 * @param relevanceScore 上游相关度分数
 */
public record LegalQaCitationVO(
        String documentId,
        String kbChunkId,
        String lawName,
        String articleNo,
        String quotedText,
        Double relevanceScore
) {
}
