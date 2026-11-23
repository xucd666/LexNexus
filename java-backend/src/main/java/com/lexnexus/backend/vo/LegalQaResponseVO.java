package com.lexnexus.backend.vo;

import java.util.List;

/**
 * 本地法律问答的成功响应，不依赖内部 AI-Service 传输类型。
 *
 * @param requestId 与本次上游调用共享的请求标识
 * @param answer AI-Service 返回的回答
 * @param citations 引用列表，无引用时序列化为空数组
 */
public record LegalQaResponseVO(String requestId, String answer, List<LegalQaCitationVO> citations) {
}
