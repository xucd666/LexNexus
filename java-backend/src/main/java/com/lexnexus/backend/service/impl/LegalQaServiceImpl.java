package com.lexnexus.backend.service.impl;

import com.lexnexus.backend.client.aiservice.AiServiceClient;
import com.lexnexus.backend.dto.aiservice.AgentContext;
import com.lexnexus.backend.dto.aiservice.LegalQaInput;
import com.lexnexus.backend.service.LegalQaService;
import com.lexnexus.backend.vo.LegalQaCitationVO;
import com.lexnexus.backend.vo.LegalQaResponseVO;
import org.springframework.stereotype.Service;

import java.util.UUID;

/** 负责单次法律问答的调用编排及对外结果映射，Agent 执行由 AI-Service 负责。 */
@Service
public class LegalQaServiceImpl implements LegalQaService {

    private final AiServiceClient client;

    public LegalQaServiceImpl(AiServiceClient client) {
        this.client = client;
    }

    /** {@inheritDoc} 问题和追踪标识通过现有客户端发送，返回的引用保持原有顺序。 */
    @Override
    public LegalQaResponseVO answer(String question) {
        // 由 Java 为每次调用生成自己的追踪标识，Client 会核对上游回显。
        String requestId = UUID.randomUUID().toString();
        AgentContext context = new AgentContext(requestId, null, null, null, null, null);
        var response = client.invokeLegalQa(new LegalQaInput(question), context);
        var data = response.data();
        var citations = data.citations().stream()
                .map(citation -> new LegalQaCitationVO(
                        citation.documentId(), citation.kbChunkId(), citation.lawName(),
                        citation.articleNo(), citation.quotedText(), citation.relevanceScore()))
                .toList();
        return new LegalQaResponseVO(response.requestId(), data.answer(), citations);
    }
}
