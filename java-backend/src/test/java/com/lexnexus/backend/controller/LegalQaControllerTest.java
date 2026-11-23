package com.lexnexus.backend.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.lexnexus.backend.client.aiservice.AiServiceClient;
import com.lexnexus.backend.dto.aiservice.AgentContext;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse.Citation;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse.LegalQaResult;
import com.lexnexus.backend.dto.aiservice.AiServiceError;
import com.lexnexus.backend.dto.aiservice.LegalQaInput;
import com.lexnexus.backend.exception.AiServiceException;
import com.lexnexus.backend.exception.AiServiceException.Kind;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.web.ServerProperties;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.json.JsonCompareMode;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.nullValue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 使用 local 应用上下文验证 HTTP 入参、Service 编排和错误响应，跨服务调用由 Mock 替代。 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("local")
class LegalQaControllerTest {

    private static final String PATH = "/api/v1/dev/legal-qa";

    @Autowired
    private MockMvc mvc;

    @Autowired
    private ObjectMapper json;

    @Autowired
    private ServerProperties serverProperties;

    @MockitoBean
    private AiServiceClient client;

    @Test
    void registersLocalEndpointAndMapsAnswerAndCitation() throws Exception {
        assertThat(serverProperties.getAddress().getHostAddress()).isEqualTo("127.0.0.1");
        var citation = new Citation("doc", "chunk", "示例法规", "第一条", "测试引用原文。", 0.75);
        mockAnswerWith(List.of(citation));

        var result = postQuestion("\t\u00a0合同不履行有什么后果？\u3000\n")
                .andExpect(status().isOk());
        var context = ArgumentCaptor.forClass(AgentContext.class);
        verify(client).invokeLegalQa(eq(new LegalQaInput("合同不履行有什么后果？")), context.capture());
        String requestId = context.getValue().requestId();
        assertThat(UUID.fromString(requestId).version()).isEqualTo(4);

        result.andExpect(content().json("""
                {
                  "requestId":"%s",
                  "answer":"示例回答。",
                  "citations":[{
                    "documentId":"doc", "kbChunkId":"chunk", "lawName":"示例法规",
                    "articleNo":"第一条", "quotedText":"测试引用原文。", "relevanceScore":0.75
                  }]
                }
                """.formatted(requestId), JsonCompareMode.STRICT));
    }

    @Test
    void accepts4000CodePointsAndKeepsEmptyCitationsAndDistinctRequestIds() throws Exception {
        mockAnswerWith(List.of());
        String question = "法".repeat(3999) + "😀";

        var first = postQuestion(" " + question + " ").andExpect(status().isOk());
        var second = postQuestion(question).andExpect(status().isOk());

        var context = ArgumentCaptor.forClass(AgentContext.class);
        verify(client, times(2)).invokeLegalQa(eq(new LegalQaInput(question)), context.capture());
        String firstId = context.getAllValues().get(0).requestId();
        String secondId = context.getAllValues().get(1).requestId();
        assertThat(firstId).isNotEqualTo(secondId);
        first.andExpect(jsonPath("$.requestId").value(firstId))
                .andExpect(jsonPath("$.citations").isArray())
                .andExpect(jsonPath("$.citations").isEmpty());
        second.andExpect(jsonPath("$.requestId").value(secondId))
                .andExpect(jsonPath("$.citations").isArray())
                .andExpect(jsonPath("$.citations").isEmpty());
    }

    @ParameterizedTest
    @ValueSource(strings = {"{\"question\":\" \\t\\n\"}", "{\"question\":\"\u3000\u00a0\"}", "{}", "{\"question\":null}",
            "{\"question\":123}", "{\"question\":true}", "{\"question\":[]}", "{\"question\":"})
    void rejectsInvalidQuestionBeforeCallingClient(String body) throws Exception {
        expectInvalidInput(mvc.perform(post(PATH).contentType(APPLICATION_JSON).content(body)));
        verifyNoInteractions(client);
    }

    @Test
    void rejectsQuestionLongerThan4000CodePoints() throws Exception {
        expectInvalidInput(postQuestion("法".repeat(4001)));
        verifyNoInteractions(client);
    }

    @Test
    void trimsUnicodeEdgesAndKeepsWhitespaceInsideTheQuestion() throws Exception {
        mockAnswerWith(List.of());
        String question = "😀合同\u0085\u00a0  履行😀";

        postQuestion("\u0085\u202f" + question + "\u2029\u00a0").andExpect(status().isOk());

        verify(client).invokeLegalQa(eq(new LegalQaInput(question)), any(AgentContext.class));
    }

    @Test
    void rejects4001CodePointsAfterTrimmingSupplementaryCharactersInput() throws Exception {
        expectInvalidInput(postQuestion("\u202f" + "😀".repeat(4001) + "\u3000"));
        verifyNoInteractions(client);
    }

    @Test
    @Timeout(value = 5, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void rejectsOverlongQuestionWithAContinuousInteriorWhitespaceRun() throws Exception {
        // 保留两个非空白端点，覆盖旧正则为尾部匹配反复扫描长内部空白的场景。
        expectInvalidInput(postQuestion("合同" + " ".repeat(100_000) + "履行"));
        verifyNoInteractions(client);
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', textBlock = """
            LLM_UNAVAILABLE | 503 | QA_MODEL_UNAVAILABLE | 问答模型暂不可用。
            NETWORK_UNREACHABLE | 503 | QA_SERVICE_UNAVAILABLE | 问答服务暂不可达。
            TIMEOUT | 504 | QA_UPSTREAM_TIMEOUT | 等待问答服务响应超时。
            NETWORK_IO_ERROR | 502 | QA_UPSTREAM_FAILURE | 未能从问答服务获取有效结果。
            INVALID_RESPONSE | 502 | QA_UPSTREAM_FAILURE | 未能从问答服务获取有效结果。
            RESPONSE_TOO_LARGE | 502 | QA_UPSTREAM_FAILURE | 未能从问答服务获取有效结果。
            UNEXPECTED_HTTP_RESPONSE | 502 | QA_UPSTREAM_FAILURE | 未能从问答服务获取有效结果。
            INVALID_REQUEST | 502 | QA_UPSTREAM_CONTRACT_ERROR | 问答服务未接受内部请求。
            INVALID_AGENT_INPUT | 502 | QA_UPSTREAM_CONTRACT_ERROR | 问答服务未接受内部请求。
            AGENT_NOT_FOUND | 502 | QA_UPSTREAM_CONTRACT_ERROR | 问答服务未接受内部请求。
            """)
    void mapsUpstreamFailureWithoutExposingInternalData(Kind kind, int httpStatus, String code, String message)
            throws Exception {
        var upstream = new AiServiceError("INTERNAL_PYTHON_CODE",
                "Traceback: E:\\private\\settings.env (secret=test-api-key)", "legal_qa",
                List.of(new AiServiceError.Detail(List.of(json.valueToTree("raw_error_body")), "internal_type")));
        when(client.invokeLegalQa(any(), any())).thenAnswer(call -> {
            AgentContext context = call.getArgument(1);
            throw new AiServiceException(kind, 500, context.requestId(), upstream, "internal.Exception");
        });

        var result = postQuestion("示例问题").andExpect(status().is(httpStatus));
        var context = ArgumentCaptor.forClass(AgentContext.class);
        verify(client).invokeLegalQa(eq(new LegalQaInput("示例问题")), context.capture());
        result.andExpect(content().json(json.writeValueAsString(Map.of(
                "code", code, "message", message, "requestId", context.getValue().requestId()
        )), JsonCompareMode.STRICT));
    }

    private void mockAnswerWith(List<Citation> citations) {
        when(client.invokeLegalQa(any(), any())).thenAnswer(call -> {
            AgentContext context = call.getArgument(1);
            return new AgentInvokeResponse(context.requestId(), new LegalQaResult("示例回答。", citations));
        });
    }

    private ResultActions postQuestion(String question) throws Exception {
        return mvc.perform(post(PATH).contentType(APPLICATION_JSON)
                .content(json.writeValueAsString(Map.of("question", question))));
    }

    private void expectInvalidInput(ResultActions result) throws Exception {
        result.andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("QA_INVALID_INPUT"))
                .andExpect(jsonPath("$.requestId").value(nullValue()));
    }
}
