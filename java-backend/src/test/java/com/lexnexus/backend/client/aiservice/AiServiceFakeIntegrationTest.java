package com.lexnexus.backend.client.aiservice;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.lexnexus.backend.dto.aiservice.AgentContext;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse.Citation;
import com.lexnexus.backend.dto.aiservice.LegalQaInput;
import com.lexnexus.backend.exception.AiServiceException;
import com.lexnexus.backend.exception.AiServiceException.Kind;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.net.URI;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.catchThrowableOfType;

/**
 * 显式启用的本机 Python Fake 服务联调，使用 Spring 装配的真实客户端。
 * 启用参数为 -Dlexnexus.ai-service.integration=true，
 * 地址由 -Dlexnexus.ai-service.url=http://127.0.0.1:18080 指定。
 * phase=running 时要求 Fake 服务已启动，phase=stopped 时要求同一服务已停止。
 * 默认 Maven 测试跳过本类，避免依赖外部进程。
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE, properties = {
        "ai-service.base-url=${lexnexus.ai-service.url:http://127.0.0.1:18080}",
        "ai-service.connect-timeout=6s",
        "ai-service.read-timeout=8s"
})
@EnabledIfSystemProperty(named = "lexnexus.ai-service.integration", matches = "true")
class AiServiceFakeIntegrationTest {

    @Autowired
    private AiServiceClient client;

    @Autowired
    private ObjectMapper objectMapper;

    @BeforeEach
    void requireLoopbackTarget() {
        // 固定本机 HTTP 地址，避免测试参数误指向公网服务。
        URI target = URI.create(System.getProperty("lexnexus.ai-service.url", "http://127.0.0.1:18080"));
        assertThat(target.getScheme()).isEqualTo("http");
        assertThat(target.getHost()).isEqualTo("127.0.0.1");
    }

    @Test
    @EnabledIfSystemProperty(named = "lexnexus.ai-service.phase", matches = "running")
    void returnsFakeCitationAndEchoesRequestIdWithFullContext() throws Exception {
        AgentContext context = new AgentContext("j14-hit", "j14-trace", "1", "2", "3",
                Map.of("sample", objectMapper.readTree("[true,7,{\"name\":\"fixture\"},null]")));

        var response = client.invokeLegalQa(new LegalQaInput("合同违约金如何处理？"), context);

        assertThat(response.requestId()).isEqualTo("j14-hit");
        assertThat(response.data().answer()).isEqualTo("离线测试：已检索到测试材料，调用链路运行成功。");
        assertThat(response.data().citations()).containsExactly(new Citation(
                "fake-law-1", "fake-chunk-1", "LexNexus 离线测试材料", "测试条目 1",
                "这是用于验证违约金问题检索和引用传递的测试文本。", 1.0));
    }

    @Test
    @EnabledIfSystemProperty(named = "lexnexus.ai-service.phase", matches = "running")
    void returnsEmptyCitationsWhenRetrievalHasNoMatch() {
        var response = client.invokeLegalQa(new LegalQaInput("无命中样例"), context("j14-empty"));

        assertThat(response.requestId()).isEqualTo("j14-empty");
        assertThat(response.data().answer()).isEqualTo("未找到直接法律依据，无法给出确定性回答。");
        assertThat(response.data().citations()).isEmpty();
    }

    @Test
    @EnabledIfSystemProperty(named = "lexnexus.ai-service.phase", matches = "running")
    void classifiesRealAgentInputValidationError() {
        // 空白问题能通过客户端非空检查，由 Python 的 Agent 输入模型返回 422。
        AiServiceException failure = catchThrowableOfType(AiServiceException.class,
                () -> client.invokeLegalQa(new LegalQaInput("   "), context("j14-input-error")));

        assertThat(failure.getKind()).isEqualTo(Kind.INVALID_AGENT_INPUT);
        assertThat(failure.getHttpStatus()).isEqualTo(422);
        assertThat(failure.getRequestId()).isEqualTo("j14-input-error");
        assertThat(failure.getUpstreamError().code()).isEqualTo("INVALID_AGENT_INPUT");
        assertThat(failure.getUpstreamError().message()).isEqualTo("Agent 输入无效");
        assertThat(failure.getUpstreamError().details()).singleElement().satisfies(detail -> {
            assertThat(detail.loc()).extracting(node -> node.textValue()).containsExactly("input", "question");
            assertThat(detail.type()).isEqualTo("string_too_short");
        });
    }

    @Test
    @EnabledIfSystemProperty(named = "lexnexus.ai-service.phase", matches = "running")
    void classifiesRealEnvelopeValidationError() {
        // 空白 traceId 触发 Pydantic 请求上下文校验，验证与 Agent 输入错误的分类边界。
        AgentContext invalid = new AgentContext("j14-envelope-error", " ", null, null, null, null);
        AiServiceException failure = catchThrowableOfType(AiServiceException.class,
                () -> client.invokeLegalQa(new LegalQaInput("示例问题"), invalid));

        assertThat(failure.getKind()).isEqualTo(Kind.INVALID_REQUEST);
        assertThat(failure.getHttpStatus()).isEqualTo(422);
        assertThat(failure.getRequestId()).isEqualTo("j14-envelope-error");
        assertThat(failure.getUpstreamError().code()).isEqualTo("INVALID_REQUEST");
        assertThat(failure.getUpstreamError().details()).singleElement().satisfies(detail ->
                assertThat(detail.loc()).extracting(node -> node.textValue())
                        .containsExactly("body", "context", "traceId"));
    }

    @Test
    @EnabledIfSystemProperty(named = "lexnexus.ai-service.phase", matches = "stopped")
    void classifiesNetworkFailureAfterPythonServiceStops() {
        AiServiceException failure = catchThrowableOfType(AiServiceException.class,
                () -> client.invokeLegalQa(new LegalQaInput("示例问题"), context("j14-stopped")));

        assertThat(failure.getKind()).isEqualTo(Kind.NETWORK_UNREACHABLE);
        assertThat(failure.getRequestId()).isEqualTo("j14-stopped");
        assertThat(failure.getHttpStatus()).isNull();
        assertThat(failure.getUpstreamError()).isNull();
    }

    private static AgentContext context(String requestId) {
        return new AgentContext(requestId, null, null, null, null, null);
    }
}
