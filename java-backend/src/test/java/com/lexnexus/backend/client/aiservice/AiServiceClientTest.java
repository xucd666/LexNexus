package com.lexnexus.backend.client.aiservice;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.lexnexus.backend.dto.aiservice.AgentContext;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse.Citation;
import com.lexnexus.backend.dto.aiservice.AiServiceError;
import com.lexnexus.backend.dto.aiservice.LegalQaInput;
import com.lexnexus.backend.exception.AiServiceException;
import com.lexnexus.backend.exception.AiServiceException.Kind;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.mock.http.client.MockClientHttpResponse;
import org.springframework.test.json.JsonCompareMode;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.test.web.client.ResponseActions;
import org.springframework.web.client.RestClient;

import java.io.IOException;
import java.io.InputStream;
import java.net.SocketTimeoutException;
import java.net.http.HttpConnectTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.catchThrowableOfType;
import static org.springframework.test.web.client.ExpectedCount.once;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

/**
 * 使用 MockRestServiceServer 检查请求 JSON、响应解码和错误分类。
 * 每次调用期望一次 POST，并在用例结束时验证请求次数。
 */
class AiServiceClientTest {

    private static final String BASE_URL = "http://127.0.0.1:8000";
    private static final String REQUEST_ID = "qa-test";
    private static final LegalQaInput INPUT = new LegalQaInput("示例问题");
    private static final AgentContext CONTEXT = new AgentContext(REQUEST_ID, null, null, null, null, null);

    private final ObjectMapper objectMapper = new ObjectMapper();
    private MockRestServiceServer server;
    private AiServiceClient client;

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder().baseUrl(BASE_URL);
        server = MockRestServiceServer.bindTo(builder).build();
        client = new AiServiceClient(builder.build(), objectMapper);
    }

    @AfterEach
    void verifyExactRequestCount() {
        server.verify();
    }

    @Test
    void postsCompleteRequestAndDecodesCitation() throws Exception {
        expectPost().andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(content().json("""
                        {
                          "agentCode":"legal_qa",
                          "input":{"question":"示例问题"},
                          "context":{
                            "requestId":"qa-test","traceId":"trace-2","userId":"1",
                            "sessionId":"2","taskId":"3",
                            "metadata":{"test":[true,7,{"label":"case"},null],"optional":null}
                          }
                        }
                        """, JsonCompareMode.STRICT))
                .andRespond(withSuccess("""
                        {"requestId":"qa-test","data":{
                          "answer":" 示例回答 ",
                          "citations":[{
                            "document_id":"law-1","kb_chunk_id":"chunk-1","law_name":"示例法",
                            "article_no":"第一条","quoted_text":"示例引文","relevance_score":0.87
                          }]
                        }}
                        """, MediaType.APPLICATION_JSON));

        AgentContext context = new AgentContext(
                REQUEST_ID, "trace-2", "1", "2", "3",
                Map.of("test", objectMapper.readTree("[true,7,{\"label\":\"case\"},null]"),
                        "optional", objectMapper.nullNode()));

        AgentInvokeResponse response = client.invokeLegalQa(INPUT, context);

        assertThat(response.requestId()).isEqualTo(REQUEST_ID);
        assertThat(response.data().answer()).isEqualTo(" 示例回答 ");
        assertThat(response.data().citations()).containsExactly(
                new Citation("law-1", "chunk-1", "示例法", "第一条", "示例引文", 0.87));
    }

    @Test
    void preservesEmptyCitationsAndOmitsAbsentContextFields() {
        expectPost().andExpect(content().json("""
                        {"agentCode":"legal_qa","input":{"question":"示例问题"},"context":{"requestId":"qa-test"}}
                        """, JsonCompareMode.STRICT))
                .andRespond(withSuccess("""
                        {"requestId":"qa-test","data":{"answer":"未找到直接法律依据","citations":[]}}
                        """, MediaType.APPLICATION_JSON));

        AgentInvokeResponse response = client.invokeLegalQa(INPUT, CONTEXT);

        assertThat(response.data().answer()).isEqualTo("未找到直接法律依据");
        assertThat(response.data().citations()).isEmpty();
    }

    @Test
    void preservesNullCitationFields() {
        expectPost().andRespond(withSuccess("""
                {"requestId":"qa-test","data":{"answer":"回答","citations":[{
                  "document_id":"law-1","kb_chunk_id":"chunk-1","law_name":null,
                  "article_no":null,"quoted_text":"引用","relevance_score":null
                }]}}
                """, MediaType.APPLICATION_JSON));

        assertThat(client.invokeLegalQa(INPUT, CONTEXT).data().citations()).containsExactly(
                new Citation("law-1", "chunk-1", null, null, "引用", null));
    }

    @Test
    void sendsNormalizedRequestIdForFastApiEcho() {
        expectPost().andExpect(content().json("""
                        {"agentCode":"legal_qa","input":{"question":"示例问题"},"context":{"requestId":"qa-test"}}
                        """, JsonCompareMode.STRICT))
                .andRespond(withSuccess("""
                        {"requestId":"qa-test","data":{"answer":"回答","citations":[]}}
                        """, MediaType.APPLICATION_JSON));

        AgentContext paddedId = new AgentContext("\u00a0 qa-test \t", null, null, null, null, null);

        assertThat(client.invokeLegalQa(INPUT, paddedId).requestId()).isEqualTo(REQUEST_ID);
    }

    @Test
    void rejectsMissingRequiredRequestValuesBeforeSending() {
        assertThatExceptionOfType(NullPointerException.class).isThrownBy(() -> new LegalQaInput(null));
        assertThatExceptionOfType(NullPointerException.class)
                .isThrownBy(() -> new AgentContext(null, null, null, null, null, null));
        assertThatExceptionOfType(IllegalArgumentException.class)
                .isThrownBy(() -> new AgentContext(" \t ", null, null, null, null, null));
        assertThatExceptionOfType(IllegalArgumentException.class)
                .isThrownBy(() -> new AgentContext("x".repeat(129), null, null, null, null, null));
        assertThatExceptionOfType(NullPointerException.class).isThrownBy(() -> client.invokeLegalQa(null, CONTEXT));
        assertThatExceptionOfType(NullPointerException.class).isThrownBy(() -> client.invokeLegalQa(INPUT, null));
    }

    @ParameterizedTest(name = "{0}")
    @ValueSource(strings = {"INVALID_AGENT_INPUT", "INVALID_REQUEST"})
    void preservesValidationErrorAndMixedLocation(String code) {
        expectPost().andRespond(withStatus(HttpStatus.UNPROCESSABLE_ENTITY).contentType(MediaType.APPLICATION_JSON)
                .body("""
                        {"success":false,"error":{"code":"%s","message":"校验失败",
                         "details":[{"loc":["input","items",2,"question"],"type":"string_type"}]}}
                        """.formatted(code)));

        AiServiceException failure = invokeExpectingFailure(Kind.valueOf(code), 422);

        assertThat(failure.getUpstreamError().code()).isEqualTo(code);
        assertThat(failure.getUpstreamError().message()).isEqualTo("校验失败");
        assertThat(failure.getUpstreamError().details()).singleElement().satisfies(detail -> {
            assertThat(detail.type()).isEqualTo("string_type");
            assertThat(detail.loc()).extracting(JsonNode::toString).containsExactly(
                    "\"input\"", "\"items\"", "2", "\"question\"");
            assertThat(detail.loc().get(2).isIntegralNumber()).isTrue();
        });
    }

    @Test
    void preservesUnknownAgentError() {
        expectPost().andRespond(withStatus(HttpStatus.NOT_FOUND).contentType(MediaType.APPLICATION_JSON)
                .body("""
                        {"success":false,"error":{"code":"AGENT_NOT_FOUND","message":"Agent not found: legal_qa",
                         "agentCode":"legal_qa"}}
                        """));

        assertThat(invokeExpectingFailure(Kind.AGENT_NOT_FOUND, 404).getUpstreamError())
                .isEqualTo(new AiServiceError("AGENT_NOT_FOUND", "Agent not found: legal_qa", "legal_qa", List.of()));
    }

    @Test
    void preservesModelUnavailableErrorWithoutRetrying() {
        expectPost().andRespond(withStatus(HttpStatus.SERVICE_UNAVAILABLE).contentType(MediaType.APPLICATION_JSON)
                .body("""
                        {"success":false,"error":{"code":"LLM_UNAVAILABLE","message":"模型暂不可用"}}
                        """));

        AiServiceError error = invokeExpectingFailure(Kind.LLM_UNAVAILABLE, 503).getUpstreamError();
        assertThat(error.code()).isEqualTo("LLM_UNAVAILABLE");
        assertThat(error.message()).isEqualTo("模型暂不可用");
        assertThat(error.details()).isEmpty();
    }

    @Test
    void handlesFrameworkNotFoundWithoutTreatingItAsUnknownAgent() {
        expectPost().andRespond(withStatus(HttpStatus.NOT_FOUND).contentType(MediaType.APPLICATION_JSON)
                .body("{\"detail\":\"Not Found\"}"));

        assertThat(invokeExpectingFailure(Kind.UNEXPECTED_HTTP_RESPONSE, 404).getUpstreamError()).isNull();
    }

    @Test
    void handlesPlainTextServerErrorWithoutExposingBody() {
        expectPost().andRespond(withStatus(HttpStatus.INTERNAL_SERVER_ERROR).contentType(MediaType.TEXT_PLAIN)
                .body("internal details"));

        AiServiceException failure = invokeExpectingFailure(Kind.UNEXPECTED_HTTP_RESPONSE, 500);

        assertThat(failure.getUpstreamError()).isNull();
        assertThat(failure).hasMessageNotContaining("internal details").hasNoCause();
    }

    @Test
    void preservesWellFormedUnknownErrorCodeWithFallbackClassification() {
        expectPost().andRespond(withStatus(HttpStatus.BAD_GATEWAY).contentType(MediaType.APPLICATION_JSON)
                .body("""
                        {"success":false,"error":{"code":"NEW_ERROR","message":"Upstream message"}}
                        """));

        AiServiceException failure = invokeExpectingFailure(Kind.UNEXPECTED_HTTP_RESPONSE, 502);

        assertThat(failure.getUpstreamError().code()).isEqualTo("NEW_ERROR");
        assertThat(failure.getUpstreamError().message()).isEqualTo("Upstream message");
        assertThat(failure).hasMessageNotContaining("Upstream message");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("invalidSuccessBodies")
    void rejectsInvalidSuccessResponse(String scenario, String body) {
        expectPost().andRespond(withSuccess(body, MediaType.APPLICATION_JSON));

        assertThat(invokeExpectingFailure(Kind.INVALID_RESPONSE, 200).getUpstreamError()).isNull();
    }

    private static Stream<Arguments> invalidSuccessBodies() {
        return Stream.of(
                Arguments.of("empty body", ""),
                Arguments.of("invalid JSON", "{broken"),
                Arguments.of("trailing JSON", "{\"requestId\":\"qa-test\",\"data\":{\"answer\":\"ok\",\"citations\":[]}} {}"),
                Arguments.of("missing requestId", "{\"data\":{\"answer\":\"ok\",\"citations\":[]}}"),
                Arguments.of("numeric requestId", "{\"requestId\":7,\"data\":{\"answer\":\"ok\",\"citations\":[]}}"),
                Arguments.of("missing data", "{\"requestId\":\"qa-test\"}"),
                Arguments.of("missing answer", "{\"requestId\":\"qa-test\",\"data\":{\"citations\":[]}}"),
                Arguments.of("null answer", "{\"requestId\":\"qa-test\",\"data\":{\"answer\":null,\"citations\":[]}}"),
                Arguments.of("numeric answer", "{\"requestId\":\"qa-test\",\"data\":{\"answer\":1,\"citations\":[]}}"),
                Arguments.of("missing citations", "{\"requestId\":\"qa-test\",\"data\":{\"answer\":\"ok\"}}"),
                Arguments.of("null citations", "{\"requestId\":\"qa-test\",\"data\":{\"answer\":\"ok\",\"citations\":null}}"),
                Arguments.of("object citations", "{\"requestId\":\"qa-test\",\"data\":{\"answer\":\"ok\",\"citations\":{}}}"),
                Arguments.of("null citation item", "{\"requestId\":\"qa-test\",\"data\":{\"answer\":\"ok\",\"citations\":[null]}}"),
                Arguments.of("numeric document ID",
                        "{\"requestId\":\"qa-test\",\"data\":{\"answer\":\"ok\",\"citations\":[{\"document_id\":7}]}}"),
                Arguments.of("string relevance score",
                        "{\"requestId\":\"qa-test\",\"data\":{\"answer\":\"ok\",\"citations\":[{\"relevance_score\":\"0.7\"}]}}"),
                Arguments.of("requestId mismatch", "{\"requestId\":\"other\",\"data\":{\"answer\":\"ok\",\"citations\":[]}}"));
    }

    @ParameterizedTest
    @ValueSource(ints = {200, 503})
    void acceptsResponseExactlyAtByteLimit(int status) {
        String json = status == 200
                ? "{\"requestId\":\"qa-test\",\"data\":{\"answer\":\"回答\",\"citations\":[]}}"
                : "{\"success\":false,\"error\":{\"code\":\"LLM_UNAVAILABLE\",\"message\":\"模型暂不可用\"}}";
        // 用 UTF-8 实际字节数填充边界，验证中文正文和合法 JSON 尾部空白的处理。
        byte[] body = Arrays.copyOf(json.getBytes(StandardCharsets.UTF_8), AiServiceClient.MAX_RESPONSE_BYTES);
        Arrays.fill(body, json.getBytes(StandardCharsets.UTF_8).length, body.length, (byte) ' ');
        expectPost().andRespond(withStatus(HttpStatus.valueOf(status)).body(body).contentType(MediaType.APPLICATION_JSON));

        if (status == 200) {
            assertThat(client.invokeLegalQa(INPUT, CONTEXT).data().answer()).isEqualTo("回答");
        } else {
            assertThat(invokeExpectingFailure(Kind.LLM_UNAVAILABLE, status).getUpstreamError().message())
                    .isEqualTo("模型暂不可用");
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {200, 500})
    void rejectsOversizedResponseAndClosesStreamWithoutReadingRemainder(int status) {
        // 无穷流模拟超大响应，验证客户端最多读取上限加一个字节且能关闭响应。
        class CountingStream extends InputStream {
            int count;
            boolean closed;

            @Override
            public int read() {
                count++;
                return ' ';
            }

            @Override
            public int read(byte[] buffer, int offset, int length) {
                Arrays.fill(buffer, offset, offset + length, (byte) ' ');
                count += length;
                return length;
            }

            @Override
            public void close() {
                closed = true;
            }
        }
        CountingStream body = new CountingStream();
        expectPost().andRespond(request -> {
            MockClientHttpResponse response = new MockClientHttpResponse(body, HttpStatus.valueOf(status));
            // 即使上游声明了错误的小长度，实际正文也必须受限。
            response.getHeaders().setContentLength(1);
            return response;
        });

        AiServiceException failure = invokeExpectingFailure(Kind.RESPONSE_TOO_LARGE, status);

        assertThat(failure.getUpstreamError()).isNull();
        assertThat(failure).hasMessage("AI-Service call failed: RESPONSE_TOO_LARGE").hasNoCause();
        assertThat(body.count).isEqualTo(AiServiceClient.MAX_RESPONSE_BYTES + 1);
        assertThat(body.closed).isTrue();
    }

    @ParameterizedTest
    @MethodSource("bodyReadFailures")
    void classifiesBodyReadFailureUsingCauseChain(IOException cause, Kind expectedKind, String diagnosticType) {
        InputStream body = new InputStream() {
            @Override
            public int read() throws IOException {
                throw cause;
            }
        };
        expectPost().andRespond(request -> new MockClientHttpResponse(body, HttpStatus.OK));

        AiServiceException failure = invokeExpectingFailure(expectedKind, 200);

        assertThat(failure.getIoExceptionType()).isEqualTo(diagnosticType);
        assertThat(failure).hasMessageNotContaining("private").hasNoCause();
    }

    private static Stream<Arguments> bodyReadFailures() {
        return Stream.of(
                Arguments.of(new IOException("private", new SocketTimeoutException("private")),
                        Kind.TIMEOUT, SocketTimeoutException.class.getName()),
                Arguments.of(new IOException("private", new HttpConnectTimeoutException("private")),
                        Kind.TIMEOUT, HttpConnectTimeoutException.class.getName()),
                Arguments.of(new IOException("private connection closed"), Kind.NETWORK_IO_ERROR, IOException.class.getName()));
    }

    private ResponseActions expectPost() {
        return server.expect(once(), requestTo(BASE_URL + "/api/v1/agent/invoke"))
                .andExpect(method(HttpMethod.POST));
    }

    private AiServiceException invokeExpectingFailure(Kind expectedKind, int expectedStatus) {
        AiServiceException failure = catchThrowableOfType(
                AiServiceException.class, () -> client.invokeLegalQa(INPUT, CONTEXT));

        assertThat(failure).isNotNull();
        assertThat(failure.getKind()).isEqualTo(expectedKind);
        assertThat(failure.getHttpStatus()).isEqualTo(expectedStatus);
        assertThat(failure.getRequestId()).isEqualTo(REQUEST_ID);
        return failure;
    }
}
