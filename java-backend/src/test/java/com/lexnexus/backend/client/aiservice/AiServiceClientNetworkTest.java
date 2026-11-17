package com.lexnexus.backend.client.aiservice;

import com.lexnexus.backend.config.AiServiceClientConfiguration;
import com.lexnexus.backend.dto.aiservice.AgentContext;
import com.lexnexus.backend.dto.aiservice.LegalQaInput;
import com.lexnexus.backend.exception.AiServiceException;
import com.lexnexus.backend.exception.AiServiceException.Kind;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.autoconfigure.http.HttpMessageConvertersAutoConfiguration;
import org.springframework.boot.autoconfigure.jackson.JacksonAutoConfiguration;
import org.springframework.boot.autoconfigure.web.client.RestClientAutoConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.catchThrowableOfType;

/**
 * 使用本机 HTTP 服务和生产请求工厂验证连接失败、读取超时及 POST 发送次数。
 * 各用例独立分配端口，通过 Spring 配置创建客户端以覆盖真实装配路径。
 */
@Timeout(12)
class AiServiceClientNetworkTest {

    private static final String REQUEST_ID = "network-test";
    private static final LegalQaInput INPUT = new LegalQaInput("示例问题");
    private static final AgentContext CONTEXT = new AgentContext(REQUEST_ID, null, null, null, null, null);
    private static final String INVOKE_PATH = "/api/v1/agent/invoke";

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withConfiguration(AutoConfigurations.of(
                    JacksonAutoConfiguration.class, HttpMessageConvertersAutoConfiguration.class,
                    RestClientAutoConfiguration.class))
            .withUserConfiguration(AiServiceClientConfiguration.class)
            .withPropertyValues("ai-service.connect-timeout=1s", "ai-service.read-timeout=1s");

    private final AtomicInteger posts = new AtomicInteger();
    private HttpServer server;
    private String baseUrl;

    @BeforeEach
    void createLocalServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        baseUrl = "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @AfterEach
    void stopLocalServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void connectionRefusedIsNetworkUnreachable() {
        // 先启动再释放端口，让客户端实际经历本机连接被拒绝的过程。
        server.start();
        server.stop(0);
        server = null;

        // Windows 可能延迟报告连接拒绝，放宽本用例时限以区分连接失败与超时。
        configuredRunner()
                .withPropertyValues("ai-service.connect-timeout=6s", "ai-service.read-timeout=8s")
                .run(context -> {
                    AiServiceException failure = invokeExpectingFailure(
                            context.getBean(AiServiceClient.class), Kind.NETWORK_UNREACHABLE);
                    assertThat(failure.getHttpStatus()).isNull();
                });
    }

    @Test
    void sendsExactlyOnePostOnSuccess() {
        server.createContext(INVOKE_PATH, exchange -> {
            try (exchange) {
                countPostAndConsumeRequest(exchange);
                byte[] body = """
                        {"requestId":"network-test","data":{"answer":"示例回答","citations":[]}}
                        """.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, body.length);
                exchange.getResponseBody().write(body);
            }
        });
        server.start();

        configuredRunner().run(context -> {
            AiServiceClient client = context.getBean(AiServiceClient.class);
            assertThat(client.invokeLegalQa(INPUT, CONTEXT).data().answer()).isEqualTo("示例回答");
            assertThat(posts.get()).isEqualTo(1);
        });
    }

    @Test
    void timesOutWaitingForHeadersWithoutResendingPost() {
        // 用闭锁阻止处理器返回响应头，待客户端超时后在 finally 中释放服务端线程。
        CountDownLatch release = new CountDownLatch(1);
        server.createContext(INVOKE_PATH, exchange -> {
            try (exchange) {
                countPostAndConsumeRequest(exchange);
                awaitRelease(release);
            }
        });
        server.start();

        try {
            configuredRunner().run(context -> {
                AiServiceException failure = invokeExpectingFailure(context.getBean(AiServiceClient.class), Kind.TIMEOUT);
                assertThat(failure.getHttpStatus()).isNull();
                assertThat(posts.get()).isEqualTo(1);
            });
        } finally {
            release.countDown();
        }
    }

    @Test
    void bodyReadDeadlineUsesIoFallbackWithoutResendingPost() {
        // 先发送响应头和部分正文，再阻塞处理器，覆盖读取正文阶段的超时。
        CountDownLatch release = new CountDownLatch(1);
        server.createContext(INVOKE_PATH, exchange -> {
            try (exchange) {
                countPostAndConsumeRequest(exchange);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, 0);
                exchange.getResponseBody().write('{');
                exchange.getResponseBody().flush();
                awaitRelease(release);
            }
        });
        server.start();

        try {
            configuredRunner().run(context -> {
                // 当前 Spring/JDK 在正文超时后只报告流关闭，按证据使用 I/O 后备分类。
                AiServiceException failure = invokeExpectingFailure(context.getBean(AiServiceClient.class), Kind.NETWORK_IO_ERROR);
                assertThat(failure.getHttpStatus()).isEqualTo(200);
                assertThat(failure.getIoExceptionType()).isEqualTo(IOException.class.getName());
                assertThat(posts.get()).isEqualTo(1);
            });
        } finally {
            release.countDown();
        }
    }

    @Test
    void doesNotResendPostAfterConnectionClosesWithoutResponse() {
        server.createContext(INVOKE_PATH, exchange -> {
            try (exchange) {
                countPostAndConsumeRequest(exchange);
            }
        });
        server.start();

        configuredRunner().run(context -> {
            AiServiceException failure = invokeExpectingFailure(context.getBean(AiServiceClient.class), Kind.NETWORK_IO_ERROR);
            assertThat(failure.getHttpStatus()).isNull();
            assertThat(posts.get()).isEqualTo(1);
        });
    }

    @ParameterizedTest
    @ValueSource(ints = {200, 500})
    void interruptedBodyPreservesStatusWithoutResendingPost(int status) {
        server.createContext(INVOKE_PATH, exchange -> {
            try (exchange) {
                countPostAndConsumeRequest(exchange);
                // 声明更长的正文后提前关闭连接，制造真实的传输中断。
                exchange.sendResponseHeaders(status, 100);
                exchange.getResponseBody().write('{');
                exchange.getResponseBody().flush();
            }
        });
        server.start();

        configuredRunner().run(context -> {
            AiServiceException failure = invokeExpectingFailure(context.getBean(AiServiceClient.class), Kind.NETWORK_IO_ERROR);
            assertThat(failure.getHttpStatus()).isEqualTo(status);
            assertThat(failure.getIoExceptionType()).isEqualTo(IOException.class.getName());
            assertThat(posts.get()).isEqualTo(1);
        });
    }

    @Test
    void doesNotFollowPostRedirect() {
        server.createContext(INVOKE_PATH, exchange -> {
            try (exchange) {
                countPostAndConsumeRequest(exchange);
                exchange.getResponseHeaders().set("Location", "/redirected");
                exchange.sendResponseHeaders(307, -1);
            }
        });
        server.createContext("/redirected", exchange -> {
            try (exchange) {
                countPostAndConsumeRequest(exchange);
                exchange.sendResponseHeaders(204, -1);
            }
        });
        server.start();

        configuredRunner().run(context -> {
            AiServiceException failure = invokeExpectingFailure(
                    context.getBean(AiServiceClient.class), Kind.UNEXPECTED_HTTP_RESPONSE);
            assertThat(failure.getHttpStatus()).isEqualTo(307);
            assertThat(posts.get()).isEqualTo(1);
        });
    }

    @ParameterizedTest
    @ValueSource(strings = {"connect-timeout", "read-timeout"})
    void rejectsNonPositiveTimeouts(String property) {
        configuredRunner().withPropertyValues("ai-service." + property + "=0s")
                .run(context -> assertThat(context.getStartupFailure())
                        .hasStackTraceContaining("ai-service." + property + " must be a positive duration"));
    }

    private ApplicationContextRunner configuredRunner() {
        return contextRunner.withPropertyValues("ai-service.base-url=" + baseUrl);
    }

    private void countPostAndConsumeRequest(HttpExchange exchange) throws IOException {
        if ("POST".equals(exchange.getRequestMethod())) {
            posts.incrementAndGet();
        }
        exchange.getRequestBody().transferTo(OutputStream.nullOutputStream());
    }

    private static void awaitRelease(CountDownLatch release) {
        try {
            release.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private static AiServiceException invokeExpectingFailure(AiServiceClient client, Kind kind) {
        AiServiceException failure = catchThrowableOfType(
                AiServiceException.class, () -> client.invokeLegalQa(INPUT, CONTEXT));

        assertThat(failure).isNotNull();
        assertThat(failure.getKind()).isEqualTo(kind);
        assertThat(failure.getRequestId()).isEqualTo(REQUEST_ID);
        return failure;
    }
}
