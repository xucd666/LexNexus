package com.lexnexus.backend.config;

import com.lexnexus.backend.LexNexusApplication;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.web.servlet.context.ServletWebServerApplicationContext;
import org.springframework.boot.web.servlet.server.AbstractServletWebServerFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 通过真实 Servlet 应用上下文验证 local 的默认监听地址和外部配置覆盖。 */
class LocalServerAddressGuardTest {

    @Test
    void startsOnConfiguredDefaultLoopbackAddress() {
        try (var context = localApplication().run("--server.port=0")) {
            var serverContext = (ServletWebServerApplicationContext) context;
            var factory = context.getBean(AbstractServletWebServerFactory.class);
            assertThat(factory.getAddress().getHostAddress()).isEqualTo("127.0.0.1");
            assertThat(serverContext.getWebServer().getPort()).isPositive();
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"0.0.0.0", "::", "192.0.2.1", ""})
    void refusesExternalOverrideBeforeStartingTheServer(String address) {
        // 命令行配置优先于 application-local.properties，验证守卫检查的是实际生效的地址。
        assertThatThrownBy(() -> {
            try (var context = localApplication().run("--server.port=0", "--server.address=" + address)) {
                // 若启动保护发生回归，仍应关闭本次异常启动的上下文。
            }
        }).hasRootCauseInstanceOf(IllegalStateException.class)
                .hasRootCauseMessage("local Profile 要求 server.address 指定有效的回环地址。");
    }

    private SpringApplicationBuilder localApplication() {
        return new SpringApplicationBuilder(LexNexusApplication.class)
                .profiles("local")
                .registerShutdownHook(false);
    }
}
