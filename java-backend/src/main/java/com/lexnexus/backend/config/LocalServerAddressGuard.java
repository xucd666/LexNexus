package com.lexnexus.backend.config;

import org.springframework.boot.web.server.WebServerFactoryCustomizer;
import org.springframework.boot.web.servlet.server.AbstractServletWebServerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

import java.net.InetAddress;

/** 在 local Profile 的嵌入式 Servlet 服务启动前保护无鉴权开发接口。 */
@Component
@Profile("local")
@Order(Ordered.LOWEST_PRECEDENCE)
public class LocalServerAddressGuard implements WebServerFactoryCustomizer<AbstractServletWebServerFactory> {

    /**
     * 检查 Spring Boot 装配到 WebServerFactory 的最终监听地址。
     *
     * @param factory 已应用 Spring Boot 服务器配置的工厂
     * @throws IllegalStateException 未指定有效回环监听地址时拒绝启动
     */
    @Override
    public void customize(AbstractServletWebServerFactory factory) {
        // 等 Boot 应用 server.address 的有效配置后再检查，包括环境变量和命令行覆盖。
        InetAddress address = factory.getAddress();
        if (address == null || !address.isLoopbackAddress()) {
            throw new IllegalStateException("local Profile 要求 server.address 指定有效的回环地址。");
        }
    }
}
