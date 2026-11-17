package com.lexnexus.backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/** Java 后端启动入口，负责组件扫描和 Spring Boot 应用装配。 */
@SpringBootApplication
public class LexNexusApplication {

    /**
     * 启动应用并加载外部配置。
     *
     * @param args Spring Boot 命令行参数
     */
    public static void main(String[] args) {
        SpringApplication.run(LexNexusApplication.class, args);
    }
}
