package com.lexnexus.backend.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.lexnexus.backend.client.aiservice.AiServiceClient;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;

/** 注册 AI-Service 配置属性，并为跨服务调用装配独立的 JDK HTTP 客户端。 */
@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties(AiServiceProperties.class)
public class AiServiceClientConfiguration {

    /** 复用 Boot 的消息转换配置，并将两类超时实际应用到对应网络组件。 */
    @Bean
    AiServiceClient aiServiceClient(RestClient.Builder builder, ObjectMapper objectMapper, AiServiceProperties properties) {
        HttpClient httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .followRedirects(HttpClient.Redirect.NEVER)
                .connectTimeout(properties.connectTimeout())
                .build();

        // JDK 客户端控制连接建立，Spring 请求工厂控制等待和读取响应的时间。
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(properties.readTimeout());

        RestClient restClient = builder.baseUrl(properties.baseUrl().toString())
                .requestFactory(requestFactory)
                .build();
        return new AiServiceClient(restClient, objectMapper);
    }
}
