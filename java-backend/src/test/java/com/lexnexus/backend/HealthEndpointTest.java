package com.lexnexus.backend;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;

/** 启动随机端口上的真实 Web 服务，验证应用装配与健康检查契约。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class HealthEndpointTest {

    @Autowired
    private TestRestTemplate restTemplate;

    @Test
    void returnsHealthFromRunningServer() {
        ResponseEntity<JsonNode> response = restTemplate.getForEntity("/api/v1/health", JsonNode.class);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getHeaders().getContentType()).isEqualTo(MediaType.APPLICATION_JSON);

        JsonNode body = response.getBody();
        assertThat(body).isNotNull();
        assertThat(body.path("service").asText()).isEqualTo("lexnexus-java-backend");
        assertThat(body.path("status").asText()).isEqualTo("UP");
    }
}
