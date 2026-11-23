package com.lexnexus.backend.controller;

import com.lexnexus.backend.client.aiservice.AiServiceClient;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** 验证默认 Profile 的 Bean 和 HTTP 路由均不注册法律问答开发接口。 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("default")
class LegalQaDefaultProfileTest {

    @Autowired
    private MockMvc mvc;

    @Autowired
    private ApplicationContext context;

    @MockitoBean
    private AiServiceClient client;

    @Test
    void hidesLocalEndpointOnDefaultProfile() throws Exception {
        assertThat(context.getBeansOfType(LegalQaController.class)).isEmpty();
        mvc.perform(post("/api/v1/dev/legal-qa").contentType(APPLICATION_JSON)
                        .content("{\"question\":\"示例问题\"}"))
                .andExpect(status().isNotFound());
        verifyNoInteractions(client);
    }
}
