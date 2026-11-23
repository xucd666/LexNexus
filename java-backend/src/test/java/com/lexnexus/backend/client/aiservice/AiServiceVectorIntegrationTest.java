package com.lexnexus.backend.client.aiservice;

import com.lexnexus.backend.config.AiServiceProperties;
import com.lexnexus.backend.dto.aiservice.AgentContext;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse.Citation;
import com.lexnexus.backend.dto.aiservice.LegalQaInput;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 通过真实 HTTP 验证 Java 对八条《民法典》样本的向量检索结果及请求标识解码。
 * 仅在显式设置 lexnexus.ai-service.vector-integration=true 时运行，目标须是本机
 * fake + vector AI-Service。服务端向量化与引用校验的执行证据由联调时独立采集。
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE, properties = {
        "ai-service.base-url=${lexnexus.ai-service.url:http://127.0.0.1:18080}"
})
@EnabledIfSystemProperty(named = "lexnexus.ai-service.vector-integration", matches = "true")
class AiServiceVectorIntegrationTest {

    private static final String DOCUMENT_ID = "cn-civil-code-2020-05-28";
    private static final String LAW_NAME = "中华人民共和国民法典";
    private static final String OFFLINE_ANSWER = "离线测试：已检索到测试材料，调用链路运行成功。";

    @Autowired
    private AiServiceClient client;

    @Autowired
    private AiServiceProperties properties;

    @BeforeEach
    void requireLoopbackTarget() {
        assertThat(properties.baseUrl().getScheme()).isEqualTo("http");
        assertThat(properties.baseUrl().getHost()).isEqualTo("127.0.0.1");
    }

    @ParameterizedTest(name = "民法典第 {0} 条")
    @CsvSource(delimiter = '|', textBlock = """
            577 | 第五百七十七条 | 供应商到期没有交货，守约方一般可以主张哪些违约救济？
            578 | 第五百七十八条 | 距离交货还有一个月，卖家已明确说绝不会交货，必须等期限届满才能追究责任吗？
            579 | 第五百七十九条 | 房客连着三个月没交房租，房东能要求补交吗？
            580 | 第五百八十条 | 完成约定的非金钱义务需要付出过高成本，对方还能要求强制履行吗？
            581 | 第五百八十一条 | 约定工作不能强制原承包人履行，可以让违约方承担请第三方做完的费用吗？
            582 | 第五百八十二条 | 买来的机器质量没达到约定，合同也没约定违约责任，可以要求修理或换货吗？
            583 | 第五百八十三条 | 对方修好了交付设备的缺陷，但维修期间还有其他损失，能另外索赔吗？
            584 | 第五百八十四条 | 对方违约让我失去合同正常履行后本来能赚到的利润，赔偿可以包括这部分吗？
            """)
    void returnsExpectedArticleFromVectorSearch(int articleNumber, String articleNo, String question)
            throws NoSuchAlgorithmException {
        List<Citation> citations = invokeAndCheck("j20-article-" + articleNumber, question, OFFLINE_ANSWER);
        assertThat(citations).extracting(Citation::articleNo).contains(articleNo);
    }

    @Test
    void observesCitationsForQuestionOutsideSnapshot() throws NoSuchAlgorithmException {
        // 离线模式跳过模型的依据充分性判断，范围外问题仍可能召回可信但不相关的条文。
        invokeAndCheck("j20-outside", "劳动合同到期公司不续签，经济补偿按多少个月的工资计算？", null);
    }

    private List<Citation> invokeAndCheck(String requestId, String question, String expectedAnswer)
            throws NoSuchAlgorithmException {
        var response = client.invokeLegalQa(new LegalQaInput(question),
                new AgentContext(requestId, requestId, null, null, null, null));
        assertThat(response.requestId()).isEqualTo(requestId);
        var data = response.data();
        assertThat(data.answer()).isNotBlank();
        if (expectedAnswer != null) {
            assertThat(data.answer()).isEqualTo(expectedAnswer);
        }
        System.out.printf("J20_RESPONSE\t%s\t%d\t%s%n", response.requestId(), data.citations().size(), data.answer());
        for (Citation citation : data.citations()) {
            assertThat(citation.documentId()).isEqualTo(DOCUMENT_ID);
            assertThat(citation.kbChunkId()).matches("[a-f0-9]{32}");
            assertThat(citation.lawName()).isEqualTo(LAW_NAME);
            assertThat(citation.articleNo()).isNotBlank();
            assertThat(citation.quotedText()).isNotBlank();
            assertThat(citation.relevanceScore()).isNotNull().isFinite();
            // 使用正文摘要与 Python 的实际检索、校验结果逐条核对，保留空白和换行差异。
            byte[] quoteHash = MessageDigest.getInstance("SHA-256")
                    .digest(citation.quotedText().getBytes(StandardCharsets.UTF_8));
            System.out.printf("J20_CITATION\t%s\t%s\t%s\t%s\t%s\t%s\t%s%n", response.requestId(),
                    citation.documentId(), citation.kbChunkId(), citation.lawName(), citation.articleNo(),
                    HexFormat.of().formatHex(quoteHash), citation.relevanceScore());
        }
        return data.citations();
    }
}
