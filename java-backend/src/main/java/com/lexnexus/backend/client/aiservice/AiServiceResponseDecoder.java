package com.lexnexus.backend.client.aiservice;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.node.MissingNode;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse.Citation;
import com.lexnexus.backend.dto.aiservice.AgentInvokeResponse.LegalQaResult;
import com.lexnexus.backend.dto.aiservice.AiServiceError;
import com.lexnexus.backend.exception.AiServiceException;
import com.lexnexus.backend.exception.AiServiceException.Kind;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * 客户端包内的响应解码器，负责严格检查 JSON 类型并转换为传输模型。
 * 使用树节点读取，避免将数字等错误类型自动转换为字符串后误判为有效响应。
 * 调用方先完成正文大小限制，本类仅解析已完整读取且未超限的字节数组。
 */
final class AiServiceResponseDecoder {

    private static final List<String> CITATION_TEXT_FIELDS = List.of(
            "document_id", "kb_chunk_id", "law_name", "article_no", "quoted_text");

    private final ObjectReader jsonReader;

    AiServiceResponseDecoder(ObjectMapper objectMapper) {
        // 独立 ObjectReader 拒绝尾随 JSON 和重复字段，避免修改共享 ObjectMapper。
        jsonReader = objectMapper.reader()
                .with(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
                .with(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY);
    }

    /** 校验成功体的必填字段和回显 ID，引用中的可空字段按 Python 契约保留。 */
    AgentInvokeResponse decodeSuccess(byte[] body, int status, String requestId) {
        JsonNode root = readJson(body);
        JsonNode data = root.path("data");
        JsonNode answer = data.path("answer");
        JsonNode citations = data.path("citations");
        if (!requestId.equals(root.path("requestId").textValue()) || !answer.isTextual() || !citations.isArray()) {
            throw new AiServiceException(Kind.INVALID_RESPONSE, status, requestId, null);
        }

        List<Citation> decoded = new ArrayList<>();
        for (JsonNode item : citations) {
            JsonNode score = item.path("relevance_score");
            boolean invalidText = CITATION_TEXT_FIELDS.stream().map(item::path)
                    .anyMatch(field -> !field.isMissingNode() && !field.isNull() && !field.isTextual());
            boolean invalidScore = !score.isMissingNode() && !score.isNull()
                    && (!score.isNumber() || !Double.isFinite(score.doubleValue()));
            if (!item.isObject() || invalidText || invalidScore) {
                throw new AiServiceException(Kind.INVALID_RESPONSE, status, requestId, null);
            }
            // Python 引用采用 snake_case，此处显式映射到 Java record 的 camelCase 成员。
            decoded.add(new Citation(
                    item.path("document_id").textValue(), item.path("kb_chunk_id").textValue(),
                    item.path("law_name").textValue(), item.path("article_no").textValue(),
                    item.path("quoted_text").textValue(), score.isNumber() ? score.doubleValue() : null));
        }

        return new AgentInvokeResponse(requestId, new LegalQaResult(answer.textValue(), List.copyOf(decoded)));
    }

    /** 识别结构化上游错误；框架错误页或非契约 JSON 使用统一的后备分类。 */
    AiServiceException decodeHttpError(byte[] body, int status, String requestId) {
        JsonNode root = readJson(body);
        JsonNode success = root.path("success");
        JsonNode error = root.path("error");
        String code = error.path("code").textValue();
        String message = error.path("message").textValue();
        if (!success.isBoolean() || success.booleanValue() || code == null || message == null) {
            return new AiServiceException(Kind.UNEXPECTED_HTTP_RESPONSE, status, requestId, null);
        }

        AiServiceError upstream = new AiServiceError(
                code, message, error.path("agentCode").textValue(), readDetails(error.path("details")));
        // 错误码和 HTTP 状态必须匹配，防止将异常组合错误归入已知业务语义。
        Kind kind = switch (code) {
            case "INVALID_AGENT_INPUT" -> status == 422 ? Kind.INVALID_AGENT_INPUT : Kind.UNEXPECTED_HTTP_RESPONSE;
            case "INVALID_REQUEST" -> status == 422 ? Kind.INVALID_REQUEST : Kind.UNEXPECTED_HTTP_RESPONSE;
            case "AGENT_NOT_FOUND" -> status == 404 ? Kind.AGENT_NOT_FOUND : Kind.UNEXPECTED_HTTP_RESPONSE;
            case "LLM_UNAVAILABLE" -> status == 503 ? Kind.LLM_UNAVAILABLE : Kind.UNEXPECTED_HTTP_RESPONSE;
            default -> Kind.UNEXPECTED_HTTP_RESPONSE;
        };
        return new AiServiceException(kind, status, requestId, upstream);
    }

    /** 只提取符合契约的校验详情，并保留 loc 中字段名称和数组索引的原始类型。 */
    private static List<AiServiceError.Detail> readDetails(JsonNode details) {
        if (!details.isArray()) {
            return List.of();
        }
        List<AiServiceError.Detail> decoded = new ArrayList<>();
        for (JsonNode detail : details) {
            JsonNode loc = detail.path("loc");
            String type = detail.path("type").textValue();
            if (loc.isArray() && type != null) {
                List<JsonNode> segments = new ArrayList<>();
                loc.forEach(segments::add);
                if (segments.stream().allMatch(segment -> segment.isTextual() || segment.isIntegralNumber())) {
                    decoded.add(new AiServiceError.Detail(List.copyOf(segments), type));
                }
            }
        }
        return List.copyOf(decoded);
    }

    private JsonNode readJson(byte[] body) {
        try {
            JsonNode root = jsonReader.readTree(body);
            return root == null ? MissingNode.getInstance() : root;
        } catch (IOException malformedJson) {
            // 由调用方根据 HTTP 状态分类；解析异常中的响应片段不进入对外异常消息。
            return MissingNode.getInstance();
        }
    }
}
