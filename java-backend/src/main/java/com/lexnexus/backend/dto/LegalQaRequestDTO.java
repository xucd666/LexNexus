package com.lexnexus.backend.dto;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import org.hibernate.validator.constraints.CodePointLength;

/**
 * 本地法律问答的 HTTP 输入。先去除首尾空白，再校验非空和长度。
 *
 * @param question 法律问题，最多 4000 个 Unicode 码点
 */
public record LegalQaRequestDTO(
        @NotBlank @CodePointLength(max = 4000) String question
) {
    /** 在 Bean Validation 检查之前规范化首尾的 Unicode 空白。 */
    public LegalQaRequestDTO {
        if (question != null) {
            question = stripEdgeWhitespace(question);
        }
    }

    @JsonCreator(mode = JsonCreator.Mode.PROPERTIES)
    static LegalQaRequestDTO fromJson(@JsonProperty("question") JsonNode question) {
        // 保留严格的字符串契约，避免 Jackson 把数字或布尔值转成合法问题。
        return new LegalQaRequestDTO(question != null && question.isTextual() ? question.textValue() : null);
    }

    private static String stripEdgeWhitespace(String value) {
        int start = 0;
        int end = value.length();
        // 两侧各到第一个非空白码点就停止，避免在长度校验前重复扫描中间的空白。
        while (start < end) {
            int codePoint = value.codePointAt(start);
            if (!isUnicodeWhitespace(codePoint)) {
                break;
            }
            start += Character.charCount(codePoint);
        }
        while (start < end) {
            int codePoint = value.codePointBefore(end);
            if (!isUnicodeWhitespace(codePoint)) {
                break;
            }
            end -= Character.charCount(codePoint);
        }
        return value.substring(start, end);
    }

    private static boolean isUnicodeWhitespace(int codePoint) {
        // 沿用旧正则 (?U)\s 的 White_Space 集合，包含不换行空格和 U+0085。
        return Character.isSpaceChar(codePoint)
                || (codePoint >= 0x09 && codePoint <= 0x0D)
                || codePoint == 0x85;
    }
}
