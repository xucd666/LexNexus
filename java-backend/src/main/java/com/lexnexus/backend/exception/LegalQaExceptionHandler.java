package com.lexnexus.backend.exception;

import com.lexnexus.backend.controller.LegalQaController;
import com.lexnexus.backend.vo.LegalQaErrorVO;
import org.springframework.context.annotation.Profile;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/** 将本地法律问答的输入错误和上游异常转换为固定的安全响应。 */
@Profile("local")
@RestControllerAdvice(assignableTypes = LegalQaController.class)
public class LegalQaExceptionHandler {

    @ExceptionHandler({MethodArgumentNotValidException.class, HttpMessageNotReadableException.class})
    ResponseEntity<LegalQaErrorVO> invalidInput() {
        return error(HttpStatus.BAD_REQUEST, "QA_INVALID_INPUT",
                "请提交有效的 JSON，question 须为 1 至 4000 个字符的非空白字符串。", null);
    }

    @ExceptionHandler(AiServiceException.class)
    ResponseEntity<LegalQaErrorVO> upstreamFailure(AiServiceException failure) {
        // 异常携带的 requestId 由 Java 生成，上游错误内容只用于内部诊断。
        String requestId = failure.getRequestId();
        return switch (failure.getKind()) {
            case LLM_UNAVAILABLE -> error(HttpStatus.SERVICE_UNAVAILABLE, "QA_MODEL_UNAVAILABLE",
                    "问答模型暂不可用。", requestId);
            case NETWORK_UNREACHABLE -> error(HttpStatus.SERVICE_UNAVAILABLE, "QA_SERVICE_UNAVAILABLE",
                    "问答服务暂不可达。", requestId);
            case TIMEOUT -> error(HttpStatus.GATEWAY_TIMEOUT, "QA_UPSTREAM_TIMEOUT",
                    "等待问答服务响应超时。", requestId);
            case INVALID_REQUEST, INVALID_AGENT_INPUT, AGENT_NOT_FOUND ->
                    error(HttpStatus.BAD_GATEWAY, "QA_UPSTREAM_CONTRACT_ERROR",
                            "问答服务未接受内部请求。", requestId);
            case NETWORK_IO_ERROR, INVALID_RESPONSE, RESPONSE_TOO_LARGE, UNEXPECTED_HTTP_RESPONSE ->
                    error(HttpStatus.BAD_GATEWAY, "QA_UPSTREAM_FAILURE",
                            "未能从问答服务获取有效结果。", requestId);
        };
    }

    private static ResponseEntity<LegalQaErrorVO> error(
            HttpStatus status, String code, String message, String requestId) {
        return ResponseEntity.status(status).body(new LegalQaErrorVO(code, message, requestId));
    }
}
