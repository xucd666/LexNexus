package com.lexnexus.backend.controller;

import com.lexnexus.backend.dto.LegalQaRequestDTO;
import com.lexnexus.backend.service.LegalQaService;
import com.lexnexus.backend.vo.LegalQaResponseVO;
import jakarta.validation.Valid;
import org.springframework.context.annotation.Profile;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** 仅在 local Profile 下接收开发者的法律问答请求。 */
@Profile("local")
@RestController
@RequestMapping("/api/v1/dev/legal-qa")
public class LegalQaController {

    private final LegalQaService service;

    public LegalQaController(LegalQaService service) {
        this.service = service;
    }

    /**
     * 校验请求正文并同步返回法律问答结果。
     *
     * @param request 包含已规范化的问题
     * @return 问答结果
     */
    @PostMapping
    public LegalQaResponseVO answer(@Valid @RequestBody LegalQaRequestDTO request) {
        return service.answer(request.question());
    }
}
