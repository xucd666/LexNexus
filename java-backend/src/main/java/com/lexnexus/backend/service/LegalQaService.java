package com.lexnexus.backend.service;

import com.lexnexus.backend.vo.LegalQaResponseVO;

/** 法律问答的业务入口，由调用方提供已校验的问题。 */
public interface LegalQaService {

    /**
     * 同步获取单次问答结果。
     *
     * @param question 已去除首尾空白且通过校验的法律问题
     * @return 回答、引用及本次请求标识
     */
    LegalQaResponseVO answer(String question);
}
