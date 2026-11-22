import json
import logging
from collections import Counter

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from app.agent.legal_qa.prompts import LEGAL_EVIDENCE_PROMPT, LEGAL_QA_PROMPT
from app.agent.legal_qa.state import EvidenceStatus, LegalQaState
from app.llm.deepseek import DeepSeekInvocationError, invoke_chat
from app.schemas.contracts import Citation
from app.tools.citation_validator import CitationValidator
from app.tools.law import LawSearch

logger = logging.getLogger(__name__)


class _EvidenceDecision(BaseModel):
    """依据判断模型的内部输出协议。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: EvidenceStatus
    citation_ids: list[str]


def _unique_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    # 默认 JSON 解析会保留同名字段的最后一个值，避免这种歧义放行引用。
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("依据判断含有重复字段")
    return fields


class LegalQaNodes:
    def __init__(
        self,
        law_tool: LawSearch,
        llm: BaseChatModel | None = None,
        *,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self._law_tool = law_tool
        self._llm = llm
        self._citation_validator = citation_validator

    async def retrieve_law(self, state: LegalQaState) -> LegalQaState:
        context = state["context"]
        return {
            "sources": await self._law_tool.search(
                state["question"], request_id=context.request_id, trace_id=context.trace_id
            )
        }

    async def generate_answer(self, state: LegalQaState) -> LegalQaState:
        if state.get("evidence_status") == "insufficient":
            return {"answer": "当前核验通过的法条不足以直接回答该问题。"}
        if state.get("evidence_status") == "uncertain":
            return {"answer": "无法可靠判断现有法条是否足以回答该问题。"}
        sources = state.get("sources", [])
        if not sources:
            return {"answer": "未找到直接法律依据，无法给出确定性回答。"}
        if self._llm is None:
            return {"answer": "离线测试：已检索到测试材料，调用链路运行成功。"}
        messages = LEGAL_QA_PROMPT.format_messages(question=state["question"], sources=sources)
        response = await invoke_chat(self._llm, messages)
        answer = response.text.strip()
        if not answer:
            raise DeepSeekInvocationError("DeepSeek 未返回有效回答")
        return {"answer": answer}

    async def validate_citations(self, state: LegalQaState) -> LegalQaState:
        sources = state.get("sources", [])
        if self._citation_validator is None:
            # 兼容 Fake 和没有配置参考校验的自定义检索器。
            return {
                "sources": [
                    source
                    for source in sources
                    if source.document_id
                    and source.document_id.strip()
                    and source.kb_chunk_id
                    and source.kb_chunk_id.strip()
                    and source.quoted_text
                    and source.quoted_text.strip()
                ]
            }

        # 更新图状态中的 sources，使提示词和 API 响应共享同一批通过校验的引用。
        accepted: list[Citation] = []
        rejected: Counter[str] = Counter()
        for source in sources:
            issue = self._citation_validator.validate(source)
            if issue is None:
                accepted.append(source)
            else:
                rejected[issue.value] += 1
        if rejected:
            logger.warning("法律引用校验拒绝 %d 条，原因分布 %s", rejected.total(), dict(rejected))
        return {"sources": accepted}

    async def assess_evidence(self, state: LegalQaState) -> LegalQaState:
        sources = state.get("sources", [])
        # 在线装配只有 live 模式提供模型，只有向量检索路径提供独立引用校验器。
        # 无候选、离线模式和沿用原行为的自定义检索器都跳过判断。
        if not sources or self._llm is None or self._citation_validator is None:
            return {}

        # 相似度只用于召回排序，充分性判断只看问题和核验通过的候选原文及标识。
        payload = json.dumps(
            {
                "question": state["question"],
                "candidates": [source.model_dump(exclude={"relevance_score"}) for source in sources],
            },
            ensure_ascii=False,
        )
        messages = LEGAL_EVIDENCE_PROMPT.format_messages(payload=payload)
        # 请求失败沿用现有模型异常处理。只有返回内容格式不合法才降为 uncertain。
        response = await invoke_chat(self._llm, messages)
        try:
            decision = _EvidenceDecision.model_validate(
                json.loads(response.text, object_pairs_hook=_unique_json_fields)
            )
            selected = set(decision.citation_ids)
            if decision.status == "supported" and not selected:
                raise ValueError("supported 缺少引用 ID")
            if decision.status != "supported" and selected:
                raise ValueError("未确认充分性时不得选取引用")
            if len(selected) != len(decision.citation_ids) or not selected.issubset(
                {source.kb_chunk_id for source in sources}
            ):
                raise ValueError("判断结果含有未知或重复的引用 ID")
        except ValueError, RecursionError:
            logger.warning("依据判断结果无效，按无法可靠判断处理")
            return {"evidence_status": "uncertain", "sources": []}

        # 只保留由模型选中的原始 Citation，保持检索顺序，不采纳模型生成的正文或字段。
        return {
            "evidence_status": decision.status,
            "sources": [source for source in sources if source.kb_chunk_id in selected],
        }
