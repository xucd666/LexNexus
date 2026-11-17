from langchain_core.language_models import BaseChatModel

from app.agent.legal_qa.prompts import LEGAL_QA_PROMPT
from app.agent.legal_qa.state import LegalQaState
from app.llm.deepseek import DeepSeekInvocationError, invoke_chat
from app.tools.law import LawSearch


class LegalQaNodes:
    def __init__(self, law_tool: LawSearch, llm: BaseChatModel | None = None) -> None:
        self._law_tool = law_tool
        self._llm = llm

    async def retrieve_law(self, state: LegalQaState) -> LegalQaState:
        context = state["context"]
        return {
            "sources": await self._law_tool.search(
                state["question"], request_id=context.request_id, trace_id=context.trace_id
            )
        }

    async def generate_answer(self, state: LegalQaState) -> LegalQaState:
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
        # 只有具备来源标识和引用文本的材料才能进入提示词和结构化引用。
        sources = [
            source
            for source in state.get("sources", [])
            if source.document_id
            and source.document_id.strip()
            and source.kb_chunk_id
            and source.kb_chunk_id.strip()
            and source.quoted_text
            and source.quoted_text.strip()
        ]
        return {"sources": sources}
