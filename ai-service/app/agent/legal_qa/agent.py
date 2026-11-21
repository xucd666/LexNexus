from langchain_core.language_models import BaseChatModel
from pydantic import JsonValue, ValidationError

from app.agent.core.base_agent import BaseAgent
from app.agent.exceptions import AgentInputError
from app.agent.legal_qa.graph import LegalQaGraph
from app.agent.legal_qa.nodes import LegalQaNodes
from app.schemas.contracts import AgentContext, LegalQaInput, LegalQaResult
from app.tools.citation_validator import CitationValidator
from app.tools.law import LawSearch


class LegalQaAgent(BaseAgent):
    def __init__(
        self,
        law_tool: LawSearch,
        llm: BaseChatModel | None = None,
        *,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self._graph = LegalQaGraph(
            LegalQaNodes(law_tool, llm, citation_validator=citation_validator)
        ).workflow

    async def invoke(
        self, input: LegalQaInput | dict[str, JsonValue], context: AgentContext
    ) -> LegalQaResult:
        try:
            payload = LegalQaInput.model_validate(input)
        except ValidationError as exc:
            raise AgentInputError(exc) from exc
        state = await self._graph.ainvoke({"question": payload.question, "context": context})
        return LegalQaResult(answer=state["answer"], citations=state.get("sources", []))
