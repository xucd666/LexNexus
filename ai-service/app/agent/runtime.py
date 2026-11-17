from pydantic import BaseModel, JsonValue

from app.agent.exceptions import AgentNotFoundError
from app.agent.registry import AgentRegistry
from app.schemas.contracts import AgentContext


class AgentRuntime:
    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    async def invoke(self, agent_code: str, input: dict[str, JsonValue], context: AgentContext) -> BaseModel:
        agent = self._registry.get(agent_code)
        if agent is None:
            raise AgentNotFoundError(agent_code)
        return await agent.invoke(input, context)
