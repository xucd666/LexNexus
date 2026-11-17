from abc import ABC, abstractmethod

from pydantic import BaseModel, JsonValue

from app.schemas.contracts import AgentContext


class BaseAgent(ABC):
    """各 Agent 校验自己的业务输入并返回 Pydantic 模型。"""

    @abstractmethod
    async def invoke(self, input: dict[str, JsonValue], context: AgentContext) -> BaseModel:
        raise NotImplementedError
