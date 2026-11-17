from pydantic import ValidationError


class AgentNotFoundError(Exception):
    """Raised when an agent code is not registered."""

    def __init__(self, agent_code: str) -> None:
        self.agent_code = agent_code
        super().__init__(f"Agent not found: {agent_code}")


class AgentInputError(Exception):
    """Agent 专属输入校验失败。"""

    def __init__(self, error: ValidationError) -> None:
        self.details = error.errors(include_input=False, include_context=False, include_url=False)
        super().__init__("Agent 输入无效")
