from app.agent.core.base_agent import BaseAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent_code: str, agent: BaseAgent) -> None:
        if not agent_code or agent_code != agent_code.strip():
            raise ValueError("Agent code must be non-empty and have no surrounding whitespace")
        if agent_code in self._agents:
            raise ValueError(f"Agent already registered: {agent_code}")
        self._agents[agent_code] = agent

    def get(self, agent_code: str) -> BaseAgent | None:
        return self._agents.get(agent_code)
