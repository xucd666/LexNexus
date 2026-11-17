import unittest
from unittest.mock import AsyncMock, patch

from app.agent.core.base_agent import BaseAgent
from app.agent.exceptions import AgentNotFoundError
from app.agent.registry import AgentRegistry
from app.agent.runtime import AgentRuntime
from app.schemas.contracts import AgentContext, LegalQaResult


class StubAgent(BaseAgent):
    async def invoke(self, input, context: AgentContext) -> LegalQaResult:
        return LegalQaResult(answer=f"{context.request_id}: {input['question']}")


class AgentRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = AgentRegistry()
        self.runtime = AgentRuntime(self.registry)

    def test_registry_requires_explicit_registration(self):
        self.assertIsNone(self.registry.get("stub"))
        agent = StubAgent()
        self.registry.register("stub", agent)
        self.assertIs(self.registry.get("stub"), agent)
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.registry.register("stub", StubAgent())
        self.assertIs(self.registry.get("stub"), agent)

    async def test_invokes_agent_for_requested_code(self):
        self.registry.register("first", StubAgent())
        selected = StubAgent()
        self.registry.register("selected", selected)

        context = AgentContext(requestId="runtime-test")
        payload = {"question": "runtime test"}
        with patch.object(selected, "invoke", AsyncMock(wraps=selected.invoke)) as invoke:
            result = await self.runtime.invoke("selected", payload, context)

        invoke.assert_awaited_once_with(payload, context)
        self.assertEqual(result, LegalQaResult(answer="runtime-test: runtime test"))

    async def test_raises_domain_error_for_unknown_agent(self):
        with self.assertRaisesRegex(AgentNotFoundError, "Agent not found: missing"):
            await self.runtime.invoke("missing", {}, AgentContext(requestId="missing-agent"))
