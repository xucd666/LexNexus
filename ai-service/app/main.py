from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.agent.exceptions import AgentInputError, AgentNotFoundError
from app.agent.legal_qa.agent import LegalQaAgent
from app.agent.registry import AgentRegistry
from app.agent.runtime import AgentRuntime
from app.api.exception_handlers import (
    agent_input_handler,
    agent_not_found_handler,
    deepseek_invocation_handler,
    request_validation_handler,
)
from app.api.v1.agent import router
from app.core.config import Settings
from app.llm.deepseek import DeepSeekInvocationError, create_chat_model
from app.tools.law import FakeLawTool, LawSearch


def create_app(settings: Settings | None = None, *, law_tool: LawSearch | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        config = settings if settings is not None else Settings()
        if config.mode == "live":
            if law_tool is None or isinstance(law_tool, FakeLawTool):
                raise RuntimeError("live 模式需要注入真实法律检索实现")
            retriever = law_tool
            llm = create_chat_model(config)
        else:
            retriever = law_tool if law_tool is not None else FakeLawTool()
            llm = None

        registry = AgentRegistry()
        registry.register("legal_qa", LegalQaAgent(retriever, llm))
        application.state.agent_runtime = AgentRuntime(registry)
        yield

    application = FastAPI(title="LexNexus AI Service", lifespan=lifespan)
    application.add_exception_handler(AgentNotFoundError, agent_not_found_handler)
    application.add_exception_handler(AgentInputError, agent_input_handler)
    application.add_exception_handler(RequestValidationError, request_validation_handler)
    application.add_exception_handler(DeepSeekInvocationError, deepseek_invocation_handler)
    application.include_router(router)
    return application


app = create_app()
