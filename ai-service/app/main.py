from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

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
from app.tools.bge_embedding import BgeEmbedding
from app.tools.citation_validator import CitationReferenceError, CitationValidator
from app.tools.law import FakeLawTool, LawSearch
from app.tools.milvus_store import MilvusStore
from app.tools.vector_law_search import VectorLawSearch


def _load_citation_validator(config: Settings) -> CitationValidator:
    source_path = config.citation_reference_source_path.strip()
    metadata_path = config.citation_reference_metadata_path.strip()
    missing = [
        name
        for name, value in (
            ("AI_SERVICE_CITATION_REFERENCE_SOURCE_PATH", source_path),
            ("AI_SERVICE_CITATION_REFERENCE_METADATA_PATH", metadata_path),
        )
        if not value
    ]
    if missing:
        raise CitationReferenceError(f"向量检索缺少可信引用快照配置：{', '.join(missing)}")
    return CitationValidator.from_snapshot(Path(source_path), Path(metadata_path))


@asynccontextmanager
async def _lifespan_retriever(
    config: Settings, law_tool: LawSearch | None
) -> AsyncIterator[tuple[LawSearch, CitationValidator | None]]:
    """按注入优先级装配检索器及其引用校验器，只关闭本服务创建的存储。"""
    if law_tool is not None:
        # 显式注入的向量检索器也需要可信快照，资源仍归调用方管理。
        validator = _load_citation_validator(config) if isinstance(law_tool, VectorLawSearch) else None
        yield law_tool, validator
        return
    if config.law_search_backend == "fake":
        yield FakeLawTool(), None
        return

    model_path = config.bge_model_path.strip()
    db_path = config.milvus_db_path.strip()
    collection = config.milvus_collection_name.strip()
    missing = [
        name
        for name, value in (
            ("AI_SERVICE_BGE_MODEL_PATH", model_path),
            ("AI_SERVICE_MILVUS_DB_PATH", db_path),
            ("AI_SERVICE_MILVUS_COLLECTION_NAME", collection),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"vector 法律检索缺少配置：{', '.join(missing)}")

    # 先核对独立快照，失败时无需加载模型或持有数据库锁。
    validator = _load_citation_validator(config)

    # Lite 3.x 打开空目录也会初始化库，先确认已有入库脚本创建的 collections 目录。
    if not (Path(db_path).expanduser() / "collections").is_dir():
        raise FileNotFoundError(f"未找到已初始化的 Milvus Lite 数据库目录：{db_path}")

    embedding = BgeEmbedding(model_path)
    store = MilvusStore(db_path, collection, create=False)
    retriever = None
    try:
        retriever = VectorLawSearch(embedding, store, top_k=config.law_search_top_k)
        yield retriever, validator
    finally:
        # 请求取消后线程可能仍在执行，等待检索器停止提交并排空在途工作，再关闭存储。
        if retriever is not None:
            await retriever.stop()
        store.close()


def create_app(settings: Settings | None = None, *, law_tool: LawSearch | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        config = settings if settings is not None else Settings()
        async with _lifespan_retriever(config, law_tool) as (retriever, validator):
            if config.mode == "live" and isinstance(retriever, FakeLawTool):
                raise RuntimeError("live 模式需要注入真实法律检索实现，或在未显式注入时启用 vector 后端")
            llm = create_chat_model(config) if config.mode == "live" else None

            registry = AgentRegistry()
            registry.register("legal_qa", LegalQaAgent(retriever, llm, citation_validator=validator))
            application.state.agent_runtime = AgentRuntime(registry)
            try:
                yield
            finally:
                del application.state.agent_runtime

    application = FastAPI(title="LexNexus AI Service", lifespan=lifespan)
    application.add_exception_handler(AgentNotFoundError, agent_not_found_handler)
    application.add_exception_handler(AgentInputError, agent_input_handler)
    application.add_exception_handler(RequestValidationError, request_validation_handler)
    application.add_exception_handler(DeepSeekInvocationError, deepseek_invocation_handler)
    application.include_router(router)
    return application


app = create_app()
