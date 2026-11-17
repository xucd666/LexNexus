"""Agent API 和 Legal QA 的数据契约。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class AgentContext(BaseModel):
    """跨 Agent 的最小调用上下文。"""

    model_config = ConfigDict(populate_by_name=True)
    request_id: Identifier = Field(alias="requestId")
    trace_id: Identifier | None = Field(default=None, alias="traceId")
    user_id: Identifier | None = Field(default=None, alias="userId")
    session_id: Identifier | None = Field(default=None, alias="sessionId")
    task_id: Identifier | None = Field(default=None, alias="taskId")
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class LegalQaInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    document_id: str | None = None
    kb_chunk_id: str | None = None
    law_name: str | None = None
    article_no: str | None = None
    quoted_text: str | None = None
    relevance_score: float | None = None


class LegalQaResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class AgentInvokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_code: Identifier = Field(alias="agentCode")
    input: dict[str, JsonValue]
    context: AgentContext


class AgentInvokeResponse(BaseModel):
    request_id: str = Field(alias="requestId")
    data: dict[str, JsonValue]


class ValidationErrorDetail(BaseModel):
    loc: list[str | int]
    type: str


class AgentValidationError(BaseModel):
    code: Literal["INVALID_REQUEST", "INVALID_AGENT_INPUT"]
    message: str
    details: list[ValidationErrorDetail]


class AgentValidationErrorResponse(BaseModel):
    success: Literal[False]
    error: AgentValidationError
