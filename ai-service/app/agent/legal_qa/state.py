from typing import Literal, TypedDict

from app.schemas.contracts import AgentContext, Citation

EvidenceStatus = Literal["supported", "insufficient", "uncertain"]


class LegalQaState(TypedDict, total=False):
    question: str
    context: AgentContext
    sources: list[Citation]
    evidence_status: EvidenceStatus
    answer: str
