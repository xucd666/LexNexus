from langgraph.graph import END, START, StateGraph

from app.agent.legal_qa.nodes import LegalQaNodes
from app.agent.legal_qa.state import LegalQaState


class LegalQaGraph:
    def __init__(self, nodes: LegalQaNodes) -> None:
        graph = StateGraph(LegalQaState)
        graph.add_node("retrieve_law", nodes.retrieve_law)
        graph.add_node("validate_citations", nodes.validate_citations)
        graph.add_node("generate_answer", nodes.generate_answer)
        graph.add_edge(START, "retrieve_law")
        graph.add_edge("retrieve_law", "validate_citations")
        graph.add_edge("validate_citations", "generate_answer")
        graph.add_edge("generate_answer", END)
        self.workflow = graph.compile()
