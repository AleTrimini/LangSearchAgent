from functools import partial
from langgraph.graph import StateGraph, END
from state import ResearchState
from agents.supervisor import supervisor_node
from agents.researcher import researcher_node
from agents.summarizer import summarizer_node
from agents.writer import writer_node
from config import AgentModels, DEFAULT_MODELS


def build_graph(models: AgentModels = DEFAULT_MODELS) -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", partial(supervisor_node, models=models))
    graph.add_node("researcher", partial(researcher_node, models=models))
    graph.add_node("summarizer", partial(summarizer_node, models=models))
    graph.add_node("writer", partial(writer_node, models=models))

    graph.set_entry_point("supervisor")

    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "summarizer")
    graph.add_edge("summarizer", "writer")
    graph.add_edge("writer", END)

    return graph.compile()
