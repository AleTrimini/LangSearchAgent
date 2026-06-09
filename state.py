from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    search_results: List[str]
    summary: str
    final_report: str
