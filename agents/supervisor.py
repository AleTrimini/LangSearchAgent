from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import ResearchState
from config import AgentModels, DEFAULT_MODELS


SYSTEM_PROMPT = """You are a research supervisor. Your job is to:
1. Validate and refine the user's research query to make it more precise and searchable
2. Return the refined query as a single sentence, with no additional commentary

Always preserve the original language of the query — never translate it.
If the query is already clear and specific, return it unchanged."""


def supervisor_node(state: ResearchState, models: AgentModels = DEFAULT_MODELS) -> ResearchState:
    llm = ChatOpenAI(model=models.supervisor, temperature=0)

    if not state.get("search_results"):
        next_agent = "researcher"
    elif not state.get("summary"):
        next_agent = "summarizer"
    elif not state.get("final_report"):
        next_agent = "writer"
    else:
        next_agent = "END"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    refined_query = response.content.strip()

    return {
        **state,
        "query": refined_query,
        "next": next_agent,
    }
