from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import ResearchState
from config import AgentModels, DEFAULT_MODELS


SYSTEM_PROMPT = """You are a research supervisor. Your job is to:
1. Validate and refine the user's research query to make it more precise and searchable
2. Return the refined query as a single sentence, with no additional commentary

Always preserve the original language of the query — never translate it.
If the query is already clear and specific, return it unchanged."""


def supervisor_node(state: ResearchState, models: AgentModels = DEFAULT_MODELS) -> dict:
    llm = ChatOpenAI(model=models.supervisor, temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    refined_query = response.content.strip()

    return {"query": refined_query}
