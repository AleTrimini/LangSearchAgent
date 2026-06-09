from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import ResearchState
from tools.search import web_search
from config import AgentModels, DEFAULT_MODELS


SYSTEM_PROMPT = """You are a research agent. Your job is to search the web and gather relevant information about a topic.
Given a query, generate 2-3 targeted search queries and collect the results.
Always return the raw search results so they can be summarized later."""


def researcher_node(state: ResearchState, models: AgentModels = DEFAULT_MODELS) -> ResearchState:
    llm = ChatOpenAI(model=models.researcher, temperature=0)
    llm_with_tools = llm.bind_tools([web_search])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research this topic thoroughly: {state['query']}"),
    ]

    response = llm_with_tools.invoke(messages)
    search_results = []

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "web_search":
                result = web_search.invoke(tool_call["args"])
                search_results.append(result)

    if not search_results:
        fallback = web_search.invoke({"query": state["query"]})
        search_results.append(fallback)

    return {
        **state,
        "search_results": search_results,
        "messages": state["messages"] + [response],
        "next": "summarizer",
    }
