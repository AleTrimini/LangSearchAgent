from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from state import ResearchState
from config import AgentModels, DEFAULT_MODELS


SYSTEM_PROMPT = """You are a summarization agent. You receive raw web search results and produce a clean,
structured summary of the key findings. Focus on accuracy and relevance to the original query.
Organize the information into clear themes or sections.
Always respond in the same language as the original query."""


def summarizer_node(state: ResearchState, models: AgentModels = DEFAULT_MODELS) -> ResearchState:
    llm = ChatOpenAI(model=models.summarizer, temperature=0)

    results_text = "\n\n---\n\n".join(state["search_results"])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Original query: {state['query']}\n\nSearch results to summarize:\n\n{results_text}"
        ),
    ]

    response = llm.invoke(messages)

    return {
        "summary": response.content,
        "messages": [response],
    }
