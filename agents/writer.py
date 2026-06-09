from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import ResearchState
from config import AgentModels, DEFAULT_MODELS


SYSTEM_PROMPT = """You are a professional report writer. Given a research summary, produce a well-structured,
engaging final report. The report should include:
- An executive summary (2-3 sentences)
- Key findings (bullet points)
- Detailed analysis (2-3 paragraphs)
- Conclusion

Write in a clear, professional tone suitable for a general audience.
Always respond in the same language as the original query."""


def writer_node(state: ResearchState, models: AgentModels = DEFAULT_MODELS) -> ResearchState:
    llm = ChatOpenAI(model=models.writer, temperature=0.3)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Topic: {state['query']}\n\nResearch summary to turn into a report:\n\n{state['summary']}"
        ),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "final_report": response.content,
        "messages": state["messages"] + [response],
        "next": "END",
    }
