"""End-to-end tests for the research pipeline using mocked LLM and search tool."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock, call
from langchain_core.messages import AIMessage

import pytest
from state import ResearchState
from graph import build_graph
from agents.supervisor import supervisor_node
from agents.researcher import researcher_node
from agents.summarizer import summarizer_node
from agents.writer import writer_node


MOCK_SEARCH_RESULT = (
    "Title: AI in 2025\nURL: https://example.com\nSnippet: AI has advanced significantly."
)


def make_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)


def base_state(**kwargs) -> ResearchState:
    defaults: ResearchState = {
        "query": "Latest AI developments in 2025",
        "messages": [],
        "search_results": [],
        "summary": "",
        "final_report": "",
    }
    return {**defaults, **kwargs}


# --- Supervisor ---

class TestSupervisorNode:
    def test_refines_query_using_llm(self):
        refined = "Latest advances in artificial intelligence and machine learning 2025"
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message(refined)
            result = supervisor_node(base_state())
        assert result["query"] == refined

    def test_strips_whitespace_from_llm_response(self):
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("  refined query  \n")
            result = supervisor_node(base_state())
        assert result["query"] == "refined query"

    def test_sends_original_query_to_llm(self):
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("refined")
            supervisor_node(base_state(query="my original query"))
        invoke_args = mock_llm_cls.return_value.invoke.call_args[0][0]
        human_message = invoke_args[-1]
        assert "my original query" in human_message.content

    def test_uses_configured_model(self):
        from config import AgentModels
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("q")
            supervisor_node(base_state(), models=AgentModels(supervisor="gpt-4o-mini"))
        mock_llm_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0)


# --- Researcher ---

class TestResearcherNode:
    def test_uses_tool_calls_from_llm(self):
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"name": "web_search", "args": {"query": "AI 2025"}}
        ]
        mock_response.content = ""

        with patch("agents.researcher.ChatOpenAI") as mock_llm_cls, \
             patch("agents.researcher.web_search") as mock_search:
            mock_llm_cls.return_value.bind_tools.return_value.invoke.return_value = mock_response
            mock_search.invoke.return_value = MOCK_SEARCH_RESULT
            result = researcher_node(base_state())

        mock_search.invoke.assert_called_once_with({"query": "AI 2025"})
        assert result["search_results"] == [MOCK_SEARCH_RESULT]

    def test_falls_back_to_direct_search_when_no_tool_calls(self):
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = ""

        with patch("agents.researcher.ChatOpenAI") as mock_llm_cls, \
             patch("agents.researcher.web_search") as mock_search:
            mock_llm_cls.return_value.bind_tools.return_value.invoke.return_value = mock_response
            mock_search.invoke.return_value = MOCK_SEARCH_RESULT
            result = researcher_node(base_state(query="fallback query"))

        mock_search.invoke.assert_called_once_with({"query": "fallback query"})
        assert result["search_results"] == [MOCK_SEARCH_RESULT]

    def test_multiple_tool_calls_produce_multiple_results(self):
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"name": "web_search", "args": {"query": "query A"}},
            {"name": "web_search", "args": {"query": "query B"}},
        ]
        mock_response.content = ""

        with patch("agents.researcher.ChatOpenAI") as mock_llm_cls, \
             patch("agents.researcher.web_search") as mock_search:
            mock_llm_cls.return_value.bind_tools.return_value.invoke.return_value = mock_response
            mock_search.invoke.side_effect = ["result A", "result B"]
            result = researcher_node(base_state())

        assert result["search_results"] == ["result A", "result B"]

    def test_llm_response_added_to_messages(self):
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = ""

        with patch("agents.researcher.ChatOpenAI") as mock_llm_cls, \
             patch("agents.researcher.web_search") as mock_search:
            mock_llm_cls.return_value.bind_tools.return_value.invoke.return_value = mock_response
            mock_search.invoke.return_value = MOCK_SEARCH_RESULT
            result = researcher_node(base_state())

        assert mock_response in result["messages"]


# --- Summarizer ---

class TestSummarizerNode:
    def test_produces_summary_from_search_results(self):
        with patch("agents.summarizer.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("Key findings: AI is booming.")
            result = summarizer_node(base_state(search_results=[MOCK_SEARCH_RESULT]))

        assert result["summary"] == "Key findings: AI is booming."

    def test_includes_query_and_results_in_prompt(self):
        with patch("agents.summarizer.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("summary")
            summarizer_node(base_state(
                query="test query",
                search_results=["result one", "result two"],
            ))

        invoke_args = mock_llm_cls.return_value.invoke.call_args[0][0]
        human_message = invoke_args[-1]
        assert "test query" in human_message.content
        assert "result one" in human_message.content
        assert "result two" in human_message.content

    def test_multiple_results_are_joined(self):
        with patch("agents.summarizer.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("summary")
            summarizer_node(base_state(search_results=["r1", "r2", "r3"]))

        invoke_args = mock_llm_cls.return_value.invoke.call_args[0][0]
        assert "r1" in invoke_args[-1].content
        assert "r2" in invoke_args[-1].content
        assert "r3" in invoke_args[-1].content


# --- Writer ---

class TestWriterNode:
    def test_produces_final_report_from_summary(self):
        with patch("agents.writer.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("## Report\nDetailed findings.")
            result = writer_node(base_state(summary="A summary."))

        assert result["final_report"] == "## Report\nDetailed findings."

    def test_includes_query_and_summary_in_prompt(self):
        with patch("agents.writer.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("report")
            writer_node(base_state(query="my topic", summary="my summary"))

        invoke_args = mock_llm_cls.return_value.invoke.call_args[0][0]
        human_message = invoke_args[-1]
        assert "my topic" in human_message.content
        assert "my summary" in human_message.content

    def test_uses_nonzero_temperature_for_creativity(self):
        from config import AgentModels
        with patch("agents.writer.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("report")
            writer_node(base_state(summary="s"), models=AgentModels())
        _, kwargs = mock_llm_cls.call_args
        assert kwargs.get("temperature", 0) > 0


# --- Full graph (end-to-end) ---

class TestGraph:
    def test_full_pipeline_produces_report(self):
        supervisor_response = make_ai_message("Latest AI developments in 2025")
        summary_response = make_ai_message("Summary of AI findings.")
        report_response = make_ai_message("## Final Report\nAI is advancing rapidly.")

        researcher_response = AIMessage(content="", tool_calls=[])

        with patch("agents.supervisor.ChatOpenAI") as sup_llm, \
             patch("agents.researcher.ChatOpenAI") as res_llm, \
             patch("agents.researcher.web_search") as mock_search, \
             patch("agents.summarizer.ChatOpenAI") as sum_llm, \
             patch("agents.writer.ChatOpenAI") as wri_llm:

            sup_llm.return_value.invoke.return_value = supervisor_response
            res_llm.return_value.bind_tools.return_value.invoke.return_value = researcher_response
            mock_search.invoke.return_value = MOCK_SEARCH_RESULT
            sum_llm.return_value.invoke.return_value = summary_response
            wri_llm.return_value.invoke.return_value = report_response

            graph = build_graph()
            result = graph.invoke(base_state())

        assert result["final_report"] == "## Final Report\nAI is advancing rapidly."
        assert result["summary"] == "Summary of AI findings."
        assert len(result["search_results"]) == 1

    def test_supervisor_refined_query_propagates_to_downstream_agents(self):
        refined = "Refined: quantum computing breakthroughs 2025"
        researcher_response = AIMessage(content="", tool_calls=[])

        with patch("agents.supervisor.ChatOpenAI") as sup_llm, \
             patch("agents.researcher.ChatOpenAI") as res_llm, \
             patch("agents.researcher.web_search") as mock_search, \
             patch("agents.summarizer.ChatOpenAI") as sum_llm, \
             patch("agents.writer.ChatOpenAI") as wri_llm:

            sup_llm.return_value.invoke.return_value = make_ai_message(refined)
            res_llm.return_value.bind_tools.return_value.invoke.return_value = researcher_response
            mock_search.invoke.return_value = MOCK_SEARCH_RESULT
            sum_llm.return_value.invoke.return_value = make_ai_message("Summary.")
            wri_llm.return_value.invoke.return_value = make_ai_message("Report.")

            graph = build_graph()
            result = graph.invoke(base_state(query="quantum computing"))

        assert result["query"] == refined
        sum_invoke_args = sum_llm.return_value.invoke.call_args[0][0]
        assert refined in sum_invoke_args[-1].content
