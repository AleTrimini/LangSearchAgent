"""End-to-end tests for the research pipeline using mocked LLM and search tool."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock
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
        "next": "",
    }
    return {**defaults, **kwargs}


# --- Supervisor ---

class TestSupervisorNode:
    def test_routes_to_researcher_when_no_results(self):
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message(
                "Latest AI developments in 2025"
            )
            result = supervisor_node(base_state())
        assert result["next"] == "researcher"

    def test_routes_to_summarizer_when_results_present(self):
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message(
                "Latest AI developments in 2025"
            )
            result = supervisor_node(base_state(search_results=[MOCK_SEARCH_RESULT]))
        assert result["next"] == "summarizer"

    def test_routes_to_writer_when_summary_present(self):
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message(
                "Latest AI developments in 2025"
            )
            result = supervisor_node(
                base_state(search_results=[MOCK_SEARCH_RESULT], summary="A summary.")
            )
        assert result["next"] == "writer"

    def test_routes_to_end_when_report_present(self):
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message(
                "Latest AI developments in 2025"
            )
            result = supervisor_node(
                base_state(
                    search_results=[MOCK_SEARCH_RESULT],
                    summary="A summary.",
                    final_report="A report.",
                )
            )
        assert result["next"] == "END"

    def test_refines_query_from_llm_response(self):
        refined = "Latest advances in artificial intelligence and machine learning 2025"
        with patch("agents.supervisor.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message(refined)
            result = supervisor_node(base_state())
        assert result["query"] == refined


# --- Researcher ---

class TestResearcherNode:
    def test_populates_search_results(self):
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

        assert len(result["search_results"]) == 1
        assert result["search_results"][0] == MOCK_SEARCH_RESULT
        assert result["next"] == "summarizer"

    def test_falls_back_to_direct_search_when_no_tool_calls(self):
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = ""

        with patch("agents.researcher.ChatOpenAI") as mock_llm_cls, \
             patch("agents.researcher.web_search") as mock_search:
            mock_llm_cls.return_value.bind_tools.return_value.invoke.return_value = mock_response
            mock_search.invoke.return_value = MOCK_SEARCH_RESULT
            result = researcher_node(base_state())

        assert result["search_results"] == [MOCK_SEARCH_RESULT]


# --- Summarizer ---

class TestSummarizerNode:
    def test_produces_summary(self):
        with patch("agents.summarizer.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("Key findings: ...")
            result = summarizer_node(base_state(search_results=[MOCK_SEARCH_RESULT]))

        assert result["summary"] == "Key findings: ..."
        assert result["next"] == "writer"


# --- Writer ---

class TestWriterNode:
    def test_produces_final_report(self):
        with patch("agents.writer.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value.invoke.return_value = make_ai_message("## Report\n...")
            result = writer_node(base_state(summary="A summary."))

        assert result["final_report"] == "## Report\n..."
        assert result["next"] == "END"


# --- Full graph (end-to-end) ---

class TestGraph:
    def test_full_pipeline_produces_report(self):
        supervisor_response = make_ai_message("Latest AI developments in 2025")
        summary_response = make_ai_message("Summary of AI findings.")
        report_response = make_ai_message("## Final Report\nAI is advancing rapidly.")

        mock_researcher_response = AIMessage(content="")
        mock_researcher_response.tool_calls = []

        with patch("agents.supervisor.ChatOpenAI") as sup_llm, \
             patch("agents.researcher.ChatOpenAI") as res_llm, \
             patch("agents.researcher.web_search") as mock_search, \
             patch("agents.summarizer.ChatOpenAI") as sum_llm, \
             patch("agents.writer.ChatOpenAI") as wri_llm:

            sup_llm.return_value.invoke.return_value = supervisor_response
            res_llm.return_value.bind_tools.return_value.invoke.return_value = mock_researcher_response
            mock_search.invoke.return_value = MOCK_SEARCH_RESULT
            sum_llm.return_value.invoke.return_value = summary_response
            wri_llm.return_value.invoke.return_value = report_response

            graph = build_graph()
            result = graph.invoke(base_state())

        assert result["final_report"] == "## Final Report\nAI is advancing rapidly."
        assert result["summary"] == "Summary of AI findings."
        assert len(result["search_results"]) == 1
