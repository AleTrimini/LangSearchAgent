import os
from dotenv import load_dotenv
from graph import build_graph
from state import ResearchState

load_dotenv()


def run_research(query: str) -> dict:
    graph = build_graph()

    initial_state: ResearchState = {
        "query": query,
        "messages": [],
        "search_results": [],
        "summary": "",
        "final_report": "",
        "next": "",
    }

    print(f"\n{'='*60}")
    print(f"Starting research pipeline for: {query}")
    print(f"{'='*60}\n")

    result = graph.invoke(initial_state)

    print("\n--- FINAL REPORT ---\n")
    print(result["final_report"])

    return result


if __name__ == "__main__":
    query = input("Enter your research query: ").strip()
    if not query:
        query = "What are the latest developments in quantum computing in 2025?"

    run_research(query)
#ssss