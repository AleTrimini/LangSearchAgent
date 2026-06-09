from ddgs import DDGS
from langchain_core.tools import tool
from typing import List


@tool
def web_search(query: str) -> str:
    """Search the web for information about a given query."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
            for r in results
        )
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def multi_search(queries: List[str]) -> List[str]:
    """Run multiple web searches and return all results."""
    results = []
    for q in queries:
        results.append(f"Query: {q}\nResult: {web_search.invoke({'query': q})}")
    return results
