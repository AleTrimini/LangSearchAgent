import time
from ddgs import DDGS
from langchain_core.tools import tool

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


@tool
def web_search(query: str) -> str:
    """Search the web for information about a given query."""
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
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
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
    return f"Search failed after {_MAX_RETRIES} attempts: {last_error}"
