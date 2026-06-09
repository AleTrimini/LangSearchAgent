# LangSearch — Multi-Agent Research Pipeline

Automated research pipeline built with **LangGraph** and **LangChain**, with a Streamlit web interface. Give it a question, get back a structured markdown report — web search, summarization and writing fully automated through a supervised agent graph.

## Architecture

```
User Query
    │
    ▼
Supervisor Agent  ── refines the query
    │
    ├──► Researcher Agent ──► DuckDuckGo Search
    │
    ▼
Summarizer Agent ──► condenses raw results
    │
    ▼
Writer Agent ──► final markdown report
```

The flow is orchestrated by **LangGraph** via a shared-state graph with fixed edges: Supervisor → Researcher → Summarizer → Writer.

## Agents

| Agent | Role |
|---|---|
| **Supervisor** | Refines the query before passing it to the pipeline |
| **Researcher** | Generates search queries and fetches results via DuckDuckGo |
| **Summarizer** | Condenses raw results into a structured summary |
| **Writer** | Produces a professional final report in markdown |

## Features

- **Free web search** via DuckDuckGo — no additional API key required
- **Live pipeline status**: each agent shows its progress in real time
- **Progressive output**: search results, summary and final report appear as they are produced
- **Downloadable report** as a `.txt` file at the end of the run
- **API key from the UI**: enter it directly in the sidebar without editing any config file
- **Per-agent model selection**: each agent can use a different OpenAI model (e.g. `gpt-4o` for the writer, `gpt-4o-mini` for the summarizer) directly from the sidebar

## Project structure

```
LangSearch/
├── pyproject.toml
├── requirements.txt
├── config.py           ← OpenAI model config per agent
├── app.py              ← Streamlit web interface
├── graph.py            ← LangGraph graph definition
├── state.py            ← shared state across agents
├── main.py             ← CLI entry point
├── agents/
│   ├── supervisor.py
│   ├── researcher.py
│   ├── summarizer.py
│   └── writer.py
├── tools/
│   └── search.py       ← DuckDuckGo tool
└── tests/
    └── test_pipeline.py ← end-to-end tests with mocks
```

## Requirements

- Python 3.10+
- OpenAI account with an API key

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AleTrimini/LangSearch.git
cd LangSearch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the API key
cp .env.example .env
# edit .env and add your OPENAI_API_KEY
```

## Usage

**Web interface (recommended)**
```bash
streamlit run app.py
```
Opens automatically at `http://localhost:8501`

**CLI**
```bash
python main.py
```

## Tests

```bash
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — multi-agent graph orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM abstraction and tool use
- [OpenAI GPT-4o](https://platform.openai.com/docs/models) — LLM backbone
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) — free web search
- [Streamlit](https://streamlit.io) — web interface
