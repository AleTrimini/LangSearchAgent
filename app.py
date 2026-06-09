import os
import streamlit as st
from dotenv import load_dotenv
from graph import build_graph
from state import ResearchState
from config import AgentModels

load_dotenv()

AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

st.set_page_config(
    page_title="Research Pipeline",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Research Pipeline")
st.caption("Multi-agent pipeline: Supervisor → Researcher → Summarizer → Writer")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if effective_key:
        os.environ["OPENAI_API_KEY"] = effective_key

    st.divider()
    st.markdown("**Model per agent**")
    model_supervisor = st.selectbox("🧭 Supervisor", AVAILABLE_MODELS, index=0)
    model_researcher = st.selectbox("🔎 Researcher", AVAILABLE_MODELS, index=0)
    model_summarizer = st.selectbox("📝 Summarizer", AVAILABLE_MODELS, index=1)
    model_writer = st.selectbox("✍️ Writer", AVAILABLE_MODELS, index=0)

query = st.text_input(
    "Enter your research question",
    placeholder="e.g. What are the latest developments in AI in 2025?",
)

run_button = st.button("Start Research", type="primary", disabled=not query)

if run_button and query:
    if not effective_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
        st.stop()

    models = AgentModels(
        supervisor=model_supervisor,
        researcher=model_researcher,
        summarizer=model_summarizer,
        writer=model_writer,
    )
    graph = build_graph(models=models)

    initial_state: ResearchState = {
        "query": query,
        "messages": [],
        "search_results": [],
        "summary": "",
        "final_report": "",
        "next": "",
    }

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📡 Pipeline status")
        status_supervisor = st.status("🧭 Supervisor", state="running")
        status_researcher = st.status("🔎 Researcher", state="running")
        status_summarizer = st.status("📝 Summarizer", state="running")
        status_writer = st.status("✍️ Writer", state="running")

    with col2:
        st.subheader("📄 Output")
        search_expander = st.expander("Web search results", expanded=False)
        summary_expander = st.expander("Summary", expanded=False)
        report_placeholder = st.empty()

    try:
        status_supervisor.update(state="complete")
        status_researcher.update(state="running")

        for step in graph.stream(initial_state):
            node_name = list(step.keys())[0]
            node_state = step[node_name]

            if node_name == "researcher":
                results = node_state.get("search_results", [])
                with search_expander:
                    for i, r in enumerate(results, 1):
                        st.markdown(f"**Result {i}**")
                        st.text(r[:800] + "..." if len(r) > 800 else r)
                        st.divider()
                status_researcher.update(state="complete")
                status_summarizer.update(state="running")

            elif node_name == "summarizer":
                summary = node_state.get("summary", "")
                with summary_expander:
                    st.markdown(summary)
                status_summarizer.update(state="complete")
                status_writer.update(state="running")

            elif node_name == "writer":
                report = node_state.get("final_report", "")
                report_placeholder.markdown(report)
                status_writer.update(state="complete")

        st.success("Research complete!")

        final_state = node_state
        if final_state.get("final_report"):
            st.download_button(
                label="📥 Download report",
                data=final_state["final_report"],
                file_name=f"report_{query[:30].replace(' ', '_')}.txt",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"Pipeline error: {e}")
