"""Streamlit UI for the RAG pipeline.

Run with:  ./.venv/bin/streamlit run app.py

Imports the pipeline directly rather than calling the FastAPI service, so the whole
app is one process. That matters for deployment -- Hugging Face Spaces runs a single
process, and a separate API server would be extra plumbing for no benefit here.
`api.py` remains the programmatic interface for anything that is not this UI.
"""

from pathlib import Path

import streamlit as st

from ask import LLM_MODEL, TOP_K, answer, describe_location
from build_index import get_collection, index_pdf, index_url
from ingest import DATA_DIR
from retriever import HybridRetriever

st.set_page_config(page_title="Ask the floorplan papers", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Loading index...")
def load_retriever():
    """Build the retriever once per session, not once per question.

    Constructing it reads every chunk out of Chroma and computes BM25 statistics over
    the whole corpus. Streamlit re-runs this script top to bottom on every interaction,
    so without caching that cost would be paid on each keystroke-triggered rerun.
    """
    return HybridRetriever()


def index_uploaded_file(uploaded):
    """Save an uploaded PDF into data/ and add it to the index."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / uploaded.name
    path.write_bytes(uploaded.getbuffer())

    added, blanks = index_pdf(get_collection(), path)
    load_retriever.clear()  # the corpus changed, so the cached retriever is stale
    return path, added, blanks


st.title("Ask the floorplan papers")
st.caption(
    "Answers come only from the indexed documents. Every answer lists the sources it "
    "used, so you can check it yourself."
)

with st.sidebar:
    st.header("Documents")

    retriever = load_retriever()
    # One entry per source, showing its title rather than a raw filename or URL --
    # a long URL wraps over several lines and says less than the page's own title.
    sources = {}
    for meta in retriever.metadatas:
        sources.setdefault(meta["source"], meta)

    st.write(f"**{len(retriever.documents)} chunks** from {len(sources)} document(s):")
    for source, meta in sorted(sources.items(), key=lambda kv: kv[1]["title"].lower()):
        chunks = sum(1 for m in retriever.metadatas if m["source"] == source)
        if meta.get("kind") == "web":
            st.markdown(f"- [{meta['title']}]({source}) · {chunks} chunks")
        else:
            st.markdown(f"- **{meta['title'][:60]}**  \n  `{source}` · {chunks} chunks")

    st.divider()
    uploaded = st.file_uploader("Add a PDF", type="pdf")
    if uploaded is not None and st.button("Index this PDF", use_container_width=True):
        with st.spinner(f"Reading and indexing {uploaded.name}..."):
            path, added, blanks = index_uploaded_file(uploaded)
        if added == 0:
            st.error(
                f"No text could be extracted from {uploaded.name}. "
                "It is most likely a scanned PDF -- pages that are images contain no "
                "text to read, and handling those would need OCR."
            )
        else:
            if blanks:
                st.warning(f"{blanks} page(s) had no extractable text and were skipped.")
            st.success(f"Indexed {uploaded.name}: {added} chunks.")
            st.rerun()

    st.divider()
    url = st.text_input("Or add a web page", placeholder="https://...")
    if url and st.button("Index this URL", use_container_width=True):
        with st.spinner(f"Fetching {url}..."):
            try:
                added, page_title = index_url(get_collection(), url.strip())
            except Exception as exc:
                st.error(str(exc))
            else:
                load_retriever.clear()
                st.success(f"Indexed “{page_title}”: {added} chunks.")
                st.rerun()
    st.caption(
        "A page is fetched once and indexed as a snapshot — it is not re-checked when "
        "you ask a question, so the content can go stale."
    )

    st.divider()
    st.caption(f"Model: `{LLM_MODEL}` · top-k: {TOP_K} · hybrid BM25 + vector retrieval")

question = st.text_input(
    "Your question",
    placeholder="e.g. How many samples does the CubiCasa5K dataset contain?",
)

if question.strip():
    try:
        with st.spinner("Retrieving and answering..."):
            text, hits = answer(question.strip(), retriever=load_retriever())
    except Exception as exc:
        st.error(
            f"Could not reach the language model ({type(exc).__name__}). "
            "Check that Ollama is running and the model is pulled."
        )
    else:
        st.markdown("### Answer")
        st.write(text)

        st.markdown("### Sources")
        st.caption(
            "The numbers match the citations in the answer. Open one to read the exact "
            "text the model was given."
        )
        for i, (chunk, meta, score) in enumerate(hits, start=1):
            label = f"[{i}] {describe_location(meta)}"
            with st.expander(label):
                st.caption(meta["title"])
                st.write(chunk)
