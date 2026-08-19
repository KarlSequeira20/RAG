"""HTTP API over the RAG pipeline.

Run with:  ./.venv/bin/uvicorn api:app --reload

One endpoint that takes a question and returns an answer with its sources, plus a
health check. The interesting part is not the routing -- it is that the retriever is
built once at startup rather than per request (see `lifespan` below).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ask import LLM_MODEL, TOP_K, answer
from retriever import HybridRetriever

MAX_QUESTION_CHARS = 1000

# Populated at startup. Building a HybridRetriever reads every chunk out of Chroma and
# computes BM25 word statistics over the whole corpus. That is fine once, and wasteful
# on every request -- so it happens here, not inside the endpoint.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the retriever before the server accepts traffic, and hold it."""
    state["retriever"] = HybridRetriever()
    yield
    state.clear()


app = FastAPI(
    title="RAG over floorplan research papers",
    description="Ask a question, get an answer grounded in the indexed PDFs, with sources.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    # Pydantic enforces these before the endpoint runs, so an empty or oversized
    # question is rejected with a 422 and never reaches the model.
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="A question about the indexed papers.",
    )


class Source(BaseModel):
    n: int
    title: str
    source: str
    page: str
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model: str


@app.get("/health")
def health():
    """Cheap liveness check that also reports how much is indexed."""
    retriever = state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded.")
    return {"status": "ok", "chunks_indexed": len(retriever.documents), "model": LLM_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        # min_length=1 still allows a question that is only whitespace.
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    try:
        text, hits = answer(question, retriever=state["retriever"])
    except Exception as exc:
        # Most likely Ollama is not running or the model is not pulled. Report it as a
        # dependency failure rather than letting a stack trace surface as a 500.
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the language model ({type(exc).__name__}). Is Ollama running?",
        ) from exc

    sources = [
        Source(
            n=i,
            title=meta["title"],
            source=meta["source"],
            page=meta["page_label"],
            snippet=chunk[:300],
        )
        for i, (chunk, meta, _score) in enumerate(hits, start=1)
    ]
    return AskResponse(question=question, answer=text, sources=sources, model=LLM_MODEL)
