# RAG "Chat With Documents" — Project Roadmap

**Goal:** Build and deploy a working RAG (Retrieval-Augmented Generation) system that answers questions from a set of PDFs, with source citations. End result: a public GitHub repo + live deployed URL + demo video, portfolio-ready for AI/ML fresher roles.

**Why this project:** RAG is one of the most in-demand skills in AI engineering hiring right now. This teaches you the real mechanics behind LLM apps — chunking, embeddings, retrieval, evaluation, deployment — not just prompting.

**Total time:** 3 weeks, a few hours on weekdays + more on weekends.

**How to use this with Claude Code:** Work through one day at a time. Don't jump ahead. At the start of each session, tell Claude Code which Day you're on and paste in the relevant section below. See `CLAUDE.md` for how Claude should behave while helping you (short answers, no spoon-feeding).

---

## Week 1 — Core Pipeline (get it working locally)

### Day 1-2: Setup + Document Ingestion
- [ ] Set up Python environment (venv or conda)
- [ ] Install: `langchain`, `chromadb`, `pypdf`
- [ ] Pick your document set — use something you have a real reason to build (Atlas coursework notes, or a public technical doc set). Avoid a random generic Kaggle PDF — a real use case makes your README and interview story stronger.
- [ ] Write an ingestion script:
  - Load PDFs
  - Split into chunks (research: what's a reasonable chunk size and why does it matter?)
  - Generate embeddings for each chunk
  - Store embeddings in ChromaDB
- **Checkpoint:** You can run one script and see chunks stored in your local Chroma DB.

### Day 3-4: Retrieval + Generation
- [ ] Build the retrieval chain: query → embed the query → fetch top-k similar chunks → pass chunks + question to the LLM
- [ ] Start with Ollama locally (you already know this from work) — can swap to a hosted API later
- [ ] Get a basic question-answering loop working end-to-end (script or notebook is fine for now)
- **Checkpoint:** You can ask a question in your terminal and get a relevant answer back.

### Day 5-7: Source Citation (your standout feature)
- [ ] Modify the pipeline to track which chunk/page each answer came from
- [ ] Surface the source (page number, chunk snippet) alongside every answer
- [ ] Test with 10-15 realistic questions. For each one, manually check: Is the answer actually grounded in the cited source, or is it hallucinating?
- [ ] Expect to find bugs here — bad chunking, wrong retrieval, mismatched citations. This is normal and where most of the real learning happens.
- **Checkpoint:** Every answer comes with an accurate, verifiable source.

---

## Week 2 — Turn It Into a Real Application

### Day 8-10: API Layer
- [ ] Wrap the pipeline in FastAPI — one endpoint that takes a question, returns an answer + sources
- [ ] Add error handling: What happens with an empty query? An unanswerable question? A huge PDF? A malformed request?
- **Checkpoint:** You can hit your API with a tool like curl/Postman and get a clean JSON response, including for bad inputs.

### Day 11-12: Frontend
- [ ] Build a simple UI in Streamlit or Gradio: upload a PDF, ask questions, see answer + cited sources
- [ ] Keep it functional and clean — don't overengineer the UI, the pipeline is the point
- **Checkpoint:** Someone unfamiliar with your code could open the app and use it without instructions.

### Day 13-14: Evaluation
- [ ] Write a small eval set: 20-30 question/expected-answer pairs
- [ ] Measure basic retrieval accuracy — are the right chunks being pulled for each question?
- [ ] Optional but valuable: look into RAGAS for a standard evaluation metric you can quote in your README
- **Checkpoint:** You have real numbers describing how well your system performs, not just a gut feeling.

---

## Week 3 — Deploy and Document

### Day 15-17: Deployment
- [ ] Deploy on Hugging Face Spaces or Streamlit Community Cloud (both free, both give a public URL)
- [ ] Test from a fresh browser/device (or ask a friend to try it) to confirm it works for someone who isn't you
- **Checkpoint:** A stranger can open your link and use the app right now.

### Day 18-19: Documentation
- [ ] Write a proper README: what it does, architecture diagram (even a simple hand-drawn/draw.io one is fine), how to run it locally, your eval results, what you'd improve with more time
- [ ] Record a 60-90 second demo video (Loom is fine) showing it working
- **Checkpoint:** Someone could understand and evaluate your project from the README alone, without running any code.

### Day 20-21: Polish + Buffer
- [ ] Fix whatever broke during deployment (something usually does)
- [ ] Clean up code comments and commit history
- [ ] Write a short LinkedIn post or portfolio blurb about it
- **Checkpoint:** Repo is public, live, documented, and something you'd be comfortable linking in a resume.

---

## Ground Rules for This Project
1. **If you're stuck on something for a whole day, ship an imperfect version and note the limitation in the README rather than stalling.** A working V1 beats a stalled V2.
2. **You're building this to learn, not just to finish it.** If you don't understand why a step works, stop and understand it before moving to the next step — see `CLAUDE.md` for how your AI assistant should help you get there without doing it for you.
3. **After this project ships, the natural next one is the floor-plan/CV object detection project** — it'll go faster because you'll already have the ingest → process → serve → deploy muscle memory from this one.
