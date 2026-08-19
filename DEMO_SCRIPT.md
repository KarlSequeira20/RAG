# Demo video script — 85 seconds

Recording notes, not part of the system. Loom or QuickTime is fine.

## Before you hit record

- **Warm the model.** Ask one throwaway question first. A cold `llama3.1:8b` takes
  several extra seconds on its first call, and that dead air is the difference between
  looking fast and looking broken.
- **Decide on the corpus.** Either remove the travel-site URL for a clean papers-only
  demo, or keep it and use it deliberately in the URL section. Don't leave it sitting
  there unexplained.
- **Restart Streamlit** so the sidebar is fresh and no cached state is showing.
- **Browser at 100% zoom**, window sized so the sidebar and answer are both visible
  without scrolling. Close other tabs — the tab bar is in shot.
- **Answers take about 5 seconds.** Cut those gaps in editing rather than narrating over
  them. Two seconds of silence reads as a pause; five reads as a hang.

---

## 0:00–0:10 — What it is

**Show:** the app, question box empty, sidebar visible.

> "This answers questions about a set of research papers. It runs entirely on my laptop —
> no API keys — and every answer shows exactly where it came from."

Keep this to one breath. Nobody watches a demo for the preamble.

---

## 0:10–0:30 — An answer with a checkable source

**Type:** `How many samples does the CubiCasa5K dataset contain?`

**Show:** the answer, then click open source **[1]** so the chunk text is visible.

> "It says 5,000, and cites source one. Opening it shows the exact text from page 4 of
> the paper that the model was given. So I'm not trusting the model — I can check it."

**This is the shot that matters.** The expanded source is the whole point of the project.
Hold on it for a beat longer than feels natural.

---

## 0:30–0:45 — The refusal

**Type:** `What is the architecture of the BERT language model?`

**Show:** "That isn't covered in these documents."

> "Now something the papers don't cover. It refuses — and the model definitely knows what
> BERT is. That's the proof it's answering from my documents rather than from memory.
> It didn't even call the model here: nothing scored above the relevance threshold, so
> there was nothing to answer from."

Note this comes back in about a second, unlike the others. Worth pointing out — it looks
like a bug if you don't explain the speed difference.

---

## 0:45–1:00 — Add a source live

**Show:** paste a URL into the sidebar, click Index, then ask a question about it.

Suggested: the Wikipedia RAG article, then ask `What is retrieval-augmented generation?`

> "It's not limited to the PDFs. I can point it at any web page — it strips out the
> navigation and ads, keeps the article, and indexes it the same way. Now it answers from
> that too, cited by section instead of page number."

If indexing takes more than a few seconds, cut it. Show the click and then the result.

---

## 1:00–1:20 — Why it works, and what broke

**Show:** the README results table, or just talk over the app.

> "Retrieval is hybrid — meaning-based search plus keyword search — because meaning-based
> search alone couldn't tell four papers about floorplans apart. A cross-encoder then
> reranks the results. The correct chunk reaches the model 100% of the time on my
> evaluation set.
>
> The interesting part was the bugs. One paper had the LaTeX source of every equation
> hidden inside it — a quarter of my whole index was invisible junk, and nothing in the
> output looked wrong. I only found it by measuring."

**This is the section that separates you from other candidates.** Everyone demos a
working RAG app; almost nobody can name a bug they found by measurement.

---

## 1:20–1:25 — Close

> "Code, evaluation numbers and the full write-up are in the repo."

**Show:** the GitHub URL on screen. Say it once, don't spell it out.

---

## Things to avoid

- Reading the answer aloud while it's on screen. The viewer can read.
- Explaining what RAG is. Anyone watching either knows or doesn't care.
- Apologising for the 8B model, the local setup, or anything else. State what it does.
- Showing code. Nobody watches code in a video; that's what the repo is for.
- Going over 90 seconds. If it's running long, cut section 0:45–1:00 first — web
  ingestion is a nice-to-have; the citation and the refusal are not.
