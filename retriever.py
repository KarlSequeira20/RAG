"""Hybrid retrieval: keyword search and vector search, merged.

Vector search matches meaning, so it handles a question phrased differently from
the text. It is weak at exact tokens -- proper nouns, model names, numbers -- because
those carry little meaning to compress into a vector.

Keyword search (BM25) is the mirror image: it matches exact words and nothing else,
so it nails "FloorPlanCAD" or "15,000" but fails completely when the question and the
document use different words for the same idea.

Running both and combining the rankings covers each one's blind spot.
"""

import re

import chromadb
import ollama
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi

from build_index import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

# Standard constant from the Reciprocal Rank Fusion paper. It damps the difference
# between the top ranks so a single search method cannot dominate the merged list.
RRF_K = 60

# RRF rewards agreement, which becomes a problem when only one method can possibly
# find something. The chunk listing FloorPlanCAD's three contributions ranks #1 by
# keyword and #250 by vector -- it is exactly the chunk we need, and pure RRF buries
# it beneath chunks both methods rank moderately. Reserving a couple of slots for each
# method's own top hits keeps a confident single-source result from being out-voted.
RESERVED_PER_METHOD = 2

# Reranking model. A cross-encoder: unlike the embedding model, which encodes the
# question and a chunk separately into fixed vectors, this reads the question and one
# chunk *together* and scores how well that chunk answers that question. Much more
# accurate, and far too slow to run over the whole corpus -- so retrieval narrows the
# field to a handful of candidates first and this only reorders those.
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# How many candidates to hand the reranker. Measured: hit@10 is 100% while hit@6 is 94%,
# so the correct chunk is nearly always inside the first 10-20 candidates but not always
# ranked highly. That gap is exactly what reranking is for.
RERANK_CANDIDATES = 20

# Chunks scoring below this after reranking are dropped rather than sent to the model.
# Measured across a mixed corpus (floorplan papers, a Wikipedia article, a travel site):
# genuinely relevant chunks score 0.85-1.00 and irrelevant ones score exactly 0.0000 --
# the cross-encoder is decisive about irrelevance, so the gap is enormous and the exact
# cutoff barely matters. 0.05 sits far below any observed real hit.
#
# This matters most on a mixed corpus. Retrieval always returns k results whether or not
# anything is relevant, so without a floor a question about one topic arrives at the
# model padded with confidently-irrelevant chunks from another.
MIN_RELEVANCE = 0.05


def tokenize(text):
    """Split into lowercase word tokens for keyword matching.

    Hyphens are removed before splitting so that "FloorPlan-CAD" and "FloorPlanCAD"
    produce the same token. PDFs break words across lines and papers are inconsistent
    about hyphenating their own names; as long as documents and queries are normalised
    the same way, the two still match.
    """
    return re.findall(r"\w+", text.lower().replace("-", ""))


class HybridRetriever:
    def __init__(self):
        self.collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
            COLLECTION_NAME
        )
        # BM25 needs the whole corpus in memory to compute word statistics. Fine at
        # this scale (a few hundred chunks); a larger corpus would want a real
        # keyword index such as Elasticsearch instead.
        data = self.collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        self.position_of = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(d) for d in self.documents])
        self._ranker = None  # loaded on first rerank, not on construction

    def vector_ranking(self, question, depth):
        """Indices of the chunks closest in meaning, best first."""
        vector = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
        result = self.collection.query(query_embeddings=[vector], n_results=depth)
        # Chroma returns chunk ids; map them to positions in our in-memory lists so
        # both rankings speak the same language.
        return [self.position_of[chunk_id] for chunk_id in result["ids"][0]]

    def keyword_ranking(self, question, depth):
        """Indices of the chunks with the strongest exact-word overlap, best first."""
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:depth] if scores[i] > 0]

    def rerank(self, question, candidates, k):
        """Rescore candidates with the cross-encoder and keep the best k."""
        if self._ranker is None:
            self._ranker = Ranker(model_name=RERANK_MODEL)

        request = RerankRequest(
            query=question,
            passages=[{"id": idx, "text": self.documents[idx]} for idx, _score in candidates],
        )
        ranked = self._ranker.rerank(request)
        return [(item["id"], item["score"]) for item in ranked[:k]]

    def search(self, question, k=4, depth=30, rerank=True, sources=None):
        """Merge both rankings with Reciprocal Rank Fusion.

        Each list votes for a chunk with 1/(RRF_K + rank). A chunk that both methods
        rank moderately well beats one that only a single method loves, which is
        exactly the behaviour we want -- it is the disagreements that cause misses.
        """
        rankings = (self.vector_ranking(question, depth), self.keyword_ranking(question, depth))

        # Optional restriction to specific documents. Filtering after ranking (rather
        # than querying a subset) keeps one code path, at the cost of searching chunks
        # that will be discarded -- negligible at this scale.
        if sources is not None:
            allowed = set(sources)
            rankings = tuple(
                [i for i in ranking if self.metadatas[i]["source"] in allowed]
                for ranking in rankings
            )

        scores = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking, start=1):
                scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank)

        # Each method's own top picks go in first, then RRF fills the rest.
        # When reranking, gather a wider candidate pool -- the cross-encoder decides the
        # final order, so the job here is recall, not precision.
        want = RERANK_CANDIDATES if rerank else k
        chosen = []
        for ranking in rankings:
            for idx in ranking[:RESERVED_PER_METHOD]:
                if idx not in chosen:
                    chosen.append(idx)

        for idx in sorted(scores, key=lambda i: scores[i], reverse=True):
            if len(chosen) >= want:
                break
            if idx not in chosen:
                chosen.append(idx)

        candidates = [(i, scores.get(i, 0.0)) for i in chosen[:want]]
        if rerank:
            candidates = self.rerank(question, candidates, k)
            # Only reranker scores are comparable across questions; RRF scores are not,
            # so the floor is applied only when reranking actually ran.
            candidates = [(i, s) for i, s in candidates if s >= MIN_RELEVANCE]

        return [(self.documents[i], self.metadatas[i], s) for i, s in candidates[:k]]
