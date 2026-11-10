import re
import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
import urllib.request
import json
import urllib.parse

from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.claim import AtomicClaim
from proofgraph.retriever.fire_policy import FIREStoppingPolicy

class HybridRetriever:
    """
    Two-stage Hybrid Retriever:
    - Sparse Lexical: BM25 (rank_bm25) for high-precision entity/number matches
    - Dense Semantic: FastEmbed (BAAI/bge-small-en-v1.5) for conceptual alignment
    - Reciprocal Rank Fusion (RRF, k=60)
    - Fallback: Live Wikipedia API search when local corpus lacks sufficient evidence
    """
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embed_model = TextEmbedding(model_name=embedding_model_name)
        self.passages: List[EvidencePassage] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.fire_policy = FIREStoppingPolicy()

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def index_passages(self, passages: List[EvidencePassage]):
        """Indexes a corpus of EvidencePassage objects for hybrid search."""
        if not passages:
            return
        self.passages = passages
        self.corpus_tokens = [self.tokenize(p.text) for p in passages]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # Compute dense embeddings
        texts = [p.text for p in passages]
        embeddings_gen = self.embed_model.embed(texts)
        self.dense_embeddings = np.array(list(embeddings_gen))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
        norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
        return np.dot(a / norm_a, (b / norm_b).T)

    def search_local(self, query: str, top_k: int = 5) -> List[EvidencePassage]:
        """Performs hybrid BM25 + dense semantic retrieval over indexed corpus with RRF fusion."""
        if not self.passages or self.bm25 is None or self.dense_embeddings is None:
            return []

        # 1. BM25 scores & ranking
        query_tokens = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked_idx = np.argsort(bm25_scores)[::-1]

        # 2. Dense semantic similarity & ranking
        query_emb = np.array(list(self.embed_model.embed([query])))
        dense_sims = self._cosine_similarity(query_emb, self.dense_embeddings)[0]
        dense_ranked_idx = np.argsort(dense_sims)[::-1]

        # 3. Reciprocal Rank Fusion (k=60)
        rrf_scores = np.zeros(len(self.passages))
        k = 60
        for rank, idx in enumerate(bm25_ranked_idx):
            rrf_scores[idx] += 1.0 / (k + rank + 1)
        for rank, idx in enumerate(dense_ranked_idx):
            rrf_scores[idx] += 1.0 / (k + rank + 1)

        # Sort by final RRF score
        top_indices = np.argsort(rrf_scores)[::-1][:top_k]
        
        results: List[EvidencePassage] = []
        for idx in top_indices:
            p = self.passages[idx]
            # Create a copy with updated relevance score
            relevance = float(dense_sims[idx])
            res_passage = p.model_copy()
            res_passage.relevance_score = max(0.0, min(1.0, relevance))
            results.append(res_passage)

        return results

    def search_wikipedia_live(self, query: str, max_results: int = 3) -> List[EvidencePassage]:
        """Live fallback search against Wikipedia API for open-domain claims."""
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json&utf8=1&srlimit={max_results}"
            req = urllib.request.Request(url, headers={"User-Agent": "ProofGraph-Research/1.0 (contact@proofgraph.ai)"})
            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode("utf-8"))
                
            search_items = data.get("query", {}).get("search", [])
            live_passages = []
            for item in search_items:
                raw_snippet = item.get("snippet", "")
                clean_snippet = re.sub(r'<[^>]+>', '', raw_snippet)
                title = item.get("title", "")
                page_id = item.get("pageid", "")
                source_url = f"https://en.wikipedia.org/?curid={page_id}"
                
                live_passages.append(
                    EvidencePassage(
                        text=f"{title}: {clean_snippet}",
                        source_url=source_url,
                        source_title=f"Wikipedia: {title}",
                        source_domain="wikipedia.org",
                        credibility_score=0.88,  # Established encyclopedic baseline
                        relevance_score=0.75,
                        metadata={"title": title, "pageid": page_id}
                    )
                )
            return live_passages
        except Exception as e:
            return []

    def retrieve(self, claim: AtomicClaim, top_k: int = 5, use_live_fallback: bool = True) -> List[EvidencePassage]:
        """Executes hybrid retrieval with FIRE dynamic stopping policy and optional live web fallback."""
        query = claim.text
        # Query augmentation: if claim has entities, prioritize entity context
        if claim.entities:
            query = f"{' '.join(claim.entities[:2])} {claim.text}"

        retrieved = self.search_local(query, top_k=top_k)

        # FIRE Early-Stopping Check
        if self.fire_policy.should_terminate(claim, retrieved, current_hop=1):
            return retrieved

        # If local retrieval yielded low relevance or empty, fallback to live Wikipedia
        if use_live_fallback and (not retrieved or retrieved[0].relevance_score < 0.65):
            search_query = ' '.join(claim.entities[:3]) if claim.entities else claim.text
            live_hits = self.search_wikipedia_live(search_query, max_results=top_k)
            if live_hits:
                # Merge local and live hits, avoiding near-duplicates
                retrieved = live_hits + retrieved
                retrieved = retrieved[:top_k]

        return retrieved
