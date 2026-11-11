import re
import numpy as np
from typing import List, Dict, Optional, Set
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
import urllib.request
import json
import urllib.parse

from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.claim import AtomicClaim
from proofgraph.retriever.fire_policy import FIREIterativeDecisionModule
from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent

import hashlib

class DeterministicHashEmbedding:
    """
    Offline fallback embedding model for network-restricted environments and fast unit testing.
    Produces deterministic 384-dimensional normalized vector embeddings using token hashing.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]):
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = re.findall(r'\w+', text.lower())
            if not words:
                yield vec
                continue
            for word in words:
                h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec /= norm
            yield vec

class HybridRetriever:
    """
    Multi-Stage Hybrid Retriever with FIRE Iterative Loop:
    - Sparse Lexical: BM25 (rank_bm25) for high-precision entity/number matches
    - Dense Semantic: FastEmbed (BAAI/bge-small-en-v1.5) for conceptual alignment
    - Reciprocal Rank Fusion (RRF, k=60)
    - FIRE Iterative Retrieval Loop: Dynamic stopping with entity-gap query reformulation
    - Live Wikipedia API fallback for open-domain fact verification
    - Robust offline fallback when running in firewalled or sandboxed environments
    """
    def __init__(
        self,
        embedding_model=None,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        max_hops: int = 3,
        prefer_offline: bool = False,
        tinyfish_agent: Optional[TinyFishWebAgent] = None
    ):
        self.passages: List[EvidencePassage] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.fire_module = FIREIterativeDecisionModule(max_hops=max_hops)
        self.tinyfish_agent = tinyfish_agent or TinyFishWebAgent()

        if embedding_model is not None:
            self.embed_model = embedding_model
        elif prefer_offline:
            self.embed_model = DeterministicHashEmbedding()
        else:
            try:
                self.embed_model = TextEmbedding(model_name=embedding_model_name)
            except Exception:
                # Graceful offline fallback: network restricted or model not cached
                self.embed_model = DeterministicHashEmbedding()


    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def index_passages(self, passages: List[EvidencePassage]):
        """Indexes a corpus of EvidencePassage objects for hybrid search."""
        if not passages:
            return
        self.passages = passages
        self.corpus_tokens = [self.tokenize(p.text) for p in passages]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
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

        # Compute normalized BM25 and dense scores to provide balanced multimodal relevance
        max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) > 0 and np.max(bm25_scores) > 0 else 1.0

        results: List[EvidencePassage] = []
        for idx in top_indices:
            p = self.passages[idx]
            norm_bm25 = max(0.0, float(bm25_scores[idx]) / max_bm25)
            dense_score = max(0.0, float(dense_sims[idx]))
            # Multimodal hybrid relevance fusing lexical keyword match with semantic similarity
            relevance = 0.5 * norm_bm25 + 0.5 * dense_score
            res_passage = p.model_copy()
            res_passage.relevance_score = max(0.0, min(1.0, float(relevance)))
            results.append(res_passage)

        return results

    def search_tinyfish_live(self, query: str, max_results: int = 3) -> List[EvidencePassage]:
        """Dispatches an autonomous fact-checking query to the TinyFish Web Agent."""
        if not self.tinyfish_agent.is_available:
            return []
        return self.tinyfish_agent.research_claim_on_web(query, max_evidence=max_results)

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
                        credibility_score=0.88,
                        relevance_score=0.75,
                        metadata={"title": title, "pageid": page_id}
                    )
                )
            return live_passages
        except Exception:
            return []

    def _deduplicate_passages(self, passages: List[EvidencePassage], threshold: float = 0.90) -> List[EvidencePassage]:
        """Filters out near-duplicate passages to avoid context redundancy (SANCTUARY pattern)."""
        if len(passages) <= 1:
            return passages
        texts = [p.text for p in passages]
        embeddings = np.array(list(self.embed_model.embed(texts)))
        sim_matrix = self._cosine_similarity(embeddings, embeddings)
        
        keep_indices = []
        for i in range(len(passages)):
            is_dup = False
            for j in keep_indices:
                if sim_matrix[i, j] > threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)
                
        return [passages[i] for i in keep_indices]

    def retrieve(self, claim: AtomicClaim, top_k: int = 5, use_live_fallback: bool = True) -> List[EvidencePassage]:
        """
        Executes the FIRE iterative retrieval loop:
        1. Formulates initial query.
        2. Evaluates retrieved evidence pool.
        3. Decides whether to terminate early or generate next targeted query.
        4. Continues across hops until evidence is sufficient or max_hops is reached.
        """
        current_query = claim.text
        executed_queries = [current_query]
        pooled_evidence: List[EvidencePassage] = []
        seen_ids: Set[str] = set()

        for hop in range(1, self.fire_module.max_hops + 1):
            # 1. Search local corpus
            local_hits = self.search_local(current_query, top_k=top_k)
            for hit in local_hits:
                if hit.id not in seen_ids:
                    seen_ids.add(hit.id)
                    pooled_evidence.append(hit)

            # 2. Live fallback if local corpus is empty or top relevance is weak
            if use_live_fallback and (not pooled_evidence or pooled_evidence[0].relevance_score < 0.65):
                # Priority 2a: Query TinyFish autonomous web research agent if configured
                if self.tinyfish_agent.is_available:
                    tf_hits = self.tinyfish_agent.research_claim_on_web(current_query, max_evidence=top_k)
                    for hit in tf_hits:
                        if hit.id not in seen_ids:
                            seen_ids.add(hit.id)
                            pooled_evidence.append(hit)

                # Priority 2b: Wikipedia live search API fallback
                if not pooled_evidence or pooled_evidence[0].relevance_score < 0.65:
                    live_hits = self.search_wikipedia_live(current_query, max_results=top_k)
                    for hit in live_hits:
                        if hit.id not in seen_ids:
                            seen_ids.add(hit.id)
                            pooled_evidence.append(hit)

            # Sort pooled evidence by relevance
            pooled_evidence.sort(key=lambda x: x.relevance_score, reverse=True)

            # 3. FIRE Decision Module: Should we terminate early or reformulate query?
            should_stop, next_query = self.fire_module.evaluate_step(
                claim=claim,
                retrieved_pool=pooled_evidence,
                current_hop=hop,
                executed_queries=executed_queries
            )

            if should_stop or not next_query:
                break

            current_query = next_query
            executed_queries.append(current_query)

        # Deduplicate and return top_k
        deduped = self._deduplicate_passages(pooled_evidence)
        return deduped[:top_k]
