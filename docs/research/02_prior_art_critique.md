# Prior Art Analysis & Architectural Critique

To engineer ProofGraph to be measurably superior to existing automated fact-checking systems, we conducted a forensic analysis of existing implementations, research prototypes, and open-source frameworks. This document highlights what prior systems got right, their catastrophic failure modes, and how ProofGraph solves them.

---

## 1. The Superficial Prototype: TRUSTGRAPH
* **Repository:** [DataQUEEN99/TRUSTGRAPH](https://github.com/DataQUEEN99/TRUSTGRAPH-AI-Evidence-Intelligence-Claim-Verification-Platform)
* **Architecture:** Streamlit web application contained within a single Jupyter Notebook (), utilizing Pandas, Plotly, and NetworkX.
* **How it Operates:**
  - Uses naive regex string-splitting to segment text.
  - Hardcodes 6 fixed media outlets (Reuters, AP, BBC, FactCheck.org, NYT, Snopes).
  - Uses static mock reliability scores and random heuristics to simulate verification scores.
  - Knowledge graph is a static star graph: one central 'CLAIM' node connected by rigid edges to the 6 mock source nodes.
* **Fatal Flaws & Shortcomings:**
  1. **Zero Real Retrieval:** Does not query the web, vector stores, or knowledge graphs.
  2. **No NLI Reasoning:** Uses no transformer entailment models or LLMs for claim verification.
  3. **No Provenance:** Does not record passage spans, crawl timestamps, or transformation lineages.
* **ProofGraph Solution:** Build an enterprise-grade FastAPI service that executes live multi-stage hybrid retrieval, true 3-way NLI entailment classification, and dynamically constructs a W3C PROV-DM graph where evidence nodes, claim nodes, and contradiction edges reflect genuine model reasoning.

---

## 2. The Brittle Pipeline: LOKI / OpenFactVerification (COLING 2025)
* **Repository:** [Libr-AI/OpenFactVerification](https://github.com/Libr-AI/OpenFactVerification)
* **Architecture:** 5-step asynchronous pipeline: Check-worthiness Detection -> Query Formulation -> Evidence Retrieval (via Serper API) -> Claim-Evidence NLI Verification -> Report Generation.
* **Key Strengths:** Modular design, support for multilingual inputs, asynchronous execution.
* **Fatal Flaws & Shortcomings:**
  1. **Brittle Output Parsing:** Relies on open LLMs (LLaMA-3) to generate complex JSON. Without constrained decoding, JSON formatting frequently breaks, requiring fragile retry queues.
  2. **Flat Output:** Generates unstructured text summaries. It lacks relationship mapping between conflicting sources and fails to represent claim dependencies.
  3. **No Security Boundary:** Scraped web snippets are piped directly into LLM prompts without sanitization, leaving the system completely open to prompt injection attacks.
* **ProofGraph Solution:**
  - Enforce schema validation using Pydantic models with constrained decoding (XGrammar / Outlines).
  - Model claims and evidence as an interconnected graph with first-class contradiction edges.
  - Implement the **Dual-LLM Security Pattern**: A Quarantined LLM extracts facts from untrusted web pages, while a Privileged LLM reasons over the graph.

---

## 3. The Iterative Search Pioneer: FIRE (NAACL 2025)
* **Repository:** [mbzuai-nlp/fire](https://github.com/mbzuai-nlp/fire)
* **Architecture:** Agentic fact-checking loop replacing fixed-N retrieval with a unified decision module that dynamically chooses between outputting a final verdict or emitting the next search query.
* **Key Strengths:**
  - **7.6x LLM computational cost reduction** and **16.5x search API cost reduction** compared to conventional static frameworks (FacTool, Factcheck-GPT).
  - Demonstrates that lightweight models (GPT-4o-mini) perform within ~1% of large reasoning models (o1-preview) on fact-checking when paired with good retrieval, offering a 766x cost advantage.
* **Fatal Flaws & Shortcomings:**
  1. **No Source Credibility Layer:** Treats all retrieved web domains equally.
  2. **No Graph Structure:** Produces only scalar veracity predictions without evidence provenance graphs.
  3. **Query Drift:** Can fall into repetitive loops without strict diversity prompts.
* **ProofGraph Solution:** Adopt FIRE's dynamic early stopping and iterative query generation policy, but embed it inside a graph-aware orchestrator that evaluates source credibility and stores evidence provenance in W3C PROV-DM format.

---

## 4. The Enterprise Full-Stack App: Veracity (IJCAI 2025)
* **Paper:** IJCAI 2025 Demonstration
* **Architecture:** Next.js frontend, FastAPI backend, PostgreSQL database, hosted on GCP/Kubernetes. Features an expert dashboard and user feedback tracking.
* **Key Strengths:** Production-ready web architecture and a curated media credibility database.
* **Fatal Flaws & Shortcomings:**
  - Coarse document-level checks rather than atomic claim decomposition.
  - No handling of prompt injection in retrieved content.
  - Lacks fine-grained multi-hop evidence reasoning.
* **ProofGraph Solution:** Combine Veracity's clean full-stack architectural design with atomic claim decomposition (Claimify) and interactive Cytoscape/NetworkX knowledge graph visualization.

---

## 5. Summary Matrix: Where ProofGraph Wins

| Feature / Capability | TRUSTGRAPH | LOKI | FIRE | Veracity | ProofGraph |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real Hybrid Retrieval (Dense+BM25)** | ❌ (Mock) | ⚠️ (Serper only) | ⚠️ (Google only) | ⚠️ (Search API) | ✅ (BM25 + Dense + ColBERT) |
| **Iterative Search & Early Stopping** | ❌ | ❌ | ✅ | ❌ | ✅ (FIRE-style policy) |
| **Atomic Claim Decomposition** | ❌ (Regex) | ⚠️ (Single-step) | ⚠️ (Sentence) | ❌ (Document) | ✅ (Claimify 4-Stage) |
| **Reference Indeterminacy Gate** | ❌ | ❌ | ❌ | ❌ | ✅ (RefNLI Overlap Filter) |
| **Source Credibility Weighting** | ⚠️ (Fake) | ❌ | ❌ | ✅ (Domain score) | ✅ (CONFACT SBA-CoT) |
| **W3C PROV-DM Provenance Graph** | ❌ (Star mock)| ❌ (Flat text) | ❌ (Scalar) | ❌ (Relational) | ✅ (PROV-DM Standard) |
| **First-Class Contradiction Edges** | ❌ | ❌ | ❌ | ❌ | ✅ (Typed Disagreement Links) |
| **Prompt Injection Quarantine** | ❌ | ❌ | ❌ | ❌ | ✅ (Dual-LLM Security) |
