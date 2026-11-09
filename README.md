# ProofGraph: Autonomous AI Evidence Intelligence & Claim Verification Engine

> **Project:** Hack on Holiday Hackathon (Reinforce Club, Scaler School of Technology)  
> **Track:** Advanced Track / Product  
> **Repository:** [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph)  
> **Status:** Phase 1 Complete — Research & Scaffolding Published

---

## 🧭 Executive Summary

**ProofGraph** is an autonomous fact-checking and evidence intelligence engine that decomposes complex multi-claim documents into verifiable propositions, retrieves high-precision evidence using a hybrid multi-stage retrieval pipeline, rigorously verifies claims while guarding against reference mismatch and source bias, models all findings into a **W3C PROV-DM** knowledge graph with first-class **contradiction edges**, and generates auditable research reports.

This repository contains the complete 20-page research landscape, benchmark deep dives, prior art teardowns, architectural specifications, and the modular codebase skeleton.

---

## 🏛️ System Architecture

```
User Document / Research Claim
              │
              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 1. Atomic Claim Extraction & Decontextualization      │
   │    • 4-Stage Claimify Pipeline (ACL 2025)              │
   │    • Context Brackets `[...]` & Ambiguity Abstention   │
   │    • Rule: N_subclaims <= N_sentences (Hu et al. 2024) │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 2. Multi-Stage Hybrid Retrieval & Dynamic Stopping    │
   │    • BM25 Lexical + Dense Semantic (all-MiniLM-L6-v2)  │
   │    • ColBERT Token MaxSim / Reciprocal Rank Fusion     │
   │    • FIRE Agent (NAACL 2025): 7.6x LLM / 16.5x Search   │
   │      Cost Savings via Early Stopping & Diversity Query │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 3. NLI Verification & Context Guardrails              │
   │    • RefNLI Filter: Entity & Jaccard > 0.15 overlap    │
   │      (Eliminates >80% false contradiction trap)        │
   │    • CONFACT (2025): Source Credibility Augmentation  │
   │      at generation stage (SBA-CoT) (+10% F1 gain)      │
   │    • Calibrated Abstention (Unverifiable / NEI)        │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 4. W3C PROV-DM Graph with Contradiction Edges         │
   │    • Entities (Claims, Evidence Passages, Sources)     │
   │    • Activities (Decompose, Retrieve, Verify)          │
   │    • First-class CONTRADICTS & DISAGREES_WITH edges    │
   │    • JSON-LD / NetworkX / PyVis Interactive Explorer   │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 5. Audit Reports & Dual-LLM Security Boundary         │
   │    • Quarantined LLM for untrusted web snippets        │
   │    • Markdown / HTML report with provenance backlinks  │
   └────────────────────────────────────────────────────────┘
```

---

## 🔬 Forensic Research & Prior Art Teardown

Before writing a single line of code, we thoroughly audited existing implementations and academic benchmarks. All findings are documented in [`docs/research/`](docs/research/):

1. **[01. Complete 20-Page Evidence Landscape](docs/research/01_executive_landscape.txt):** The exhaustive 83-source research report covering fact-checking surveys, LLM hallucinations, prompt injection defenses (OWASP LLM01:2025), and retrieval bottlenecks.
   - Raw PDF available at: [`docs/assets/ProofGraph-An-Exhaustive-Evidence-Linked-Landscape.pdf`](docs/assets/ProofGraph-An-Exhaustive-Evidence-Linked-Landscape.pdf)
2. **[02. Prior Art Analysis & Critique](docs/research/02_prior_art_critique.md):**
   - **TRUSTGRAPH (GitHub):** A 100% mock demo (hardcoded 6 sources, regex split, zero retrieval, star graph). ProofGraph replaces this with real hybrid retrieval and dynamic PROV-DM graphs.
   - **LOKI (COLING 2025):** 5-step pipeline with brittle open-LLM JSON formatting and no security boundary. ProofGraph adds Pydantic schema validation and the Dual-LLM pattern.
   - **FIRE (NAACL 2025):** Demonstrates **7.6x LLM cost and 16.5x search cost reduction** via a unified stopping decision module.
   - **Veracity (IJCAI 2025):** Clean full-stack Next.js + FastAPI + PostgreSQL reference architecture.
3. **[03. Benchmarks, Datasets & Metrics](docs/research/03_benchmarks_and_datasets.md):**
   - **FEVER:** 185k claims, lexical mutation bias (~62% claim-only accuracy without evidence).
   - **AVeriTeC (NeurIPS 2023 / FEVER 8 2025):** Real-world misinformation with the **Ev2R recall metric (threshold >= 0.44)**.
   - **SciFact:** Scientific claim verification with claim-document pair evaluations.
   - **CONFACT (2025):** Evaluates conflicting evidence. Shows human raters trusted 69.8% of misleading sources, and vanilla RAG fails majority-vote heuristics.
   - **Decomposition Dilemmas (arXiv:2411.02400):** Over-fragmenting claims hurts strong verifiers ($E_d + E_r > \Delta A$).
4. **[04. Core Technical Patterns](docs/research/04_technical_patterns.md):**
   - Qdrant hybrid retrieval + ColBERT MaxSim late interaction.
   - Claimify 4-stage proposition extraction with bracketed context.
   - RefNLI reference indeterminacy mitigation.
   - W3C PROV-DM ontology and contradiction edge definitions.
5. **[05. Hack on Holiday Event Specifications](docs/research/05_hackathon_brief_and_specs.md):**
   - Complete rules, rubrics, and the 11 official Idea Jar problem statements ([JSON format](docs/research/all_problem_statements.json)).

---

## 🗂️ Repository Structure

```
proofgraph/
├── docs/
│   ├── assets/
│   │   └── ProofGraph-An-Exhaustive-Evidence-Linked-Landscape.pdf
│   ├── research/
│   │   ├── 01_executive_landscape.txt
│   │   ├── 02_prior_art_critique.md
│   │   ├── 03_benchmarks_and_datasets.md
│   │   ├── 04_technical_patterns.md
│   │   ├── 05_hackathon_brief_and_specs.md
│   │   └── all_problem_statements.json
│   └── implementation_plan.md
├── proofgraph/
│   ├── __init__.py
│   ├── schemas/         # Pydantic data models (Claims, Evidence, PROV-DM, Verdicts)
│   ├── decomposer/      # Claimify 4-stage extraction & decontextualization
│   ├── retriever/       # Hybrid Dense + BM25 + FIRE dynamic stopping loop
│   ├── verifier/        # 3-way NLI verifier + RefNLI filter + CONFACT credibility
│   ├── graph/           # W3C PROV-DM knowledge graph builder with contradiction links
│   └── reporter/        # Audit report generator & Cytoscape/PyVis visualizer
├── tests/               # Unit and integration test suites
├── configs/             # Configuration files
├── requirements.txt     # Python package dependencies
├── .gitignore
└── README.md
```

---

## 🗓️ 10-Day Collaborative Roadmap

- [x] **Milestone 1 (Days 1–2): Research Synthesis & Scaffolding**
  - Literature audit, prior art teardown, architectural specifications, repository scaffolding.
- [ ] **Milestone 2 (Days 3–4): Atomic Claim Decomposition Engine**
  - Claimify 4-stage extractor, ambiguity detector, decontextualization with context brackets.
- [ ] **Milestone 3 (Days 5–6): Multi-Stage Hybrid Retrieval & FIRE Agent**
  - Dense + BM25 hybrid indexing, ColBERT token MaxSim, FIRE iterative search loop with dynamic stopping.
- [ ] **Milestone 4 (Day 7): Verification, NLI & Source Credibility Layer**
  - RefNLI alignment gate (Jaccard > 0.15), 3-way NLI verifier, CONFACT generation-stage credibility injection.
- [ ] **Milestone 5 (Day 8): W3C PROV-DM Graph & Contradiction Edges**
  - Entities, Activities, Agents graph modeling, first-class contradiction edge resolution.
- [ ] **Milestone 6 (Day 9): Interactive Visualizer & Audit Report Generator**
  - Interactive graph UI (NetworkX / PyVis / Cytoscape), Dual-LLM prompt injection quarantine.
- [ ] **Milestone 7 (Day 10): Ablation Benchmarking & Final Submission**
  - Comparative benchmark against Standard RAG baseline on AVeriTeC / CONFACT, video demo, and finale slides.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.7)
- Recommended: CUDA-capable GPU (NVIDIA RTX 3050+ with >= 4GB VRAM) for local embeddings

### Setup
```bash
git clone https://github.com/gr-0x076/proofgraph.git
cd proofgraph

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 👥 Team & Collaboration
* **Event:** Hack on Holiday 2026 (Scaler School of Technology)
* **Team:** Reinforce SPG
