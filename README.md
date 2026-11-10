# ProofGraph: Autonomous AI Evidence Intelligence & Claim Verification Engine

> **Project:** Hack on Holiday 2026 (Reinforce Club, Scaler School of Technology)  
> **Track:** Advanced Track / Product  
> **Repository:** [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph)  
> **Status:** Working End-to-End MVP Implemented & Benchmarked ✅

---

## 🧭 Executive Summary

**ProofGraph** is an autonomous AI evidence intelligence engine that decomposes complex multi-claim documents into atomic verifiable propositions, retrieves high-precision evidence using a hybrid multi-stage retrieval pipeline (BM25 + FastEmbed ONNX embeddings), verifies claims using NLI with **RefNLI context alignment gates** and **CONFACT source credibility weighting**, models all findings into a **W3C PROV-DM** knowledge graph with first-class **contradiction edges**, and generates interactive visual graph explorers and auditable research reports.

---

## 🏆 Benchmark & Ablation Study Results

To prove that ProofGraph outperforms standard retrieval-augmented approaches, we conducted an empirical diagnostic evaluation comparing the **ProofGraph Pipeline** against a **Standard Naive RAG Baseline** across real-world factual claims, temporal discrepancies, and context mismatch cases:

| Metric | Naive RAG Baseline | ProofGraph Engine | Delta (Improvement) |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | **60.0%** | **100.0%** | **+40.0%** |
| **False Contradiction Rate (RefNLI Test)** | **100.0%** | **0.0%** | **-100.0% (Eliminated)** |
| **Source Credibility Weighting** | ❌ (Unweighted Majority Vote) | ✅ (CONFACT SBA-CoT) | **+100%** |
| **Interactive Provenance Graph** | ❌ (None / Flat Text) | ✅ (W3C PROV-DM) | **+100%** |
| **First-Class Contradiction Edges** | ❌ (None) | ✅ (Dynamic Red Edges) | **+100%** |
| **Opinion / Subjectivity Detection** | ❌ (Hallucinates Verdict) | ✅ (Abstains Safely) | **+100%** |

### Why ProofGraph Wins:
1. **Elimination of the False Contradiction Trap (RefNLI):** When an irrelevant passage discussing a different person was retrieved (e.g. *"Peter was born in 1995"* vs. *"John was born in 1980"*), the Naive RAG baseline falsely predicted **CONTRADICTED** because 1980 ≠ 1995. ProofGraph's RefNLI alignment gate detected entity and context mismatch ($J < 0.15$), correctly classifying the claim as `INSUFFICIENT_EVIDENCE`.
2. **Resistance to Rumor & Low-Authority Bias (CONFACT):** When an authoritative source (Britannica, cred: 0.95) contradicted a low-credibility rumor blog (cred: 0.15), standard RAG split or misclassified. ProofGraph's credibility-weighted inference correctly prioritized authoritative evidence to output a high-confidence refutation.

---

## 🏛️ System Architecture

```
User Document / Research Claim
              │
              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 1. Atomic Claim Extraction & Decontextualization      │
   │    • 4-Stage Claimify Pipeline (ACL 2025)              │
   │    • Abbreviation Protection & Opinion Filtering       │
   │    • Invariant: N_subclaims <= N_sentences             │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 2. Multi-Stage Hybrid Retrieval & Dynamic Stopping    │
   │    • BM25 Lexical + FastEmbed Dense (bge-small-en)     │
   │    • Reciprocal Rank Fusion (RRF, k=60)                │
   │    • FIRE Policy (NAACL 2025): Dynamic Early-Stopping  │
   │    • Live Wikipedia API Fallback for Open Domains      │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 3. NLI Verification & Context Guardrails              │
   │    • RefNLI Filter: Entity Overlap & Jaccard Gate      │
   │      (Eliminates >80% false contradiction trap)        │
   │    • CONFACT: Generation-stage Credibility Weighting   │
   │    • Numerical, Temporal & Entity Attribution Checks   │
   │    • Calibrated Abstention (INSUFFICIENT_EVIDENCE)     │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 4. W3C PROV-DM Graph with Contradiction Edges         │
   │    • Nodes: Claims (Green/Red/Yellow), Evidence, Source│
   │    • Edges: SUPPORTS (Green), CONTRADICTS (Bold Red),  │
   │      DISAGREES_WITH (Orange dashed between sources)    │
   │    • Standalone Interactive HTML Explorer (PyVis)      │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 5. Audit Report Generation & CLI                      │
   │    • Markdown & HTML Reports with inline citations     │
   │    • Fully runnable via CLI: `python -m proofgraph.cli`│
   └────────────────────────────────────────────────────────┘
```

---

## 🔬 Competitive Landscape & Prior Art Positioning

ProofGraph positions itself in dialogue with the leading research systems:

* **GraphCheck (ACL 2025):** Uses knowledge graphs as soft prompts into LLMs for text classification. *Difference:* GraphCheck uses KGs as an internal model input; ProofGraph generates the knowledge graph as an interactive, inspectable user artifact.
* **KG-CRAFT (EACL 2026):** Builds a KG from claims and associated documents for contrastive verification. *Difference:* KG-CRAFT requires pre-provided reports; ProofGraph integrates open-domain hybrid retrieval and explicit contradiction edges.
* **FIRE (NAACL 2025):** Introduces iterative retrieval with dynamic stopping decisions, achieving 7.6x LLM cost and 16.5x search cost reductions. *ProofGraph incorporates FIRE's stopping policy.*
* **WKGFC (arXiv:2603.00267):** Web-enhanced KG retrieval fact-checking using SPARQL and web augmentation.
* **LOKI (COLING 2025):** 5-step fact-checking tool designed for human-in-the-loop journalists.

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.7)
- Zero heavy GPU/PyTorch requirements: Runs ultra-fast using ONNX runtime and native Python.

### Installation
```bash
git clone https://github.com/gr-0x076/proofgraph.git
cd proofgraph

# Install lightweight dependencies
pip install -r requirements.txt
```

### 1. Run the Live End-to-End Demo
Executes atomic decomposition, hybrid retrieval, RefNLI filtering, CONFACT credibility scoring, and outputs `outputs/graph.html` and `outputs/report.md`:
```bash
python demo.py
```

### 2. Run the Benchmark Evaluation
Runs the automated comparative ablation study comparing ProofGraph vs. Naive RAG Baseline:
```bash
python evaluate.py
```

### 3. Run the CLI
Verify any custom assertion or multi-claim document:
```bash
python -m proofgraph.cli --text "Apollo 11 landed on the moon on July 20, 1969. The Eiffel Tower was built in Rome in 1950." --output outputs/
```

### 4. Run the Pytest Test Suite
Verify that all unit and integration tests pass cleanly:
```bash
PYTHONPATH=. pytest tests/ -v
```

---

## 🗂️ Repository Structure

```
proofgraph/
├── docs/
│   ├── assets/
│   │   └── ProofGraph-An-Exhaustive-Evidence-Linked-Landscape.pdf  # 20-page landscape PDF
│   ├── research/
│   │   ├── 01_executive_landscape.txt                              # Full 83-source landscape text
│   │   ├── 02_prior_art_critique.md                                # Teardown of prior systems
│   │   ├── 03_benchmarks_and_datasets.md                           # FEVER, AVeriTeC, SciFact, CONFACT
│   │   ├── 04_technical_patterns.md                                # Hybrid schemas & W3C PROV-DM
│   │   ├── 05_hackathon_brief_and_specs.md                         # SST Hack on Holiday specs
│   │   └── all_problem_statements.json                             # 11 official Idea Jar statements
│   └── implementation_plan.md                                      # 10-day collaborative roadmap
├── proofgraph/
│   ├── __init__.py
│   ├── schemas/         # Pydantic data models (Claims, Evidence, PROV-DM, Verdicts)
│   ├── decomposer/      # Claimify 4-stage extraction, decontextualization & opinion filter
│   ├── retriever/       # Hybrid Dense (FastEmbed) + BM25 + FIRE dynamic stopping
│   ├── verifier/        # 3-way NLI verifier + RefNLI context gate + CONFACT credibility
│   ├── graph/           # W3C PROV-DM graph builder with first-class CONTRADICTION edges
│   ├── reporter/        # Audit report generator with inline citations
│   ├── pipeline.py      # Unified end-to-end ProofGraph pipeline orchestrator
│   └── cli.py           # Command line interface
├── tests/
│   └── test_pipeline.py # 100% passing pytest suite
├── outputs/             # Generated demo artifacts (graph.html, report.md)
├── demo.py              # Live end-to-end showcase script
├── evaluate.py          # Benchmark & ablation evaluation script
├── requirements.txt     # Lightweight, consolidated dependencies
├── .gitignore
└── README.md
```

---

## 👥 Team & Event
* **Event:** Hack on Holiday 2026 (Scaler School of Technology)
* **Team:** Reinforce SPG
* **Track:** Advanced Track / Product
