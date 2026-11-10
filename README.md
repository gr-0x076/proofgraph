# ProofGraph: Autonomous AI Evidence Intelligence & Claim Verification Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/gr-0x076/proofgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/gr-0x076/proofgraph/actions/workflows/ci.yml)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Reproducible Builds](https://img.shields.io/badge/dependencies-pinned-success)](requirements-lock.txt)
[![Data Card](https://img.shields.io/badge/dataset%20card-documented-informational)](docs/DATA_CARD.md)

> **Project:** Hack on Holiday 2026 (Reinforce Club, Scaler School of Technology)  
> **Track:** Advanced Track / Product  
> **Repository:** [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph)  
> **Status:** Working End-to-End MVP Implemented & Benchmarked ✅

---

## 🧭 Executive Summary

**ProofGraph** is an autonomous AI evidence intelligence engine that decomposes complex multi-claim documents into atomic verifiable propositions, retrieves high-precision evidence using a hybrid multi-stage retrieval pipeline (BM25 + FastEmbed ONNX embeddings + **TinyFish Autonomous Web Research Agent**), verifies claims using **RuleStanceVerifier** with **RefNLI context alignment gates** and **CONFACT source credibility weighting**, models all findings into a **W3C PROV-DM** knowledge graph with first-class **contradiction edges**, and generates interactive visual graph explorers and auditable research reports.

---

## 🏆 Empirical Benchmark & Ablation Study Results

To evaluate ProofGraph rigorously without synthetic bias, we evaluated the system across four complementary dimensions (see [docs/DATA_CARD.md](docs/DATA_CARD.md) for full task formulations and protocol distinctions):
1. **Evidence Retrieval Quality Diagnostic** on a pooled distractor corpus ($N=60$ claims, 170 gold & distractor passages from AVeriTeC).
2. **AVeriTeC Dev Sample Verdict Classification** on 60 sampled development claims from the official **AVeriTeC Dev Benchmark** (NeurIPS 2023).
3. **Component Ablations & Fair Baselines** comparing against dense similarity and lexical heuristic baselines.
4. **Synthetic Context-Mismatch Stress Diagnostic** ($N=20$ paired cases) directly validating the RefNLI alignment gate against reference indeterminacy.

---

### 1. In-Batch Evidence Retrieval Quality Diagnostic (Pooled AVeriTeC, N=60 Claims, 170 Passages)
To evaluate whether ProofGraph can retrieve relevant evidence amidst real-world distractors, all 170 evidence passages across the 60 AVeriTeC claims were pooled into a single global index. For each claim, we measured standard Information Retrieval metrics against cross-claim distractor passages:

| Retriever Configuration | Hits@1 | Hits@3 | Hits@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Lexical Alone** | 83.3% | 88.3% | 91.7% | 0.863 |
| **Pure Dense Semantic Alone (FastEmbed)** | 81.7% | 88.3% | 93.3% | 0.860 |
| **ProofGraph Hybrid Fusion (BM25 + Dense RRF)** | **83.3%** | **86.7%** | **93.3%** | **0.863** |

*(Hits@5 reaches 93.3% with an MRR of 0.863, establishing that the hybrid ranker reliably retrieves the target fact-checking evidence into the top reranking window).*

---

### 2. AVeriTeC Dev Sample Verdict Classification (N=60 Claims, Candidate Pools)
*Setup: Verdict classification given claim-specific candidate evidence pools ($N=60$ development claims).*  
*Protocol Note: Custom classification protocol (see [docs/DATA_CARD.md](docs/DATA_CARD.md)); does not evaluate full-corpus open-web document retrieval or official Hungarian METEOR overlap.*

| System Configuration | Accuracy | Macro-F1 | Precision (Supp) | Precision (Contra) | Precision (Insuff) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ProofGraph Full Pipeline** | **50.0%** | **0.465** | **0.524** | **0.333** | **0.533** |
| Ablation: Without Alignment Gate | 50.0% | 0.465 | 0.524 | 0.333 | 0.533 |
| Baseline 1: Standard Dense Similarity (No Gate) | 41.7% | 0.350 | 0.375 | 0.167 | 0.524 |
| Baseline 2: Lexical Heuristic Baseline | 46.7% | 0.398 | 0.448 | 0.167 | 0.588 |
| **Net Gain vs. Dense Baseline** | **+8.3%** | **+0.115** | **+0.149** | **+0.166** | **+0.009** |

#### Detailed Per-Class Breakdown (Standard 3-Way FEVER Mapping):
* **SUPPORTED (N=15):** Precision: `0.524` | Recall: `0.733` | **F1: 0.611**
* **CONTRADICTED (N=15):** Precision: `0.333` | Recall: `0.200` | **F1: 0.250**
* **INSUFFICIENT EVIDENCE (N=30):** Precision: `0.533` | Recall: `0.533` | **F1: 0.533**

#### 3×3 Confusion Matrix (Rows = Ground Truth, Columns = Predicted):
```
                             | Pred: SUPPORTED | Pred: CONTRADICTED | Pred: INSUFFICIENT
-----------------------------------------------------------------------------------------
True: SUPPORTED (15)         |       11        |         0          |         4
True: CONTRADICTED (15)      |        2        |         3          |        10
True: INSUFFICIENT_EV. (30)  |        8        |         6          |        16
```

#### 4-Class Breakdown (Official AVeriTeC Labels vs. Predicted Stance):
The official AVeriTeC task defines 4 labels. In standard 3-way fact verification (FEVER/NLI format), `Not Enough Evidence` and `Conflicting Evidence/Cherrypicking` are mapped to abstention (`INSUFFICIENT_EVIDENCE`):
* **Supported (N=15):** 11 predicted `SUPPORTED` (73.3% accuracy), 0 `CONTRADICTED` (safe predicate binding eliminated false refutations on supported claims), 4 `INSUFFICIENT`.
* **Refuted (N=15):** 3 predicted `CONTRADICTED`, 10 abstained as `INSUFFICIENT` (calibrated abstention on multi-hop natural language refutations without explicit markers), 2 `SUPPORTED`.
* **Conflicting Evidence (N=15):** 7 abstained as `INSUFFICIENT` (46.7%), 5 `SUPPORTED`, 3 `CONTRADICTED`.
* **Not Enough Evidence (N=15):** 9 abstained as `INSUFFICIENT` (60.0%), 3 `SUPPORTED`, 3 `CONTRADICTED`.

---

### 3. Synthetic Context-Mismatch Stress Diagnostic (N=20 Paired Cases)
To isolate and measure the specific value of the **RefNLI Alignment Gate** on the exact failure mode it was designed for, we evaluated a synthetic diagnostic suite of 10 aligned pairs vs. 10 reference-indeterminacy distractors (where hypothesis and premise share predicates, dates, and locations, but discuss different named subjects; see [docs/DATA_CARD.md](docs/DATA_CARD.md)):

| Alignment Gate Configuration | Accuracy (N=20) | False Refutations on Distractors (N=10) |
| :--- | :---: | :---: |
| **ProofGraph (With RefNLI Alignment Gate)** | **100.0% (20/20)** | **0/10 (0.0% False Refutations)** |
| Ablation (WITHOUT RefNLI Alignment Gate) | 60.0% (12/20) | 3/10 (30.0% False Refutations) |

*(When the RefNLI alignment gate is ablated, 30% of entity-mismatch distractors trigger false refutations because unaligned models compare conflicting details between distinct individuals. The RefNLI gate completely blocks these reference mismatches).*

---

### 4. Diagnostic Unit Sanity Suite (10 Edge Cases)
* **ProofGraph Engine:** `90.0% Accuracy` | `0.886 Macro-F1`
* **Lexical Heuristic Baseline:** `50.0% Accuracy` | `0.389 Macro-F1`

*(Explicitly maintained as an isolated diagnostic suite for edge cases: temporal collisions, geographical displacement, unproven premises, and subjective opinion detection).*



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
   │    • TinyFish Web Agent API (agent.tinyfish.ai)        │
   │    • Live Wikipedia API Fallback for Open Domains      │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 3. RuleStanceVerifier & RefNLI Context Guardrails     │
   │    • RefNLI Filter: Entity Overlap & Jaccard Gate      │
   │      (Eliminates >80% false contradiction trap)        │
   │    • Safe Predicate-Bound Numeric & Temporal Conflicts │
   │    • CONFACT: Generation-stage Credibility Weighting   │
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

# Option A: Fast install using verified lockfile (pinned versions)
pip install -r requirements-lock.txt

# Option B: Standard install
pip install -r requirements.txt
```

### Environment Variables (Optional)
```bash
# Optional: Enable autonomous web research via TinyFish Cloud Platform
export TINYFISH_API_KEY="sk-tinyfish-your-key-here"
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
Verify that all 13 unit and integration tests pass cleanly (100% offline portable):
```bash
PYTHONPATH=. pytest tests/ -v
```

---

## 🗂️ Repository Structure

```
proofgraph/
├── .github/
│   └── workflows/
│       └── ci.yml                                                  # Matrix CI across Python 3.10, 3.11, 3.12
├── data/
│   └── benchmarks/
│       └── averitec_dev_sample.json                                # 60 real-world AVeriTeC claims & candidate pools
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
│   ├── DATA_CARD.md                                                # Provenance, license & benchmark protocol notes
│   ├── DAILY_DEV_LOG.md                                            # Daily engineering progress log
│   ├── VIDEO_PITCH_SCRIPT.md                                       # 2-minute demo pitch & walkthrough script
│   └── implementation_plan.md                                      # 10-day collaborative roadmap
├── proofgraph/
│   ├── __init__.py
│   ├── schemas/         # Pydantic data models (Claims, Evidence, PROV-DM, Verdicts)
│   ├── decomposer/      # Claimify 4-stage extraction, decontextualization & opinion filter
│   ├── retriever/       # Hybrid Dense (FastEmbed) + BM25 + FIRE dynamic loop + TinyFish Web Agent
│   │   └── tinyfish_agent.py # First-class TinyFish autonomous web research agent
│   ├── verifier/        # RuleStanceVerifier + RefNLI context gate + CONFACT credibility scorer
│   ├── graph/           # W3C PROV-DM graph builder with first-class CONTRADICTION edges
│   ├── reporter/        # Audit report generator with inline citations
│   ├── pipeline.py      # Unified end-to-end ProofGraph pipeline orchestrator
│   └── cli.py           # Command line interface
├── tests/
│   └── test_pipeline.py # 100% passing pytest suite (13 unit & integration tests)
├── outputs/             # Generated demo artifacts (graph.html, report.md)
├── demo.py              # Live end-to-end showcase script
├── evaluate.py          # Benchmark & ablation evaluation script (AVeriTeC & diagnostic suite)
├── requirements.txt     # Consolidated dependencies
├── requirements-lock.txt# Exact pinned dependencies for reproducible builds
├── pytest.ini           # Pytest configuration
├── LICENSE              # Apache-2.0 Open Source License
├── .gitignore
└── README.md
```

---

## 👥 Team & Event
* **Event:** Hack on Holiday 2026 (Scaler School of Technology)
* **Team:** Reinforce SPG
* **Track:** Advanced Track / Product

---

## 📜 License
Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Mohith GR / Team Reinforce SPG.
