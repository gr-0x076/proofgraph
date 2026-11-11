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

**ProofGraph** is a research prototype for evidence-backed claim review. It decomposes text with a heuristic claim splitter, retrieves from a local corpus using BM25 and FastEmbed ONNX embeddings, and can search the web with TinyFish Search and fetch candidate pages with TinyFish Fetch. Its current stance verifier is rule-based, its alignment and source-prior components are inspired by published work rather than reproductions of those systems, and its NetworkX graph is PROV-DM-inspired rather than certified as standards-conformant. The goal is to make evidence and source provenance inspectable, while abstaining when its rules cannot justify a verdict.

---

## 🏆 Empirical Benchmark & Ablation Study Results

The percentages emitted by the rule verifier are uncalibrated heuristic scores, not probabilities of correctness. No calibration study has been run.

These experiments are exploratory diagnostics, not a comprehensive or unbiased evaluation. Results below include a small sampled AVeriTeC subset and a hand-authored synthetic stress suite; see [docs/DATA_CARD.md](docs/DATA_CARD.md) for task formulations and limitations:
1. **Evidence Retrieval Quality Diagnostic** on a pooled distractor corpus ($N=60$ claims, 170 gold & distractor passages from AVeriTeC).
2. **AVeriTeC Dev Sample Verdict Classification** on 60 sampled development claims from the official **AVeriTeC Dev Benchmark** (NeurIPS 2023).
3. **Component Ablations & Fair Baselines** comparing against dense similarity and lexical heuristic baselines.
4. **Synthetic Context-Mismatch Stress Diagnostic** ($N=20$ paired cases) testing a RefNLI-inspired alignment heuristic on hand-authored examples.

---

### 1. In-Batch Evidence Retrieval Quality Diagnostic (Pooled AVeriTeC, N=60 Claims, 170 Passages)
To evaluate whether ProofGraph can retrieve relevant evidence amidst real-world distractors, all 170 evidence passages across the 60 AVeriTeC claims were pooled into a single global index. For each claim, we measured standard Information Retrieval metrics against cross-claim distractor passages:

| Retriever Configuration | Hits@1 | Hits@3 | Hits@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Lexical Alone** | 83.3% | 88.3% | 91.7% | 0.863 |
| **Pure Dense Semantic Alone (FastEmbed)** | 81.7% | 88.3% | 93.3% | 0.860 |
| **ProofGraph Hybrid Fusion (BM25 + Dense RRF)** | **83.3%** | **86.7%*** | **93.3%** | **0.863** |

*\*Retrieval Trade-Off Note: On this small pooled diagnostic, hybrid RRF ties BM25 on Hits@1 (83.3%) and MRR (0.863) and ties dense retrieval on Hits@5 (93.3%). Hits@3 is slightly lower (86.7% vs. 88.3%) due to rank reciprocal dampening ($k=60$) when lexical and dense ranks diverge, but hybrid fusion avoids dense semantic drift on out-of-vocabulary named entities.*

---

### 2. AVeriTeC Development Benchmark (Oracle Candidate Pools)

*Methodology Note: To eliminate hyperparameter snooping and test-set leakage, the 60 AVeriTeC claims are partitioned into a balanced **40-Claim Dev Set** (10 per class) and an untouched **20-Claim Held-Out Test Set** (5 per class). Hyperparameter threshold sweeping is performed exclusively on Dev; the chosen threshold is frozen and evaluated once on the untouched Test set. All metrics include 1,000-resample non-parametric bootstrap 95% confidence intervals.*

#### Step 1: Hyperparameter Fallback Threshold Sweep on 40 Dev Claims
Thresholds swept exclusively on the 40 development claims (zero test-set leakage):

| Fallback Threshold | Dev Accuracy | Dev Macro-F1 | Dev Contra Recall | Selection Status |
| :---: | :---: | :---: | :---: | :--- |
| `0.50` | 42.5% | 0.423 | 0.300 | Evaluated |
| `0.60` | 42.5% | 0.423 | 0.300 | Evaluated |
| `0.70` | 42.5% | 0.423 | 0.300 | Evaluated |
| `0.75` | 42.5% | 0.423 | 0.300 | Evaluated |
| `0.80` | 42.5% | 0.423 | 0.300 | Evaluated |
| `0.85` | 42.5% | 0.423 | 0.300 | Evaluated |
| `0.90` | 47.5% | 0.463 | 0.300 | Evaluated |
| `0.95` | **50.0%** | **0.483** | **0.300** | **⭐ Selected Optimal Threshold** |

*Threshold `0.95` was frozen based on Dev Macro-F1 (0.483) and applied to the held-out test set.*

#### Step 2: Independent Evaluation on 20 Held-Out Test Claims (Untouched During Tuning)

| System Configuration (N=20 Test Claims) | Accuracy (95% Bootstrap CI) | Macro-F1 (95% Bootstrap CI) | Prec (Contra) | Rec (Contra) |
| :--- | :---: | :---: | :---: | :---: |
| **ProofGraph Hybrid (Frozen Threshold=0.95)** | **55.0%** [35%–75%] | **0.508** [0.29–0.72] | **0.500** | **0.200** |
| ProofGraph Rules Alone | 55.0% [35%–75%] | 0.508 [0.29–0.72] | 0.500 | 0.200 |
| ProofGraph Neural Alone | 60.0% [40%–80%] | 0.516 [0.24–0.72] | 1.000 | 0.400 |
| Ablation: Without Alignment Gate | 55.0% [35%–75%] | 0.508 [0.29–0.72] | 0.500 | 0.200 |
| Baseline 1: Standard Dense Similarity (No Gate) | 60.0% [40%–80%] | 0.539 [0.32–0.76] | 0.500 | 0.200 |
| Baseline 2: Lexical Heuristic Baseline | 55.0% [35%–75%] | 0.507 [0.30–0.70] | 0.333 | 0.200 |

#### Step 3: Full 60-Claim Descriptive Sample Benchmark (With Bootstrap CIs)

| System Configuration (N=60 Total Sample) | Accuracy (95% Bootstrap CI) | Macro-F1 (95% Bootstrap CI) | Prec (Contra) | Rec (Contra) |
| :--- | :---: | :---: | :---: | :---: |
| **ProofGraph Hybrid (Frozen Threshold=0.95)** | **51.7%** [38%–65%] | **0.493** [0.35–0.63] | **0.444** | **0.267** |
| ProofGraph Rules Alone | 51.7% [38%–65%] | 0.493 [0.35–0.63] | 0.444 | 0.267 |
| ProofGraph Neural Alone | 48.3% [37%–62%] | 0.343 [0.23–0.47] | 0.250 | 0.133 |
| Ablation: Without Alignment Gate | 51.7% [38%–65%] | 0.493 [0.35–0.63] | 0.444 | 0.267 |
| Baseline 1: Standard Dense Similarity (No Gate) | 41.7% [30%–53%] | 0.350 [0.24–0.45] | 0.167 | 0.067 |
| Baseline 2: Lexical Heuristic Baseline | 46.7% [35%–60%] | 0.398 [0.30–0.51] | 0.167 | 0.067 |
| **Hybrid vs. Dense Baseline** | **+10.0 pts** | **+0.143** | **+0.277** | **+0.200 (4.0×)** |

> **Key Methodological Disclosures:**
> 1. **Held-Out Procedure:** The 0.95 threshold was selected exclusively on the 40-claim development split and verified on the 20-claim held-out test split, eliminating hyperparameter snooping.
> 2. **Confidence Intervals:** Due to sample sizes ($N=20$ and $N=60$), 95% bootstrap confidence intervals span approximately $\pm 12$ to $\pm 20$ percentage points. Differences between hybrid and rule configurations remain within sampling variance; transparency is maintained by reporting empirical intervals directly.
> 3. **Candidate Pool Topicity:** The AVeriTeC candidate pool evaluation represents oracle retrieval (pre-selected candidate passages), isolating verifier accuracy from full open-web retrieval bottlenecks. Full end-to-end retrieval is tested separately via TinyFish in Section 4.

Per-claim model predictions, confusion matrices, dataset SHA-256, sweep records, cross-verifier diagnostics, and CI assertions are saved in [`outputs/benchmark_results.json`](outputs/benchmark_results.json).

---

### 3. End-to-End Case Study & Narrow Trace (`demo_case_study.py`)
To inspect exact system mechanics without relying solely on aggregate metrics, [`demo_case_study.py`](demo_case_study.py) runs an auditable trace across:
* **Compound Claim Verification:** Decomposes *"The Eiffel Tower was constructed in Rome in 1950 by Gustave Eiffel"* into atomic assertions, filters distractor passages via the RefNLI alignment gate, evaluates predicate conflicts, and generates a visual provenance graph.
* **Conservative Abstention:** Evaluates speculative or unproven claims (*"Quantum computers officially proved that P equals NP in 2024"*), demonstrating how the engine conserves abstention as `INSUFFICIENT_EVIDENCE` instead of hallucinating false support.
* **Persisted Case Study Report:** Output saved to [`outputs/demo_case_study.md`](outputs/demo_case_study.md).

---

### 3. Synthetic Context-Mismatch Stress Diagnostic (N=20 Paired Cases)
To inspect the behavior of the **RefNLI-inspired alignment gate**, we evaluated a synthetic diagnostic suite of 10 aligned pairs and 10 reference-mismatch distractors. The cases are hand-authored and share predicates, dates, or locations while varying the subject (see [docs/DATA_CARD.md](docs/DATA_CARD.md)):

| Alignment Gate Configuration | Accuracy (N=20) | False Refutations on Distractors (N=10) |
| :--- | :---: | :---: |
| **ProofGraph (With RefNLI Alignment Gate)** | **100.0% (20/20)** | **0/10 (0.0% False Refutations)** |
| Ablation (WITHOUT RefNLI Alignment Gate) | 60.0% (12/20) | 3/10 (30.0% False Refutations) |

*(In this hand-authored suite, removing the alignment gate caused 3/10 false refutations; with the gate, the suite had 0/10. These examples are not the RefNLI benchmark and do not establish generalization to unseen cases.)*

---

### 4. Dedicated Live TinyFish Web Retrieval Benchmark (`evaluate_live.py`)
To validate real-world autonomous search and page fetch independently of bundled offline fixtures, `evaluate_live.py` executes 10 diverse claims against live TinySearch and TinyFetch endpoints:
* **Search Coverage:** **100.0%** (10/10 queries returned at least one candidate passage in the recorded run). This measures retrieval presence, not whether the passages are relevant or authoritative.
* **Fetches and classification:** The saved run reports 26 candidate passages, 14 page fetches, 113.31 seconds total, and **5/10 correct end-to-end verdicts**. This is a small demonstration result, not evidence of reliable live verification.
* **Source Quality:** The run includes a mixture of sources, including Wikipedia, government/reference domains, and social or general-interest sites; source quality still needs explicit measurement and filtering.
* **Full Audit Report:** Persisted at [`outputs/tinyfish_live_evaluation.md`](outputs/tinyfish_live_evaluation.md).

---

### 5. Diagnostic Unit Sanity Suite (10 Edge Cases)
* **ProofGraph Hybrid:** `80.0% Accuracy` | `0.750 Macro-F1` on the 10 hand-authored cases in the current run; this is a diagnostic suite, not a generalization estimate.
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
   │    • Heuristic claim splitting (Claimify-inspired)     │
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
   │    • TinyFish Search → Fetch (optional live retrieval) │
   │    • Live Wikipedia API Fallback for Open Domains      │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 3. RuleStanceVerifier & RefNLI Context Guardrails     │
   │    • RefNLI-inspired lexical entity-alignment filter   │
   │    • Safe Predicate-Bound Numeric & Temporal Conflicts │
   │    • Static domain credibility prior (CONFACT-inspired)│
   │    • Rule-based abstention (INSUFFICIENT_EVIDENCE)     │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 4. PROV-DM-inspired provenance graph                  │
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
* **FIRE (NAACL 2025):** Introduces iterative retrieval with dynamic stopping decisions, achieving 7.6x LLM cost and 16.5x search cost reductions (reported in the original FIRE paper evaluations). *ProofGraph adapts FIRE's entity-coverage dynamic stopping heuristic.*
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

# Option A: Install pinned direct dependencies used by CI
pip install -r requirements-lock.txt

# Option B: Standard install
pip install -r requirements.txt
```

### Environment Variables (Optional)
```bash
# Optional: Enable TinyFish Search and Fetch in the CLI
export TINYFISH_API_KEY="sk-tinyfish-your-key-here"
```

### 1. Run the Offline Fixture Demo
Uses four bundled evidence passages; it does not contact the live web. It executes heuristic claim splitting, local hybrid retrieval, lexical alignment filtering, static domain weighting, and writes `outputs/graph.html` and `outputs/report.md`:
```bash
python demo.py
```

### 2. Run the Benchmark Evaluation
Runs the offline diagnostic comparisons across verifier backends (`--backend hybrid|rules|neural|all`):
```bash
# Default runs the ProofGraph Hybrid verifier (Rules + Neural Cross-Encoder)
python evaluate.py

# Or compare all verifier backends and ablations side-by-side
python evaluate.py --backend all
```

### 3. Run the Dedicated Live TinyFish Benchmark (Optional, Keyed)
Executes 10 real-world claim verification queries against live TinySearch and TinyFetch endpoints:
```bash
export TINYFISH_API_KEY="sk-tinyfish-your-key-here"
python evaluate_live.py
```

### 4. Run the CLI
With `TINYFISH_API_KEY` set, the CLI uses TinyFish Search followed by TinyFish Fetch for live evidence retrieval. Use `--no-live-search` to force offline mode:
```bash
python -m proofgraph.cli --text "Apollo 11 landed on the moon on July 20, 1969. The Eiffel Tower was built in Rome in 1950." --output outputs/
```

### 5. Run the Pytest Test Suite
Run the offline test suite with coverage reporting. The live TinyFish smoke test is skipped by default; set both `TINYFISH_API_KEY` and `TINYFISH_LIVE_TEST=1` to opt into a real network request:
```bash
PYTHONPATH=. pytest tests/ -v --cov=proofgraph --cov-report=term-missing
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
│   ├── decomposer/      # Heuristic claim splitting and opinion filter
│   ├── retriever/       # BM25 + FastEmbed, FIRE-inspired loop, optional TinyFish Search/Fetch
│   │   └── tinyfish_agent.py # TinyFish Search → Fetch adapter
│   ├── verifier/        # RuleStanceVerifier, DebertaNLIVerifier (ONNX), HybridNLIVerifier, RefNLI Gate
│   │   ├── alignment.py   # RefNLI Context Alignment Gate (blocks reference indeterminacy)
│   │   ├── credibility.py # Domain credibility scoring with spoofing defense
│   │   ├── neural_nli.py  # Quantized ONNX DeBERTa-v3 cross-encoder (zero-PyTorch)
│   │   └── nli.py         # Deterministic rule engine & Hybrid ensemble verifier
│   ├── graph/           # PROV-DM-inspired graph builder and contradiction edges
│   ├── reporter/        # Audit report generator with inline citations
│   ├── pipeline.py      # Unified end-to-end ProofGraph pipeline orchestrator
│   └── cli.py           # Command line interface
├── tests/
│   └── test_pipeline.py # 24 unit & integration tests with coverage tracking
├── outputs/             # Generated demo artifacts (graph.html, report.md, tinyfish_live_evaluation.md)
├── demo.py              # Offline end-to-end showcase script
├── evaluate.py          # Benchmark & ablation evaluation script (AVeriTeC & diagnostic suite)
├── evaluate_live.py     # Dedicated live TinyFish Search→Fetch evaluation benchmark
├── requirements.txt     # Consolidated dependencies
├── requirements-lock.txt# Pinned direct dependencies used by CI
├── pytest.ini           # Pytest configuration
├── LICENSE              # Apache-2.0 Open Source License
├── .gitignore
└── README.md
```

---

## 👥 Team & Event
* **Event:** Hack on Holiday 2026 (Hosted by The Reinforce Club, Scaler School of Technology)
* **Team:** AltF4
* **Members:** Mohith GR & Sudarshan Patil HJ
* **Track:** Advanced Track / Product

---

## 📜 License
Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Mohith GR, Sudarshan Patil HJ (Team AltF4).
