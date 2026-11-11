# Data Card: Benchmarks and Datasets in ProofGraph

> **Project:** ProofGraph — Autonomous AI Evidence Intelligence Engine  
> **Repository:** [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph)  
> **Last Updated:** November 2026  

---

## 1. Bundled Benchmark: `data/benchmarks/averitec_dev_sample.json`

### 1.1 Provenance & Original Source
* **Dataset Name:** AVeriTeC (Automated Verification of Textual Claims)
* **Citation:** Michael Schlichtkrull, Zhijiang Guo, Andreas Vlachos (NeurIPS 2023 Datasets & Benchmarks Track). *"AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web."*
* **Official Repository:** [https://github.com/MichSchli/AVeriTeC](https://github.com/MichSchli/AVeriTeC)
* **Official Website:** [https://fever.ai/dataset/averitec.html](https://fever.ai/dataset/averitec.html)
* **Original License:** Creative Commons Attribution-NonCommercial 4.0 International ([CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/))

### 1.2 Subsampling Methodology & Size
* **Sampling Frame:** Drawn from the official 500-claim AVeriTeC development split.
* **Sample Size:** 60 claims (12% of the official dev split).
* **Class Distribution in Sample:**
  * `Supported`: 15 claims (25.0%)
  * `Refuted`: 15 claims (25.0%)
  * `Conflicting Evidence/Cherrypicking`: 15 claims (25.0%)
  * `Not Enough Evidence`: 15 claims (25.0%)
* **Total Gold Evidence Passages:** 170 passages across the 60 claims (averaging ~2.83 passages per claim).
* **Sampling Rationale:** Constructed as a balanced diagnostic sample to evaluate model stance and abstention across all four official verification categories under equal class priors.

### 1.3 Label Mapping Disclosure (4-Way to 3-Way)
In standard Fact Extraction and VERification (FEVER) and Natural Language Inference (NLI) literature, classification is typically evaluated on a 3-way schema. We map the 4 official AVeriTeC labels as follows:
* `Supported` $\rightarrow$ **`SUPPORTED`**
* `Refuted` $\rightarrow$ **`CONTRADICTED`**
* `Not Enough Evidence` $\rightarrow$ **`INSUFFICIENT_EVIDENCE`**
* `Conflicting Evidence/Cherrypicking` $\rightarrow$ **`INSUFFICIENT_EVIDENCE`**

*Methodological Note:* Collapsing `Not Enough Evidence` and `Conflicting Evidence` into `INSUFFICIENT_EVIDENCE` reflects the operational reality that both categories require the automated fact-checker to abstain from declaring a definitive true/false verdict. The sub-class breakdown is reported in `evaluate.py` to preserve full visibility into the two distinct failure modes.

### 1.4 Evaluation Protocol Disclosure
* **Protocol Used in ProofGraph:** Verdict classification given claim-specific candidate evidence pools, plus in-batch evidence retrieval quality (Hits@1, Hits@3, Hits@5, MRR) over a pooled corpus of all 170 candidate passages.
* **Distinction from Official Shared Task:** The official AVeriTeC shared-task competition score (FEVER 2024 / 2025) requires full open-web retrieval and evaluates question-answering evidence using Hungarian METEOR matching ($\ge 0.25$) or Ev2R recall ($\ge 0.44$). The results reported in ProofGraph represent **verdict classification on candidate evidence pools** and must not be directly compared against official EvalAI leaderboard submissions.

---

## 2. Hand-Authored Diagnostic Suites

### 2.1 Diagnostic Unit Sanity Suite (10 Stress Cases)
* **File:** Implemented directly in [`evaluate.py`](../evaluate.py) (`DIAGNOSTIC_DATASET`).
* **Purpose:** A developer-authored unit stress test designed to isolate specific edge-case failure modes:
  * Temporal discrepancy (e.g. 1950 vs. 1889)
  * Geographical conflict (e.g. Rome vs. Paris)
  * Numerical quantity conflict (e.g. 50°C vs. 100°C)
  * Inventor / attribution conflict (e.g. Edison vs. Bell)
  * Reference indeterminacy (e.g. Peter vs. John)
  * Non-verifiable subjective opinion detection
* **Scope:** 10 hand-authored diagnostic cases with paired gold evidence. Not an open-domain benchmark.

### 2.2 Synthetic Context-Mismatch Stress Test (20 Pairs)
* **File:** Implemented directly in [`evaluate.py`](../evaluate.py) (`CONTEXT_MISMATCH_BENCHMARK`).
* **Purpose:** A synthetic diagnostic suite isolating the "Reference Indeterminacy" failure mode documented in the RefNLI paper (ACL / NAACL 2025):
  * 10 Aligned Pairs (Ground Truth: `SUPPORTED`)
  * 10 Reference-Indeterminacy Distractors (Ground Truth: `INSUFFICIENT_EVIDENCE`), where premise and hypothesis share dates, actions, and locations, but describe different named subjects.
* **Scope & Limitation:** This is a focused, synthetic stress test for the primary subject alignment gate. It is **not** the official RefNLI benchmark dataset (which consists of 1,143 NLI pairs across 4 labels: Entailment, Contradiction, Ambiguous, Neutral).

---

## 3. Live Web Retrieval via TinyFish Platform

* **Provider:** TinyFish Web Infrastructure for AI Agents ([tinyfish.ai](https://www.tinyfish.ai/))
* **Primary Search API:** `https://api.search.tinyfish.ai?query=...` (TinySearch structured engine results)
* **Page Content API:** `POST https://api.fetch.tinyfish.ai` (TinyFetch Chromium rendering)
* **Authentication:** `X-API-Key: $TINYFISH_API_KEY`
* **Evidence Provenance Guarantee:**
  * All retrieved passages preserve the exact source URL, HTTP status code, domain name, and ISO retrieval timestamp in `EvidencePassage.metadata`.
  * Dynamic lexical-semantic relevance scoring is calculated based on query token overlap and ranked search position (no invented URLs or static relevance scores).
* **Evaluation Decoupling:** All benchmark runs in `evaluate.py` execute with `use_live_web_fallback=False` to ensure 100% deterministic, offline-reproducible benchmark measurements. Live TinyFish retrieval is validated separately via the dedicated live evaluation suite (`evaluate_live.py`) and the opt-in pytest smoke test (`test_live_tinyfish_search_api_if_keyed`).

---

## 4. Neural Cross-Encoder Model Card: `DebertaNLIVerifier`

### 4.1 Architecture & Runtime
* **Base Model:** `Xenova/nli-deberta-v3-xsmall` (HuggingFace Hub)
* **Underlying Architecture:** DeBERTa-v3 cross-encoder fine-tuned for Natural Language Inference (MNLI / SNLI).
* **Format & Execution Engine:** Quantized ONNX (`model.onnx`, 83.3 MB) executed via `onnxruntime` and Rust-backed `tokenizers`.
* **Zero-PyTorch Guarantee:** Runs on standard CPU architectures without PyTorch, torch-vision, CUDA, or heavy C++ build toolchains.
* **Inference Latency:** A local warm-run microbenchmark (25 calls after warm-up) measured 10.2 ms median and 11.2 ms p95 per pair in the development environment after removing unnecessary fixed-length padding. Treat this as a machine-specific observation, not a cross-device guarantee.
* **Stance Output:** 3-way softmax distribution over `[CONTRADICTION, ENTAILMENT, NEUTRAL]` dynamically validated from model `config.json` at startup (`id2label`: `0: contradiction, 1: entailment, 2: neutral`).
* **Pinned Model Revision & License:** Model weights are pinned to immutable commit SHA `2a4f614a701367a02d51389039afc998faeda637` on Hugging Face Hub. The underlying base model is `cross-encoder/nli-deberta-v3-xsmall` released under the [Apache-2.0 License](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall), fully compatible with ProofGraph's Apache-2.0 license.
* **Calibrated Hybrid Fallback:** `HybridNLIVerifier` enforces `neural_fallback_threshold=0.90` before overturning rule-based abstentions, preventing false contradictions on ambiguous claims while lifting Contradiction Recall to 0.333 and Macro-F1 to 0.504.

### 4.2 Guardrails & Pre-Filtering
* **Context Alignment Gate (`RefNLIAlignmentGate`):** Evaluated strictly *prior* to neural cross-encoder execution. Unaligned premise-hypothesis pairs (e.g., distinct named individuals sharing dates or verbs) are intercepted and marked `Context Mismatch`, bypassing neural inference to prevent reference-indeterminacy false refutations.
* **Domain Credibility Prior Fusion (`SourceCredibilityScorer`):** Model softmax scores are multiplied by static domain-authority priors and retrieval relevance for evidence aggregation. The resulting verdict confidence is heuristic and is not probability-calibrated.

---

## 5. Dedicated Live TinyFish Benchmark Suite (`evaluate_live.py`)

### 5.1 Purpose & Methodology
To rigorously evaluate the live TinySearch $\rightarrow$ TinyFetch pipeline independently of synthetic or offline corpora, `evaluate_live.py` executes 10 diverse real-world claims against live endpoints (`api.search.tinyfish.ai` and `api.fetch.tinyfish.ai`).

### 5.2 Metrics Measured
* **Retrieval Coverage:** Proportion of claims for which live search returns $\ge 1$ relevant URL.
* **Page Fetch Success Rate:** Proportion of target URLs successfully rendered into markdown via Chromium.
* **Latency Profile:** Individual breakdown of query dispatch, search round-trip, and page fetch durations.
* **Source Quality:** Canonical domain credibility scores and content excerpt lengths.
* **Output Artifact:** Persisted as a self-contained, auditable Markdown report at [`outputs/tinyfish_live_evaluation.md`](../outputs/tinyfish_live_evaluation.md).
