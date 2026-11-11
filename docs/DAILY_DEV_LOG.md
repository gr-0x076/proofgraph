# ProofGraph: Daily Development Log & Continuous Progress Report

> **Event:** Hack on Holiday 2026 (Hosted by The Reinforce Club, Scaler School of Technology)
> **Team:** AltF4 (Mohith GR & Sudarshan Patil HJ)
> **Track:** Advanced Track / Product
> **Repository:** [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph)
> **Build Window:** November 5 – November 15, 2026
> *(Note: This document serves as our internal engineering changelog and milestone tracker across the 10-day build window).*

---

## 📅 Day-by-Day Milestone Log

### Day 1: November 5, 2026 — Problem Selection & Architectural Teardown
* **Milestone:** Problem Statement Selected & Research Grounding
* **Activities:**
  * Analyzed the 11 official hackathon problem statements from the Idea Jar; selected **ProofGraph (Statement #6, Advanced Product)**.
  * Conducted deep literature review of 83 academic papers and leading fact-checking architectures, including RefNLI (ACL 2024), GraphCheck (ACL 2025), FIRE (NAACL 2025), KG-CRAFT (EACL 2026), and CONFACT (2025).
  * Generated initial landscape report: `docs/research/01_executive_landscape.txt` and `docs/research/02_prior_art_critique.md`.
  * Formulated foundational design decision: reject brittle external Neo4j services in favor of an embedded, zero-dependency Python-native graph stack (NetworkX + PyVis + Pydantic v2).

### Day 2: November 6, 2026 — Research Landscape & Schema Specification
* **Milestone:** Data Contracts & Standards Definition
* **Activities:**
  * Formatted exhaustive 20-page research landscape PDF: `docs/assets/ProofGraph-An-Exhaustive-Evidence-Linked-Landscape.pdf`.
  * Designed data models strictly adhering to W3C PROV-DM:
    * `AtomicClaim`: verifiability flag, decontextualized text, entity mentions.
    * `EvidencePassage`: source URI, canonical domain, raw text, relevance score, credibility prior.
    * `ClaimVerdict`: 3-way veracity (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`), confidence calibration, justification audit trail.
    * `ProvGraphExport`: JSON serialization schema for provenance entities, activities, and derivations.
  * Documented benchmark selection criteria in `docs/research/03_benchmarks_and_datasets.md`.

### Day 3: November 7, 2026 — Decomposition Engine (Claimify Pipeline)
* **Milestone:** Atomic Proposition Extraction Engine
* **Activities:**
  * Implemented `ClaimDecomposer` in `proofgraph/decomposer/decomposer.py`.
  * Added abbreviation-protected sentence boundary tokenization (preserving `Dr.`, `U.S.`, `Inc.`, `e.g.`).
  * Built subjectivity / opinion detection filter using lexical hedging and subjective sentiment markers to abstain from non-verifiable assertions.
  * Enforced mathematical invariant: $N_{\text{subclaims}} \le N_{\text{sentences}} \times k$ to prevent combinatorial hallucination loops.

### Day 4: November 8, 2026 — Multi-Stage Hybrid Retrieval & Dynamic Stopping
* **Milestone:** Dense + Lexical Fusion & FIRE Policy
* **Activities:**
  * Implemented `HybridRetriever` in `proofgraph/retriever/hybrid_retriever.py`.
  * Integrated dual-channel retrieval:
    * Lexical: `rank-bm25` (BM25Okapi).
    * Semantic Dense: `fastembed` ONNX runtime using `BAAI/bge-small-en-v1.5` (running CPU/GPU in <50ms without PyTorch overhead).
    * Fusion: Reciprocal Rank Fusion (RRF, $k=60$).
  * Implemented `FIREIterativeDecisionModule` (`proofgraph/retriever/fire_policy.py`) for iterative multi-hop retrieval and dynamic early stopping.
  * Added live Wikipedia REST API fallback for un-indexed open-domain entities.

### Day 5: November 9, 2026 — Alignment Gates, Credibility Scorer & PROV-DM Graph
* **Milestone:** Verification Engine & Interactive Visualizer
* **Activities:**
  * Built `RefNLIAlignmentGate` in `proofgraph/verifier/alignment.py` to prevent the reference indeterminacy false-contradiction trap via token Jaccard and named entity overlap.
  * Implemented `SourceCredibilityScorer` in `proofgraph/verifier/credibility.py` implementing domain authority weighting.
  * Hardened credibility scorer against domain spoofing attacks (e.g. `reuters.com.evil.org`) using canonical URL parsing and label-bounded suffix checks.
  * Implemented `ProofGraphBuilder` in `proofgraph/graph/builder.py` with first-class `SUPPORTS` (green), `CONTRADICTS` (bold red), and `DISAGREES_WITH` (dashed orange) edges.
  * Built `ProofGraphVisualizer` using PyVis physics simulation to export standalone interactive HTML (`outputs/graph.html`).
  * Implemented `ReportGenerator` compiling auditable Markdown audit reports (`outputs/report.md`).

### Day 6: November 10, 2026 — Rigorous 4-Part Benchmark, Offline Portability & Hardening
* **Milestone:** Empirical IR & Verdict Evaluation, Offline Portability, 11 Passing Tests
* **Activities:**
  * Curated 60 real-world claims from the official **AVeriTeC Dev Benchmark** (NeurIPS 2023) across 50 international organizations (`data/benchmarks/averitec_dev_sample.json`).
  * Built comprehensive 4-part evaluation suite in `evaluate.py`:
    1. **Evidence Retrieval Quality (Pooled Distractor Corpus):** Pooled all 170 gold evidence passages from 60 claims into a single global index. Measured Hits@1 (83.3%), Hits@3 (86.7%), Hits@5 (93.3%), and MRR (0.863).
    2. **AVeriTeC Dev Split Classification:** 50.0% Accuracy and 0.464 Macro-F1 (Supported F1: 0.611) vs. standard dense similarity baseline (41.7% Accuracy, 0.350 Macro-F1) and lexical baseline (46.7% Accuracy).
    3. **Full Label Breakdown & Mapping:** Disclosed 3x3 confusion matrix and 4-class official AVeriTeC mapping (`Not Enough Evidence` and `Conflicting Evidence` mapped to calibrated abstention).
    4. **Held-Out RefNLI Evaluation (N=20):** Validated alignment gate on reference-indeterminacy distractors: ProofGraph achieved 0.0% False Refutations vs. 50.0% false refutations when the gate is ablated.
  * Added `DeterministicHashEmbedding` in `HybridRetriever` for 100% offline unit testing in firewalled/network-restricted sandboxes.
### Day 7: November 11, 2026 — TinyFish Web Infrastructure, Safe Relation-Bound Conflict Guardrails, Matrix CI & Lockfile Hardening
* **Milestone:** Live TinyFish Search Integration, Syntactic Predicate Binding, Reproducible CI & 15 Tests
* **Activities:**
  * Re-architected `TinyFishWebAgent` in `proofgraph/retriever/tinyfish_agent.py` to use official TinyFish Web Infrastructure endpoints:
    * TinySearch API (`https://api.search.tinyfish.ai?query=...`) for real-time structured search results with exact source URLs, page titles, and snippets.
    * TinyFetch API (`POST https://api.fetch.tinyfish.ai`) for full-text page content extraction.
    * Removed static relevance scores and invented URLs; implemented dynamic token-overlap and rank-discounted relevance scoring.
    * Added full error tracking and execution status observability (`last_status`, `last_error_code`, `total_queries_run`).
  * Hardened `RuleStanceVerifier` in `proofgraph/verifier/nli.py`:
    * Upgraded numeric conflict detection to require local windowed binding ($\le 35$ chars) between numbers and shared action predicates (`boil`, `born`, `built`, `launch`, etc.), eliminating false contradictions between unrelated numbers.
    * Upgraded named inventor conflict detection to require shared object overlap (e.g. `telephone`, `penicillin`) before flagging disjoint inventor names.
  * Added adversarial conflict tests (`test_adversarial_numeric_and_inventor_conflicts`) and live opt-in smoke test (`test_live_tinyfish_search_api_if_keyed`), bringing the test suite to **15 passing tests**.
  * Enforced reproducible builds in GitHub Actions (`.github/workflows/ci.yml`) by installing from `requirements-lock.txt`.
  * Updated dataset provenance, protocol disclosures, and synthetic diagnostic boundaries in `docs/DATA_CARD.md`.

---

## 📈 Quantitative Performance Summary Across All 4 Benchmarks

| Benchmark Dimension | Baseline Configuration | ProofGraph Engine | Delta (Net Improvement) |
| :--- | :---: | :---: | :---: |
| **Evidence Retrieval Quality (Pooled AVeriTeC, 170 Passages)** | Pure Dense: 81.7% Hits@1 / 0.860 MRR | **83.3% Hits@1 / 93.3% Hits@5 / 0.863 MRR** | **Superior Top-5 Retrieval** |
| **AVeriTeC Dev Split Classification (N=60 Claims)** | Dense Baseline: 41.7% Acc / 0.350 F1 | **50.0% Acc / 0.463 Macro-F1 / 0.375 Contra Prec** | **+8.3% Acc / +0.113 F1 / 2.2x Contra Prec** |
| **Synthetic Context-Mismatch Stress Diagnostic (N=20 Pairs)** | Ablated (No Gate): 30.0% False Refutations | **0.0% False Refutations (100% Acc)** | **Eliminated False Refutations** |
| **Diagnostic Unit Sanity Suite (10 Hand-Authored Cases)** | Lexical Baseline: 50.0% Acc / 0.389 F1 | **90.0% Acc / 0.886 Macro-F1** | **+40.0% Acc / +0.497 F1** |
| **Domain Spoofing Vulnerability** | Substring Match (Vulnerable) | **100% Canonical Host Filtered** | **Hardened via Label-Bounded Match** |
| **Pytest Unit & Integration Suite** | N/A | **15/15 Passing (14 Offline + 1 Live Keyed)** | **Runs in <4s** |

### Follow-up Audit Corrections — November 11, 2026

The entries above record the Day 7 state and should not be read as the current test or live-integration status. A subsequent local audit made these corrections:

* TinyFish now follows Search results with a batched Fetch request, selects a verbatim text paragraph, preserves the final page URL/title, records fetch errors, and rejects malformed or obvious private/local target URLs. Its live test is explicitly opt-in with both `TINYFISH_API_KEY` and `TINYFISH_LIVE_TEST=1`.
* Added mocked Search→Fetch provenance, URL-validation, retry, and Fetch error-path tests. Current offline result: **17 passed, 1 skipped** (the opt-in live TinyFish check). The opt-in live check then passed separately, and a CLI smoke run retrieved NASA, Wikipedia, and HISTORY pages through `tinyfish_fetch` and generated a report. `demo.py` remains an offline fixture demo.
* Updated README and pitch language to distinguish the heuristic prototype from Claimify, RefNLI, CONFACT, FIRE, and W3C PROV-DM; relabeled the 20-pair test as synthetic rather than held out; and changed displayed confidence to an uncalibrated rule score.
* Current `evaluate.py` diagnostic achieves **50.0% accuracy / 0.490 macro-F1** on the custom AVeriTeC candidate-pool sample with the Hybrid (Rules + Neural) verifier, doubling contradiction recall to 0.333. The synthetic gate result is 20/20 on its hand-authored cases. TinyFish contributes to none of these offline metrics; its live retrieval performance is evaluated separately via `evaluate_live.py`.
* `requirements-lock.txt` pins direct dependencies for CI but does not hash-lock the full transitive dependency graph; avoid describing it as a fully reproducible build.

### Day 8: November 11, 2026 — ONNX Neural Cross-Encoder, Hybrid Verification, Live TinyFish Eval Suite & CI Coverage
* **Milestone:** Neural Cross-Encoder Fallback, Hybrid NLI Ensemble, Live TinyFish Benchmark, 23 Passing Tests & 80% CI Coverage
* **Activities:**
  * **Zero-PyTorch Neural NLI Cross-Encoder (`DebertaNLIVerifier`):**
    * Integrated quantized `Xenova/nli-deberta-v3-xsmall` (83.3MB ONNX model) via `onnxruntime` and Rust `tokenizers`.
    * Sub-10ms CPU inference per premise-hypothesis pair without heavy PyTorch/CUDA dependencies.
    * Protected by `RefNLIAlignmentGate` to suppress reference-indeterminacy false contradictions before neural inference.
    * Fused with `SourceCredibilityScorer` to weight entailment/contradiction probabilities by domain authority and retrieval relevance.
  * **Hybrid NLI Ensemble (`HybridNLIVerifier`):**
    * Combines deterministic high-precision rule verification (numeric, temporal, attribution conflicts) with neural cross-encoder fallback when rules abstain.
    * Expanded `REFUTATION_MARKERS` to catch nuanced fact-checking phrases (`wrongly`, `satire`, `parody`, `imaginary`, `fabricated`, `unfounded`, `baseless`, `unsubstantiated`, `fictional`).
    * Lifted AVeriTeC Macro-F1 to **0.490** and Contradiction Recall to **0.333** (5/15, up from 0.067 on standard baselines).
  * **Dedicated Live TinyFish Retrieval Benchmark (`evaluate_live.py`):**
    * Created dedicated 10-claim live evaluation harness testing `api.search.tinyfish.ai` and `api.fetch.tinyfish.ai`.
    * Achieved 100% retrieval coverage (10/10 claims retrieved valid, authoritative web pages from `.gov`, `.edu`, and Wikipedia).
    * Extracted 30 live web pages via Chromium rendering in 115 seconds, generating auditable report at `outputs/tinyfish_live_evaluation.md`.
  * **Multimodal Relevance Calibration:**
    * Updated `HybridRetriever.search_local` to calibrate relevance scores via balanced BM25 and dense cosine fusion (`0.5 * norm_bm25 + 0.5 * dense_score`).
    * Aligned `TinyFishWebAgent` domain parsing with official TinySearch response schema (`domain` field).
  * **Pipeline Cleanliness & CI Coverage:**
    * Fixed graph visualizer link rendering in `ReportGenerator` when `output_dir is None` (setting `graph_html_rel_path=None`).
    * Added `--cov=proofgraph --cov-report=term-missing` to GitHub Actions CI workflow. At this point coverage was reported but not enforced; later audit adds an 80% minimum.

### Audit correction & Production Hardening — November 11, 2026
* **Evaluation Leakage Resolved:** Re-ran `evaluate.py --backend all` after eliminating retriever index accumulation. The evaluation now delivers candidate evidence pools directly to the example verifier without cross-claim index leakage.
* **Calibrated Hybrid Fallback (`neural_fallback_threshold=0.90`):** Requires high neural confidence (≥0.90) before overturning rule-based abstentions. This eliminates false contradictions on ambiguous claims while capturing genuine refutations, lifting Hybrid performance to **51.7% Accuracy / 0.504 Macro-F1 / 0.400 Contradiction Recall** (surpassing Rules Alone at 0.493 and Baselines at 0.350/0.398).
* **Immutable Model Revision & Label Validation:** Pinned `Xenova/nli-deberta-v3-xsmall` to immutable commit SHA `2a4f614a701367a02d51389039afc998faeda637` (Apache-2.0 base: `cross-encoder/nli-deberta-v3-xsmall`). Added dynamic runtime validation of `config.json` `id2label` confirming index 0=contradiction, 1=entailment, 2=neutral.
* **Warm-Run Dynamic Padding:** Eliminated fixed 512 padding, achieving ~10 ms median CPU inference per pair.
* **Auditable Live Web Retrieval Artifact:** Re-ran `evaluate_live.py` with real TinySearch & TinyFetch endpoints. Generated auditable table of 26 retrieved source URLs, domain distribution, and 50% live accuracy disclosure at [`outputs/tinyfish_live_evaluation.md`](../outputs/tinyfish_live_evaluation.md).
* **CI Matrix Hardening:** Added model weight caching, `--cov-fail-under=80` coverage threshold (current: 82.2%), benchmark execution, and live fixture validation across Python 3.10, 3.11, and 3.12.
