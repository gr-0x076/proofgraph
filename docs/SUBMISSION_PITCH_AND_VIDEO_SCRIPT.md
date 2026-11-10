# ProofGraph: Official Hackathon Submission Deck & 3-Minute Demo Script

> **Event:** Hack on Holiday 2026 — Reinforce Club, Scaler School of Technology  
> **Team:** Reinforce SPG  
> **Track:** Advanced Track / Product  
> **Live Repository:** [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph)  

---

## 📽️ Part 1: 3-Minute Demo Video Walkthrough Script

### [00:00 – 00:30] Hook & Problem Statement
* **Visual:** Split screen showing a generated LLM hallucination vs. a flat, unverified RAG summary.
* **Speaker:**
  > "Every modern AI system retrieves documents and generates confident answers. But when LLMs hallucinate or sources contradict each other, standard RAG collapses into a black box. Even worse, current systems suffer from the **False Contradiction Trap**—flagging claims as refuted simply because a retrieved document discusses a different person with matching dates or verbs!
  >
  > Welcome to **ProofGraph**: an autonomous AI evidence intelligence engine that doesn't just guess verdicts—it decomposes claims into atomic propositions, evaluates source credibility, resolves conflicting evidence, and builds an interactive, auditable provenance graph with first-class contradiction edges."

---

### [00:30 – 01:15] System Architecture & Deep Tech Innovations
* **Visual:** Flow diagram from README showing the 5-stage pipeline with real code snippets.
* **Speaker:**
  > "ProofGraph solves verification through five principled stages:
  > 
  > 1. **Atomic Proposition Decomposition:** Inspired by Claimify (ACL 2025), it splits multi-assertion text into atomic, decontextualized propositions while filtering out non-verifiable opinions.
  > 2. **Multi-Stage Hybrid Retrieval & TinyFish Web Agent:** Combining BM25 lexical search with FastEmbed ONNX dense embeddings via Reciprocal Rank Fusion ($k=60$), integrated with TinyFish's live Search and Fetch APIs for open-domain evidence retrieval. Guided by a FIRE-inspired heuristic (NAACL 2025), it evaluates entity coverage in real-time, executing targeted multi-hop queries when critical entities are missing.
  > 3. **RuleStanceVerifier & RefNLI Context Alignment Gate:** Before evaluating stance, a primary-subject overlap gate inspired by RefNLI blocks reference mismatches. The stance engine uses conservative, predicate-bound numeric and temporal conflict rules to eliminate false refutations.
  > 4. **CONFACT-Inspired Credibility Prior:** Sources aren't treated equally. High-authority institutions like NASA, Reuters, and peer-reviewed journals carry up to $6.5\times$ the voting mass of unverified blogs, protected by canonical hostname spoof-guarding.
  > 5. **W3C PROV-DM Graph Construction:** Exporting auditable knowledge graphs with color-coded nodes and explicit red contradiction physics."

---

### [01:15 – 02:00] Live Demonstration (`demo.py`)
* **Visual:** Screen capture running `python demo.py`, terminal output, then opening `outputs/graph.html` in the browser.
* **Speaker:**
  > "Let's run the end-to-end demo live:
  >
  > Input: *'Apollo 11 landed on the moon on July 20, 1969 with Neil Armstrong. The Eiffel Tower was constructed in Rome in 1950 by Italian engineers. Personally, I think Rome is the greatest architectural city in the world.'*
  >
  > In less than 2 seconds on a standard laptop CPU:
  > - Claim 1 is marked **🟢 SUPPORTED** with 95% confidence via NASA Archives.
  > - Claim 2 is marked **🔴 CONTRADICTED** with 95% confidence via Encyclopaedia Britannica, exposing both the temporal and geographical conflicts.
  > - Claim 3 is flagged as **🟡 INSUFFICIENT EVIDENCE (Opinion)**, safely abstaining.
  >
  > In `outputs/graph.html`: we can drag nodes, inspect evidence passages, follow green support paths, and trace bold red contradiction edges linking conflicting claims to their disproving sources."

---

### [02:00 – 02:35] Empirical Validation across 4 Diagnostic Suites
* **Visual:** Slide showing the AVeriTeC Dev Split confusion matrix, retrieval IR table, and synthetic context-mismatch stress test.
* **Speaker:**
  > "We evaluated ProofGraph across four complementary dimensions:
  >
  > 1. **In-Batch Evidence Retrieval Quality Diagnostic (Pooled AVeriTeC, 170 Passages):** On a global index with cross-claim distractors, our hybrid retriever achieved **93.3% Hits@5** and an **MRR of 0.863**.
  > 2. **AVeriTeC Dev Split Sample Classification (NeurIPS 2023, N=60 Claims):** Given candidate evidence pools, ProofGraph scored **50.0% Accuracy** and **0.463 Macro-F1**, outperforming the unweighted dense similarity baseline (41.7% Accuracy, 0.350 Macro-F1).
  > 3. **Synthetic Context-Mismatch Stress Diagnostic (N=20 Hand-Authored Pairs):** Directly targeting the reference indeterminacy failure mode highlighted in RefNLI research, the gate achieved **0/10 false refutations (0.0%)**, compared to 30.0% false refutations when ablated.
  > 4. **Diagnostic Unit Sanity Suite:** 90.0% Accuracy across edge cases, backed by 15 unit and integration tests including an opt-in live TinySearch API smoke test."

---

### [02:35 – 03:00] Impact, Future Roadmap & Conclusion
* **Visual:** Slide showing GitHub repository, reproducible commands, and team credits.
* **Speaker:**
  > "ProofGraph is fully open-source under Apache-2.0, provides reproducible builds via pinned lockfiles, runs on CPU in milliseconds via ONNX, and outputs auditable Markdown reports with inline citations.
  >
  > We are Team Reinforce SPG, and ProofGraph brings transparency, accountability, and evidence graph intelligence to claim verification. Thank you!"

---

## 📊 Part 2: Slide-by-Slide Pitch Deck Structure (Canva Presentation)

### Slide 1: Title & Team
* **Header:** ProofGraph: Autonomous AI Evidence Intelligence & Claim Verification Engine
* **Subheader:** Hack on Holiday 2026 — Advanced Track / Product
* **Team:** Reinforce SPG | Scaler School of Technology
* **Key Visual:** Hero visual of the interactive NetworkX / PyVis evidence graph with red contradiction edges.

### Slide 2: The Core Problem: The Black Box of Hallucinations
* **Points:**
  * LLMs hallucinate with high confidence.
  * Standard RAG treats all retrieved passages as equally true, leading to rumor vulnerability.
  * The **False Contradiction Trap (RefNLI)**: Naive keyword search retrieves entity-mismatched text and falsely labels true claims as refuted.
  * Lack of verifiable provenance: No graph connecting claims, sources, and conflicting findings.

### Slide 3: Our Solution: ProofGraph Architecture
* **Points:**
  * 4-Stage Claimify-inspired Decomposition (Propositions + Opinion Filtering).
  * FastEmbed Dense ONNX (`bge-small-en-v1.5`) + BM25Okapi + Reciprocal Rank Fusion ($k=60$).
  * TinyFish Live Search & Fetch Integration (api.search.tinyfish.ai & api.fetch.tinyfish.ai).
  * FIRE-inspired Dynamic Stopping & Iterative Multi-Hop Querying.
  * RefNLI Primary-Subject Alignment Gate & RuleStanceVerifier with Predicate-Bound Conflict Guards.
  * W3C PROV-DM Graph Builder with first-class contradiction edges.

### Slide 4: Diagnostic Validation & Empirical Baselines
* **Tables:**
  * **Retrieval Quality (Pooled AVeriTeC, N=60, 170 Passages):** Hits@1: 83.3%, Hits@5: 93.3%, MRR: 0.863.
  * **AVeriTeC Dev Split Sample (N=60 Claims):** ProofGraph (50.0% Acc / 0.463 Macro-F1) vs. Dense Baseline (41.7% Acc / 0.350 Macro-F1).
  * **Synthetic Context-Mismatch Stress Diagnostic (N=20):** ProofGraph 0.0% False Refutations (0/10) vs. 30.0% (3/10) when ablated.
* **Confusion Matrix:** Full 3x3 matrix showing calibrated abstention and support recognition.

### Slide 5: Interactive Evidence Visualizer & Audit Reports
* **Visuals:** Side-by-side screenshot of `outputs/graph.html` (interactive nodes & physics) and `outputs/report.md` (audit report with clickable source links).
* **Highlights:** First-class contradiction edges (`CONTRADICTS` in red, `DISAGREES_WITH` in dashed orange), instant offline export, zero Neo4j overhead.

### Slide 6: Engineering Rigor & Deliverables
* **Checklist:**
  * 15 Passing Unit & Integration Tests (`tests/test_pipeline.py`) with 14 offline unit tests and live TinySearch API testing.
  * Fully reproducible pinned environment (`requirements-lock.txt`) enforced in GitHub Actions matrix CI.
  * Full dataset card (`docs/DATA_CARD.md`) with explicit provenance and protocol disclosures.
  * Fully reproducible CLI (`python -m proofgraph.cli --text "..."`).
  * End-to-end live demo script (`python demo.py`).
  * Open-source repo under Apache-2.0: [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph).
