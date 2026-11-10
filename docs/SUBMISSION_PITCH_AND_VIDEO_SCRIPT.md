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
  > "Every modern AI system retrieves documents and generates confident answers. But when LLMs hallucinate or sources contradict each other, standard RAG collapses into a black box. Even worse, current systems suffer from the **False Contradiction Trap**—flagging claims as refuted simply because a retrieved document discusses someone else!
  >
  > Welcome to **ProofGraph**: an autonomous AI evidence intelligence engine that doesn't just guess verdicts—it decomposes claims, evaluates source credibility, resolves conflicting evidence, and builds an interactive, auditable provenance graph with first-class contradiction edges."

---

### [00:30 – 01:15] System Architecture & Deep Tech Innovations
* **Visual:** Flow diagram from README showing the 5-stage pipeline with real code snippets.
* **Speaker:**
  > "ProofGraph solves verification through five rigorous stages:
  > 
  > 1. **Claimify Decomposition:** It splits multi-assertion text into atomic, decontextualized propositions while filtering out non-verifiable opinions.
  > 2. **Hybrid Retrieval with FIRE Dynamic Stopping:** Combining BM25 lexical search with FastEmbed ONNX dense embeddings via Reciprocal Rank Fusion ($k=60$). Guided by NAACL 2025's FIRE policy, it analyzes entity coverage in real-time, executing multi-hop queries only when critical entities are missing.
  > 3. **RefNLI Context Alignment Gate:** Before evaluating stance, an entity-overlap gate blocks reference mismatches, completely eliminating false contradictions.
  > 4. **CONFACT Source Credibility Prior:** Sources aren't treated equally. High-authority institutions like NASA, Reuters, and peer-reviewed journals carry up to $6.5\times$ the voting mass of unverified blogs, protected by canonical hostname spoof-guarding.
  > 5. **W3C PROV-DM Graph Construction:** Exporting auditable knowledge graphs with color-coded nodes and explicit red contradiction physics."

---

### [01:15 – 02:00] Live Demonstration (`demo.py`)
* **Visual:** Screen capture running `python demo.py`, terminal output, then opening `outputs/graph.html` in the browser.
* **Speaker:**
  > "Let's run the end-to-end demo live:
  >
  > Input: *'Apollo 11 landed on the moon on July 20, 1969 with Neil Armstrong. The Eiffel Tower was constructed in Rome in 1950 by Italian engineers. Personally, I think Rome is the greatest architectural city in the world.'*
  >
  > In less than 2 seconds without requiring heavy GPUs:
  > - Claim 1 is marked **🟢 SUPPORTED** with 95% confidence via NASA Archives.
  > - Claim 2 is marked **🔴 CONTRADICTED** with 95% confidence via Encyclopaedia Britannica, exposing both the temporal and geographical conflicts.
  > - Claim 3 is flagged as **🟡 INSUFFICIENT EVIDENCE (Opinion)**, safely abstaining.
  >
  > Now look at the interactive graph in `outputs/graph.html`: we can drag nodes, inspect evidence passages, follow green support paths, and trace bold red contradiction edges linking conflicting claims to their disproving sources."

---

### [02:00 – 02:35] Empirical Validation on AVeriTeC Benchmark
* **Visual:** Slide showing the AVeriTeC Dev Split confusion matrix and ablation table.
* **Speaker:**
  > "We didn't rely on synthetic toy diagnostics. We evaluated ProofGraph against the public **AVeriTeC Dev Split (NeurIPS 2023)**—60 real-world claims fact-checked by 50 professional organizations:
  >
  > - ProofGraph achieved **48.3% Accuracy** and **0.455 Macro-F1**, outperforming the standard naive keyword RAG baseline (33.3% Accuracy, 0.336 Macro-F1) by **+15.0% absolute accuracy**.
  > - Supported claims reached an **F1 score of 0.632** with 80% recall.
  > - In our unit diagnostic suite, ProofGraph eliminated 100% of false contradictions caused by reference indeterminacy, scoring 90% vs 50% for naive RAG.
  > - Our complete pytest suite has 8/8 passing tests validating all pipeline components."

---

### [02:35 – 03:00] Impact, Future Roadmap & Conclusion
* **Visual:** Slide showing GitHub repository, reproducible commands, and team credits.
* **Speaker:**
  > "ProofGraph is fully open-source, W3C PROV-DM compliant, runs on CPU in milliseconds via ONNX, and provides complete auditable Markdown reports with inline citations.
  >
  > We are Team Reinforce SPG, and ProofGraph brings transparency, accountability, and mathematical rigor to automated fact-checking. Thank you!"

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
  * 4-Stage Claimify Decomposition (Propositions + Opinion Filtering).
  * FastEmbed Dense ONNX (`bge-small-en-v1.5`) + BM25Okapi + Reciprocal Rank Fusion ($k=60$).
  * FIRE Dynamic Stopping & Iterative Multi-Hop Querying.
  * RefNLI Alignment Gate & Canonical Host Spoof-Hardened Credibility Scorer.
  * W3C PROV-DM Graph Builder with first-class contradiction edges.

### Slide 4: Real-World Public Benchmark (AVeriTeC Dev Split)
* **Table:** ProofGraph (48.3% Acc / 0.455 Macro-F1) vs. Naive Baseline (33.3% Acc / 0.336 Macro-F1).
* **Metrics:** +15.0% absolute accuracy boost; Supported F1 = 0.632; Contradicted Precision = 0.231; Insufficient Ev. F1 = 0.519.
* **Confusion Matrix:** Full 3x3 matrix showing accurate abstention and support recognition.

### Slide 5: Interactive Evidence Visualizer & Audit Reports
* **Visuals:** Side-by-side screenshot of `outputs/graph.html` (interactive nodes & physics) and `outputs/report.md` (audit report with clickable source links).
* **Highlights:** First-class contradiction edges (`CONTRADICTS` in red, `DISAGREES_WITH` in dashed orange), instant offline export, zero Neo4j overhead.

### Slide 6: Engineering Rigor & Deliverables
* **Checklist:**
  * 8/8 Passing Unit & Integration Tests (`tests/test_pipeline.py`).
  * Fully reproducible CLI (`python -m proofgraph.cli --text "..."`).
  * End-to-end live demo script (`python demo.py`).
  * Standalone ONNX runtime: zero PyTorch dependency, sub-50ms inference.
  * Open-source repo: [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph).
