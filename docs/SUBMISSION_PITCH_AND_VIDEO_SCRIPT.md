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
  > "Retrieval-augmented answers can still leave a reviewer with a practical problem: which evidence supports each claim, which sources disagree, and where did the verdict come from? Context-mismatched evidence can also mislead NLI-style verification systems, as recent research documents.
  >
  > Welcome to **ProofGraph**, a claim-review prototype. It splits input into reviewable statements, retrieves and displays evidence, applies transparent rule-based checks, and builds an interactive provenance graph. It is designed to expose uncertainty and evidence paths, not replace expert fact-checking."

---

### [00:30 – 01:15] System Architecture & Deep Tech Innovations
* **Visual:** Flow diagram from README showing the 5-stage pipeline with real code snippets.
* **Speaker:**
  > "ProofGraph uses five stages, combining established ideas with lightweight prototype implementations:
  > 
  > 1. **Claim splitting:** A heuristic splitter inspired by Claimify identifies candidate statements; it is not a reproduction of Claimify.
  > 2. **Retrieval:** Local mode uses BM25 and FastEmbed with reciprocal-rank fusion. Optional TinyFish Search discovers pages and TinyFish Fetch extracts page text. A FIRE-inspired heuristic can request another query when entity coverage is low.
  > 3. **Rule-based review:** A lexical alignment filter and numeric, location, and attribution rules assign support, contradiction, or insufficient evidence. These are guardrails, not a trained NLI model.
  > 4. **Source prior:** A static domain heuristic influences evidence weighting; it is inspired by source-credibility research, not the CONFACT method.
  > 5. **Provenance graph:** NetworkX and PyVis visualize relationships in a PROV-DM-inspired schema; formal W3C conformance has not been established."

---

### [01:15 – 02:00] Offline Fixture Demonstration (`demo.py`)
* **Visual:** Screen capture running `python demo.py`, terminal output, then opening `outputs/graph.html` in the browser. State clearly that this demo uses bundled evidence fixtures; it does not make live web requests.
* **Speaker:**
  > "Let's run the offline end-to-end fixture demo:
  >
  > Input: *'Apollo 11 landed on the moon on July 20, 1969 with Neil Armstrong. The Eiffel Tower was constructed in Rome in 1950 by Italian engineers. Personally, I think Rome is the greatest architectural city in the world.'*
  >
  > In this local run:
  > - Claim 1 is marked **🟢 SUPPORTED** by a bundled passage labeled NASA Historical Archive.
  > - Claim 2 is marked **🔴 CONTRADICTED** by a bundled Encyclopaedia Britannica passage, with location and date conflicts.
  > - Claim 3 is identified as a subjective opinion and assigned **🟡 INSUFFICIENT EVIDENCE**.
  > The displayed percentage is an uncalibrated rule score, not a probability of correctness.
  >
  > In `outputs/graph.html`: we can drag nodes, inspect evidence passages, follow green support paths, and trace bold red contradiction edges linking conflicting claims to their disproving sources."

---

### [02:00 – 02:35] Empirical Validation across 4 Diagnostic Suites
* **Visual:** Slide showing the AVeriTeC Dev Split confusion matrix, retrieval IR table, and synthetic context-mismatch stress test.
* **Speaker:**
  > "We evaluated ProofGraph across four complementary dimensions:
  >
  > 1. **In-Batch Evidence Retrieval Quality Diagnostic (Pooled AVeriTeC, 170 Passages):** On a global index with cross-claim distractors, our hybrid retriever achieved **93.3% Hits@5** and an **MRR of 0.863**.
  > 2. **AVeriTeC development sample (N=60):** Given claim-specific candidate evidence pools, the current run scored **50.0% accuracy and 0.463 macro-F1** under our custom three-way mapping. This is not the official AVeriTeC shared-task score; the dense baseline scored 41.7% accuracy and 0.350 macro-F1.
  > 3. **Synthetic context-mismatch diagnostic (N=20):** On hand-authored examples, the alignment gate yielded 0/10 false refutations versus 3/10 without it. This is not RefNLI evaluation or evidence of generalization.
  > 4. **Sanity suite:** The hand-authored diagnostic suite scores 90.0% accuracy. Seventeen offline tests pass; the opt-in live TinyFish Search→Fetch test also passed separately in the local checkout. It was not part of the offline accuracy metrics."

---

### [02:35 – 03:00] Impact, Future Roadmap & Conclusion
* **Visual:** Slide showing GitHub repository, reproducible commands, and team credits.
* **Speaker:**
  > "ProofGraph is open-source under Apache-2.0 for its code, uses pinned direct dependencies in CI, and generates Markdown reports and interactive provenance graphs. The current verifier is a prototype; our next work is to evaluate on public benchmarks and improve contradiction detection.
  >
  > We are Team Reinforce SPG, and ProofGraph brings transparency, accountability, and evidence graph intelligence to claim verification. Thank you!"

---

## 📊 Part 2: Slide-by-Slide Pitch Deck Structure (Canva Presentation)

### Slide 1: Title & Team
* **Header:** ProofGraph: Evidence-Backed Claim Review Prototype
* **Subheader:** Hack on Holiday 2026 — Advanced Track / Product
* **Team:** Reinforce SPG | Scaler School of Technology
* **Key Visual:** Hero visual of the interactive NetworkX / PyVis evidence graph with red contradiction edges.

### Slide 2: The Core Problem: The Black Box of Hallucinations
* **Points:**
  * LLMs hallucinate with high confidence.
  * Standard RAG treats all retrieved passages as equally true, leading to rumor vulnerability.
  * RefNLI research documents context-mismatch errors in NLI systems; ProofGraph currently tests a small synthetic version of this failure mode.
  * Reviewers need claim-level evidence links and a clear record of source disagreement; ProofGraph makes these relationships inspectable in a graph.

### Slide 3: Our Solution: ProofGraph Architecture
* **Points:**
  * Heuristic Claimify-inspired claim splitting and opinion filtering.
  * FastEmbed Dense ONNX (`bge-small-en-v1.5`) + BM25Okapi + Reciprocal Rank Fusion ($k=60$).
  * TinyFish Live Search & Fetch Integration (api.search.tinyfish.ai & api.fetch.tinyfish.ai).
  * FIRE-inspired Dynamic Stopping & Iterative Multi-Hop Querying.
  * RefNLI-inspired lexical subject-alignment gate and rule-based conflict checks.
  * PROV-DM-inspired provenance visualization with contradiction edges.

### Slide 4: Diagnostic Validation & Empirical Baselines
* **Tables:**
  * **Retrieval Quality (Pooled AVeriTeC, N=60, 170 Passages):** Hits@1: 83.3%, Hits@5: 93.3%, MRR: 0.863.
  * **AVeriTeC Dev Split Sample (N=60 Claims):** ProofGraph (50.0% Acc / 0.463 Macro-F1) vs. Dense Baseline (41.7% Acc / 0.350 Macro-F1).
  * **Synthetic Context-Mismatch Stress Diagnostic (N=20):** 0/10 false refutations with the gate vs. 3/10 without it (hand-authored examples, not RefNLI).
  * **Confusion Matrix:** Full 3x3 matrix showing the current rule-based verdict distribution, including its false-supported and false-contradicted errors.

### Slide 5: Interactive Evidence Visualizer & Audit Reports
* **Visuals:** Side-by-side screenshot of `outputs/graph.html` (interactive nodes & physics) and `outputs/report.md` (audit report with clickable source links).
* **Highlights:** First-class contradiction edges (`CONTRADICTS` in red, `DISAGREES_WITH` in dashed orange), instant offline export, zero Neo4j overhead.

### Slide 6: Engineering Rigor & Deliverables
* **Checklist:**
  * 17 offline tests passing locally; opt-in live TinyFish Search→Fetch smoke test passed separately. It is skipped in the default suite unless both `TINYFISH_API_KEY` and `TINYFISH_LIVE_TEST=1` are set.
  * Pinned direct dependencies (`requirements-lock.txt`) enforced in GitHub Actions matrix CI; transitive dependencies are not hash-locked.
  * Full dataset card (`docs/DATA_CARD.md`) with explicit provenance and protocol disclosures.
  * CLI (`python -m proofgraph.cli --text "..."`) with offline mode and optional live retrieval.
  * Offline fixture demo (`python demo.py`); live Search→Fetch is available through the CLI when the API key and network are configured.
  * Open-source repo under Apache-2.0: [https://github.com/gr-0x076/proofgraph](https://github.com/gr-0x076/proofgraph).
