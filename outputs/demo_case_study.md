# ProofGraph: Verifier Case Study with Curated Evidence Fixtures
**Generated:** `2026-11-11 16:24:27 UTC`
**Purpose:** Demonstrates decomposition, alignment filtering, and verifier behavior on selected claims. Evidence below is hard-coded fixture input; this script does not perform live TinyFish search or fetch.

---
## 🔬 Case Study 1: Multi-Predicate Compound Claim
**Target Input Claim:**
> "The Eiffel Tower was constructed in Rome in 1950 by Gustave Eiffel."

### Step 1: Atomic Decomposition
The claim contains multiple asserting predicates. ProofGraph decomposes it into 1 atomic subclaims:
- **`claim_f397a567`**: "The Eiffel Tower was constructed in Rome in 1950 by Gustave Eiffel." (Verifiable: `True`)

### Step 2: Evidence Pool & Source Credibility
| Evidence ID | Source Domain | Credibility Prior | Relevance Score | Excerpt |
| :--- | :--- | :---: | :---: | :--- |
| `ev_wiki_eiffel` | `en.wikipedia.org` | 0.88 | 0.95 | "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, ..." |
| `ev_britannica_gustave` | `www.britannica.com` | 0.92 | 0.92 | "Gustave Eiffel was a French civil engineer and architect famous for designing th..." |
| `ev_unrelated_biography` | `unknown` | 0.50 | 0.10 | "William was born in 1980 in Toronto to diplomat parents...." |

### Step 3: Lexical Context Alignment Heuristic
This lightweight heuristic is inspired by reference-indeterminacy research; it is not the RefNLI model. The synthetic unrelated distractor should be rejected:
| Subclaim | Evidence | Status | Jaccard | Gate Decision |
| :--- | :--- | :---: | :---: | :--- |
| "The Eiffel Tower was constructed in..." | `ev_wiki_eiffel` | ✅ ALIGNED | 0.13 | Aligned: primary subject or lexical overlap threshold satisfied |
| "The Eiffel Tower was constructed in..." | `ev_britannica_gustave` | ✅ ALIGNED | 0.21 | Aligned: primary subject or lexical overlap threshold satisfied |
| "The Eiffel Tower was constructed in..." | `ev_unrelated_biography` | 🛡️ REJECTED | 0.00 | Context Mismatch: Primary subject 'The Eiffel Tower' absent in evidence (RefNLI Reference Indeterminacy) |

### Step 4: Stance Verification & Decision Diagnostics
#### Subclaim `claim_f397a567`: "The Eiffel Tower was constructed in Rome in 1950 by Gustave Eiffel."
- **Final Verdict:** `🔴 CONTRADICTED` (uncalibrated heuristic score: 95.0%)
- **Justification:** Refuted by authoritative evidence (2 contradicting sources vs 0 supporting).
- **Cross-Model Diagnostics:** Rule Stance=`CONTRADICTED` | Neural Cross-Encoder=`CONTRADICTED` (Neural Conf: `0.959`)

---
## 🛡️ Case Study 2: Conservative Abstention on Unresolved Claims
**Target Claim:**
> "Quantum computers officially proved that P equals NP in 2024."

- **Evidence:** Clay Mathematics Institute statement on P vs NP.
- **Engine Verdict:** `INSUFFICIENT_EVIDENCE` (uncalibrated heuristic score: 30.0%)
- **Analysis:** The engine refuses to hallucinate refutation or support when evidence confirms the problem remains open, demonstrating conservative abstention behavior.
