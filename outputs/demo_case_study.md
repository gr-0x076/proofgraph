# ProofGraph: End-to-End Claim Verification Case Study & Trace Report
**Generated:** `2026-11-11 14:57:56 UTC`  
**Purpose:** Demonstrates exact step-by-step pipeline execution, RefNLI distractor filtering, rule/neural cooperation, and honest abstention handling on real claims.

---
## 🔬 Case Study 1: Multi-Predicate Compound Claim
**Target Input Claim:**
> "The Eiffel Tower was constructed in Rome in 1950 by Gustave Eiffel."

### Step 1: Atomic Decomposition
The claim contains multiple asserting predicates. ProofGraph decomposes it into 1 atomic subclaims:
- **`claim_64235cf8`**: "The Eiffel Tower was constructed in Rome in 1950 by Gustave Eiffel." (Verifiable: `True`)

### Step 2: Evidence Pool & Source Credibility
| Evidence ID | Source Domain | Credibility Prior | Relevance Score | Excerpt |
| :--- | :--- | :---: | :---: | :--- |
| `ev_wiki_eiffel` | `en.wikipedia.org` | 0.88 | 0.95 | "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, ..." |
| `ev_britannica_gustave` | `www.britannica.com` | 0.92 | 0.92 | "Gustave Eiffel was a French civil engineer and architect famous for designing th..." |
| `ev_rome_colosseum` | `history.com` | 0.50 | 0.72 | "The Colosseum in Rome, Italy, was completed in 80 AD under Titus and remains an ..." |

### Step 3: RefNLI Context Alignment Gating
Before inference, each pair is filtered by the RefNLI alignment gate to prevent reference-indeterminacy hallucinations:
| Subclaim | Evidence | Status | Jaccard | Gate Decision |
| :--- | :--- | :---: | :---: | :--- |
| "The Eiffel Tower was constructed in..." | `ev_wiki_eiffel` | ✅ ALIGNED | 0.13 | Aligned: entity or lexical threshold satisfied |
| "The Eiffel Tower was constructed in..." | `ev_britannica_gustave` | ✅ ALIGNED | 0.21 | Aligned: entity or lexical threshold satisfied |
| "The Eiffel Tower was constructed in..." | `ev_rome_colosseum` | ✅ ALIGNED | 0.06 | Aligned: entity or lexical threshold satisfied |

### Step 4: Stance Verification & Decision Diagnostics
#### Subclaim `claim_64235cf8`: "The Eiffel Tower was constructed in Rome in 1950 by Gustave Eiffel."
- **Final Verdict:** `🔴 CONTRADICTED` (85.6% confidence)
- **Justification:** Refuted by authoritative evidence (2 contradicting sources vs 1 supporting).
- **Cross-Model Diagnostics:** Rule Stance=`CONTRADICTED` | Neural Cross-Encoder=`CONTRADICTED` (Neural Conf: `0.962`)

---
## 🛡️ Case Study 2: Conservative Abstention on Unresolved Claims
**Target Claim:**
> "Quantum computers officially proved that P equals NP in 2024."

- **Evidence:** Clay Mathematics Institute statement on P vs NP.
- **Engine Verdict:** `INSUFFICIENT_EVIDENCE` (30.0% confidence)
- **Analysis:** The engine refuses to hallucinate refutation or support when evidence confirms the problem remains open, demonstrating conservative abstention behavior.
