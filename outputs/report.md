# ProofGraph: Evidence-Backed Claim Review Report
*Generated on: 2026-11-11 10:48:19 UTC*

## 📊 Executive Summary
- **Total Atomic Claims Analyzed:** `3`
- **Supported (Entailed):** `1` (33.3%)
- **Contradicted (Refuted):** `1` (33.3%)
- **Insufficient Evidence / Conflicting:** `1` (33.3%)

🔍 **Interactive Knowledge Graph:** [View Interactive Evidence Graph](graph.html)

## 📜 Analyzed Document
> Apollo 11 successfully landed on the moon on July 20, 1969 with Neil Armstrong. The Eiffel Tower was constructed in Rome in 1950 by Italian engineers. Personally, I think Rome is the greatest architectural city in the world.

## 🔬 Claim-by-Claim Audit Trail
### Claim 1: ✅ `SUPPORTED` (Uncalibrated rule score: 95.0%)
**Assertion:** *"Apollo 11 successfully landed on the moon on July 20, 1969 with Neil Armstrong."*
**Identified Entities:** `Neil Armstrong`, `1969`, `July`, `Neil`, `Armstrong`
**Justification:** Supported by authoritative evidence (1 supporting sources vs 0 contradicting).

**Retrieved Evidence Passages:**
- **🟢 SUPPORTS** via [NASA Historical Archive](https://www.nasa.gov/mission/apollo-11/) *(Retrieval: `local` | Ranking signal: 0.97 (retriever ranking signal) | Source prior: 0.98)*
  > "The Apollo 11 mission landed humans on the Moon on July 20, 1969, with commander Neil Armstrong and lunar module pilot Buzz Aldrin."
- **⚪ MENTIONS** via [Encyclopaedia Britannica](https://www.britannica.com/topic/Eiffel-Tower-Paris-France) *(Retrieval: `local` | Ranking signal: 0.31 (retriever ranking signal) | Source prior: 0.92)*
  > "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. Constructed from 1887 to 1889 by Gustave Eiffel's company."
- **⚪ MENTIONS** via [History Channel](https://www.history.com/topics/ancient-rome/colosseum) *(Retrieval: `local` | Ranking signal: 0.27 (retriever ranking signal) | Source prior: 0.50)*
  > "The Colosseum in Rome was completed in 80 AD under Emperor Titus."

---
### Claim 2: ❌ `CONTRADICTED` (Uncalibrated rule score: 95.0%)
**Assertion:** *"The Eiffel Tower was constructed in Rome in 1950 by Italian engineers."*
**Identified Entities:** `The Eiffel Tower`, `1950`, `Eiffel`, `Tower`, `Rome`, `Italian`
**Justification:** Refuted by authoritative evidence (1 contradicting sources vs 0 supporting).

**Retrieved Evidence Passages:**
- **🔴 CONTRADICTS** via [Encyclopaedia Britannica](https://www.britannica.com/topic/Eiffel-Tower-Paris-France) *(Retrieval: `local` | Ranking signal: 0.89 (retriever ranking signal) | Source prior: 0.92)*
  > "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. Constructed from 1887 to 1889 by Gustave Eiffel's company."
- **⚪ MENTIONS** via [Conspiracy Chronicle](https://infowars.com/fake-eiffel-rome) *(Retrieval: `local` | Ranking signal: 0.80 (retriever ranking signal) | Source prior: 0.15)*
  > "Some modern architectural historians argue that a replica of the Eiffel Tower was secretly designed in Rome in 1950."
- **⚪ MENTIONS** via [NASA Historical Archive](https://www.nasa.gov/mission/apollo-11/) *(Retrieval: `local` | Ranking signal: 0.78 (retriever ranking signal) | Source prior: 0.98)*
  > "The Apollo 11 mission landed humans on the Moon on July 20, 1969, with commander Neil Armstrong and lunar module pilot Buzz Aldrin."

---
### Claim 3: ⚠️ `INSUFFICIENT_EVIDENCE` (Uncalibrated rule score: 100.0%)
**Assertion:** *"Personally, I think Rome is the greatest architectural city in the world."*
**Identified Entities:** `I`, `Rome`
**Justification:** Claim was classified as subjective or non-verifiable opinion.

**Retrieved Evidence Passages:**
- **⚪ MENTIONS** via [Conspiracy Chronicle](https://infowars.com/fake-eiffel-rome) *(Retrieval: `local` | Ranking signal: 0.81 (retriever ranking signal) | Source prior: 0.15)*
  > "Some modern architectural historians argue that a replica of the Eiffel Tower was secretly designed in Rome in 1950."
- **⚪ MENTIONS** via [Encyclopaedia Britannica](https://www.britannica.com/topic/Eiffel-Tower-Paris-France) *(Retrieval: `local` | Ranking signal: 0.72 (retriever ranking signal) | Source prior: 0.95)*
  > "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. Constructed from 1887 to 1889 by Gustave Eiffel's company."
- **⚪ MENTIONS** via [History Channel](https://www.history.com/topics/ancient-rome/colosseum) *(Retrieval: `local` | Ranking signal: 0.55 (retriever ranking signal) | Source prior: 0.85)*
  > "The Colosseum in Rome was completed in 80 AD under Emperor Titus."

---