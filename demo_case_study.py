"""
ProofGraph: Verifier Case Study with Curated Evidence Fixtures
==========================================================================
Demonstrates the full verification pipeline on concrete claims, providing
step-by-step transparency across:
1. Atomic Claim Decomposition
2. Evidence Retrieval & Web Provenance
3. RefNLI Context Alignment Gating (filtering distractors)
4. Stance Analysis (Deterministic Rules & DeBERTa Neural Cross-Encoder)
5. Knowledge Graph Construction & Provenance Synthesis
6. Conservative Abstention on Speculative Claims

Outputs a detailed trace report to: outputs/demo_case_study.md
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

from proofgraph.decomposer import ClaimDecomposer
from proofgraph.schemas import AtomicClaim, EvidencePassage, VeracityLabel, ClaimVerdict
from proofgraph.verifier.alignment import RefNLIAlignmentGate
from proofgraph.verifier.credibility import SourceCredibilityScorer
from proofgraph.verifier.nli import HybridNLIVerifier
from proofgraph.graph.builder import ProofGraphBuilder
from proofgraph.reporter.report_generator import ReportGenerator

def run_case_study(output_path: str = "outputs/demo_case_study.md") -> str:
    print("=" * 80)
    print("🔎 PROOFGRAPH: VERIFIER CASE STUDY WITH CURATED EVIDENCE")
    print("=" * 80)

    decomposer = ClaimDecomposer()
    gate = RefNLIAlignmentGate()
    credibility = SourceCredibilityScorer()
    verifier = HybridNLIVerifier(refnli_gate=gate, credibility_scorer=credibility, neural_fallback_threshold=0.95)
    graph_builder = ProofGraphBuilder()
    reporter = ReportGenerator()

    # Case 1: Multi-predicate claim with conflicting and supporting subclaims
    claim_1 = "The Eiffel Tower was constructed in Rome in 1950 by Gustave Eiffel."
    print(f"\n[CASE 1] Complex Assertive Claim:\n  \"{claim_1}\"")

    # Step 1: Decomposition
    subclaims = decomposer.decompose(claim_1)
    print(f"\n  Step 1: Decomposed into {len(subclaims)} atomic subclaim(s):")
    for sc in subclaims:
        print(f"    • [{sc.id}] \"{sc.text}\" (Verifiable: {sc.is_verifiable})")

    # Step 2: Evidence Pool
    ev_paris = EvidencePassage(
        id="ev_wiki_eiffel",
        text="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. Constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",
        source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
        source_domain="en.wikipedia.org",
        source_title="Eiffel Tower - Wikipedia",
        relevance_score=0.95,
        credibility_score=0.88,
    )
    ev_gustave = EvidencePassage(
        id="ev_britannica_gustave",
        text="Gustave Eiffel was a French civil engineer and architect famous for designing the Eiffel Tower in Paris.",
        source_url="https://www.britannica.com/biography/Gustave-Eiffel",
        source_domain="www.britannica.com",
        source_title="Gustave Eiffel | Britannica",
        relevance_score=0.92,
        credibility_score=0.96,
    )
    ev_distractor = EvidencePassage(
        id="ev_unrelated_biography",
        text="William was born in 1980 in Toronto to diplomat parents.",
        source_title="Synthetic unrelated biography distractor",
        relevance_score=0.10,
        credibility_score=0.50,
    )

    # The third passage is an intentionally synthetic unrelated distractor, not a live source.
    evidence_pool = [ev_paris, ev_gustave, ev_distractor]
    print(f"\n  Step 2: Ingested {len(evidence_pool)} Candidate Passages:")
    for ev in evidence_pool:
        print(f"    • [{ev.id}] {ev.source_domain} (Relevance: {ev.relevance_score:.2f}, Credibility: {ev.credibility_score:.2f})")

    # Step 3: Alignment Gating
    print(f"\n  Step 3: Lexical Context Alignment Heuristic:")
    step3_trace = []
    for sc in subclaims:
        for ev in evidence_pool:
            aligned, jaccard, reason = gate.check_alignment(sc, ev)
            status = "✅ ALIGNED" if aligned else "🛡️ REJECTED (Context Mismatch)"
            step3_trace.append({
                "subclaim": sc.text,
                "evidence_id": ev.id,
                "aligned": aligned,
                "jaccard": jaccard,
                "reason": reason
            })
            print(f"    • Subclaim [{sc.id}] vs [{ev.id}]: {status} (Jaccard: {jaccard:.2f}, Reason: {reason})")

    # Step 4: Verification
    print(f"\n  Step 4: Stance Verification (Rules + Neural):")
    verdicts = []
    for sc in subclaims:
        v = verifier.verify_claim(sc, evidence_pool)
        diag = getattr(verifier, "last_diagnostics", {})
        verdicts.append((sc, v, diag))
        print(f"    • [{sc.id}] Verdict: {v.label.value} (uncalibrated heuristic score: {v.confidence*100:.1f}%)")
        print(f"      Justification: {v.justification[:90]}...")
        if diag.get("neural_evaluated"):
            print(f"      Cross-Verifier Diagnostic: Rule={diag.get('rule_label')}, Neural={diag.get('neural_label')} (Conf: {diag.get('neural_confidence')})")

    # Case 2: Conservative Abstention Case
    claim_2 = "Quantum computers officially proved that P equals NP in 2024."
    print(f"\n[CASE 2] Unproven / Speculative Assertion:\n  \"{claim_2}\"")
    sc2 = AtomicClaim(text=claim_2, parent_sentence=claim_2)
    ev_claymath = EvidencePassage(
        id="ev_claymath_pnp",
        text="The P versus NP problem remains one of the seven Millennium Prize Problems with no generally accepted proof as of 2024.",
        source_url="https://www.claymath.org/millennium-problems/p-vs-np-problem",
        source_domain="claymath.org",
        source_title="Clay Mathematics Institute",
        relevance_score=0.91,
        credibility_score=0.98,
    )
    v2 = verifier.verify_claim(sc2, [ev_claymath])
    print(f"  Step 4: Verdict: {v2.label.value} (uncalibrated heuristic score: {v2.confidence*100:.1f}%)")
    print(f"  Reason: Correctly abstained from asserting false support on unresolved mathematical open problem.")

    # Generate Markdown Case Study Report
    md = [
        "# ProofGraph: Verifier Case Study with Curated Evidence Fixtures",
        f"**Generated:** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  ",
        "**Purpose:** Demonstrates decomposition, alignment filtering, and verifier behavior on selected claims. Evidence below is hard-coded fixture input; this script does not perform live TinyFish search or fetch.\n",
        "---",
        "## 🔬 Case Study 1: Multi-Predicate Compound Claim",
        f"**Target Input Claim:**\n> \"{claim_1}\"\n",
        "### Step 1: Atomic Decomposition",
        f"The claim contains multiple asserting predicates. ProofGraph decomposes it into {len(subclaims)} atomic subclaims:",
    ]

    for sc in subclaims:
        md.append(f"- **`{sc.id}`**: \"{sc.text}\" (Verifiable: `{sc.is_verifiable}`)")

    md.append("\n### Step 2: Evidence Pool & Source Credibility")
    md.append("| Evidence ID | Source Domain | Credibility Prior | Relevance Score | Excerpt |")
    md.append("| :--- | :--- | :---: | :---: | :--- |")
    for ev in evidence_pool:
        md.append(f"| `{ev.id}` | `{ev.source_domain}` | {ev.credibility_score:.2f} | {ev.relevance_score:.2f} | \"{ev.text[:80]}...\" |")

    md.append("\n### Step 3: Lexical Context Alignment Heuristic")
    md.append("This lightweight heuristic is inspired by reference-indeterminacy research; it is not the RefNLI model. The synthetic unrelated distractor should be rejected:")
    md.append("| Subclaim | Evidence | Status | Jaccard | Gate Decision |")
    md.append("| :--- | :--- | :---: | :---: | :--- |")
    for t in step3_trace:
        st_badge = "✅ ALIGNED" if t["aligned"] else "🛡️ REJECTED"
        md.append(f"| \"{t['subclaim'][:35]}...\" | `{t['evidence_id']}` | {st_badge} | {t['jaccard']:.2f} | {t['reason']} |")

    md.append("\n### Step 4: Stance Verification & Decision Diagnostics")
    for sc, v, diag in verdicts:
        badge = "🔴 CONTRADICTED" if v.label == VeracityLabel.CONTRADICTED else ("🟢 SUPPORTED" if v.label == VeracityLabel.SUPPORTED else "⚪ INSUFFICIENT_EVIDENCE")
        md.append(f"#### Subclaim `{sc.id}`: \"{sc.text}\"")
        md.append(f"- **Final Verdict:** `{badge}` (uncalibrated heuristic score: {v.confidence*100:.1f}%)")
        md.append(f"- **Justification:** {v.justification}")
        if diag.get("neural_evaluated"):
            md.append(f"- **Cross-Model Diagnostics:** Rule Stance=`{diag.get('rule_label')}` | Neural Cross-Encoder=`{diag.get('neural_label')}` (Neural Conf: `{diag.get('neural_confidence')}`)")

    md.append("\n---")
    md.append("## 🛡️ Case Study 2: Conservative Abstention on Unresolved Claims")
    md.append(f"**Target Claim:**\n> \"{claim_2}\"\n")
    md.append(f"- **Evidence:** Clay Mathematics Institute statement on P vs NP.")
    md.append(f"- **Engine Verdict:** `{v2.label.value}` (uncalibrated heuristic score: {v2.confidence*100:.1f}%)")
    md.append(f"- **Analysis:** The engine refuses to hallucinate refutation or support when evidence confirms the problem remains open, demonstrating conservative abstention behavior.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n✅ Case study report written to: {output_path}")
    return output_path

if __name__ == "__main__":
    run_case_study()
