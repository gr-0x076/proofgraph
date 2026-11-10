#!/usr/bin/env python3
"""
ProofGraph End-to-End Showcase Demo:
Demonstrates atomic claim decomposition, hybrid retrieval, RefNLI context alignment,
CONFACT source credibility weighting, and W3C PROV-DM knowledge graph generation
with first-class CONTRADICTION edges.
"""

from proofgraph.pipeline import ProofGraphPipeline
from proofgraph.schemas import EvidencePassage
import os

def run_demo():
    print("=" * 70)
    print("🛡️  PROOFGRAPH: AUTONOMOUS AI EVIDENCE INTELLIGENCE ENGINE — LIVE DEMO")
    print("=" * 70)

    # 1. Initialize Pipeline
    pipeline = ProofGraphPipeline()

    # 2. Seed a known local knowledge base with multi-source evidence (some conflicting!)
    corpus = [
        # Apollo 11 Evidence (Authoritative)
        EvidencePassage(
            text="The Apollo 11 mission landed humans on the Moon on July 20, 1969, with commander Neil Armstrong and lunar module pilot Buzz Aldrin.",
            source_title="NASA Historical Archive",
            source_domain="nasa.gov",
            source_url="https://www.nasa.gov/mission/apollo-11/",
            credibility_score=0.99
        ),
        # Eiffel Tower Ground Truth (Authoritative)
        EvidencePassage(
            text="The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. Constructed from 1887 to 1889 by Gustave Eiffel's company.",
            source_title="Encyclopaedia Britannica",
            source_domain="britannica.com",
            source_url="https://www.britannica.com/topic/Eiffel-Tower-Paris-France",
            credibility_score=0.95
        ),
        # Misleading Blog Post (Low Credibility - Rumor)
        EvidencePassage(
            text="Some modern architectural historians argue that a replica of the Eiffel Tower was secretly designed in Rome in 1950.",
            source_title="Conspiracy Chronicle",
            source_domain="infowars.com",
            source_url="https://infowars.com/fake-eiffel-rome",
            credibility_score=0.15
        ),
        # Irrelevant Context (To test RefNLI filter)
        EvidencePassage(
            text="The Colosseum in Rome was completed in 80 AD under Emperor Titus.",
            source_title="History Channel",
            source_domain="history.com",
            source_url="https://www.history.com/topics/ancient-rome/colosseum",
            credibility_score=0.85
        )
    ]
    pipeline.index_corpus(corpus)
    print(f"📚 Indexed {len(corpus)} local multi-source evidence passages into Hybrid Index.")

    # 3. Input Document containing True claims, False claims, and Subjective Opinions
    input_document = (
        "Apollo 11 successfully landed on the moon on July 20, 1969 with Neil Armstrong. "
        "The Eiffel Tower was constructed in Rome in 1950 by Italian engineers. "
        "Personally, I think Rome is the greatest architectural city in the world."
    )

    print("\n📄 Input Document to Verify:")
    print(f"\"{input_document}\"\n")

    output_dir = "outputs"
    result = pipeline.verify(
        document_text=input_document,
        output_dir=output_dir,
        top_k_evidence=3,
        use_live_web_fallback=False
    )

    print("-" * 70)
    print("🔬 VERIFICATION RESULTS & AUDIT TRAIL:")
    print("-" * 70)

    for idx, claim in enumerate(result.claims, 1):
        verdict = result.verdicts.get(claim.id)
        label = verdict.label.value if verdict else "UNVERIFIED"
        status_badge = "🟢 SUPPORTED" if label == "SUPPORTED" else ("🔴 CONTRADICTED" if label == "CONTRADICTED" else "🟡 INSUFFICIENT EVIDENCE")
        print(f"\n[Claim {idx}] {status_badge} (Confidence: {verdict.confidence * 100:.1f}%)")
        print(f"  • Assertion: \"{claim.text}\"")
        print(f"  • Verifiable: {claim.is_verifiable}")
        print(f"  • Justification: {verdict.justification}")
        
        passages = result.evidence_map.get(claim.id, [])
        for p in passages:
            is_supp = p.id in verdict.supporting_evidence_ids
            is_contra = p.id in verdict.contradicting_evidence_ids
            stance_badge = "SUPPORTS" if is_supp else ("CONTRADICTS" if is_contra else "FILTERED / MENTIONS")
            print(f"    - [{stance_badge}] {p.source_title} (cred: {p.credibility_score:.2f}, rel: {p.relevance_score:.2f})")
            print(f"      \"{p.text[:90]}...\"")

    print("\n" + "=" * 70)
    print("🌐 KNOWLEDGE GRAPH & REPORT ARTIFACTS GENERATED:")
    print(f"  • Interactive Graph Visualizer: {os.path.abspath(result.graph_html_path)}")
    print(f"  • Full Audit Markdown Report:   {os.path.abspath(os.path.join(output_dir, 'report.md'))}")
    print(f"  • Total Graph Nodes: {result.graph.number_of_nodes()} | Edges: {result.graph.number_of_edges()}")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
