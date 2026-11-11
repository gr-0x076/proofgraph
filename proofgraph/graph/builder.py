import networkx as nx
from typing import List, Dict, Any
from proofgraph.schemas.claim import AtomicClaim
from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.verdict import ClaimVerdict, VeracityLabel

class ProofGraphBuilder:
    """
    Constructs a NetworkX provenance graph using PROV-DM-inspired entity/activity relations.
    Highlights first-class CONTRADICTION and DISAGREEMENT edges.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(
        self,
        claims: List[AtomicClaim],
        evidence_map: Dict[str, List[EvidencePassage]],
        verdicts: Dict[str, ClaimVerdict]
    ) -> nx.DiGraph:
        self.graph.clear()

        # Track sources to add unique source nodes
        sources_seen = set()

        for claim in claims:
            verdict = verdicts.get(claim.id)
            label = verdict.label.value if verdict else "UNVERIFIED"
            conf = verdict.confidence if verdict else 0.0

            # Color coding for claim nodes
            color = "#10b981" if label == "SUPPORTED" else ("#ef4444" if label == "CONTRADICTED" else "#f59e0b")
            
            # 1. Add Claim Node
            self.graph.add_node(
                claim.id,
                node_type="claim",
                title=f"Claim: {claim.text[:60]}...",
                full_text=claim.text,
                label=f"[{label}]\n{claim.text[:35]}...",
                veracity=label,
                confidence=conf,
                color=color,
                shape="box",
                size=25,
                borderWidth=2
            )

            passages = evidence_map.get(claim.id, [])
            for ev in passages:
                # 2. Add Source Node
                source_id = f"src_{ev.source_domain.replace('.', '_')}"
                if source_id not in sources_seen:
                    sources_seen.add(source_id)
                    self.graph.add_node(
                        source_id,
                        node_type="source",
                        title=f"Source: {ev.source_title} ({ev.source_domain})\nCredibility: {ev.credibility_score:.2f}",
                        label=f"🌐 {ev.source_title}",
                        domain=ev.source_domain,
                        credibility=ev.credibility_score,
                        color="#8b5cf6",  # Purple
                        shape="ellipse",
                        size=20
                    )

                # 3. Add Evidence Node
                ev_color = "#3b82f6"  # Blue
                self.graph.add_node(
                    ev.id,
                    node_type="evidence",
                    title=f"Evidence from {ev.source_title}:\n\"{ev.text}\"\nRelevance: {ev.relevance_score:.2f} | Credibility: {ev.credibility_score:.2f}",
                    label=f"📄 {ev.text[:30]}...",
                    full_text=ev.text,
                    source_url=ev.source_url or "",
                    relevance=ev.relevance_score,
                    credibility=ev.credibility_score,
                    color=ev_color,
                    shape="dot",
                    size=15
                )

                # Link Evidence -> Source
                self.graph.add_edge(ev.id, source_id, relation="wasAttributedTo", color="#a78bfa", dashes=True)

                # 4. Add Stance / Provenance Edges
                if verdict and ev.id in verdict.supporting_evidence_ids:
                    self.graph.add_edge(
                        ev.id,
                        claim.id,
                        relation="SUPPORTS",
                        color="#10b981",  # Emerald Green
                        width=3,
                        title=f"SUPPORTS: {ev.source_title}"
                    )
                elif verdict and ev.id in verdict.contradicting_evidence_ids:
                    # FIRST-CLASS CONTRADICTION EDGE (Bold Red)
                    self.graph.add_edge(
                        ev.id,
                        claim.id,
                        relation="CONTRADICTS",
                        color="#ef4444",  # Crimson Red
                        width=4,
                        title=f"CONTRADICTS: {ev.source_title} actively refutes this assertion"
                    )
                else:
                    self.graph.add_edge(
                        ev.id,
                        claim.id,
                        relation="MENTIONS",
                        color="#9ca3af",  # Gray
                        width=1,
                        dashes=True,
                        title="MENTIONS: Context retrieved but does not directly entail"
                    )

            # 5. Detect Cross-Source Disagreements for this Claim
            if verdict and verdict.supporting_evidence_ids and verdict.contradicting_evidence_ids:
                # Add explicit DISAGREES_WITH edges between conflicting sources
                for s_id in verdict.supporting_evidence_ids:
                    for c_id in verdict.contradicting_evidence_ids:
                        self.graph.add_edge(
                            s_id,
                            c_id,
                            relation="DISAGREES_WITH",
                            color="#f97316",  # Orange alert
                            width=2,
                            dashes=True,
                            title="DISAGREEMENT: These evidence items make mutually contradictory assertions"
                        )

        return self.graph
