from typing import List, Dict, Optional, Any
import os

from proofgraph.schemas.claim import AtomicClaim
from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.verdict import ClaimVerdict
from proofgraph.decomposer.decomposer import ClaimDecomposer
from proofgraph.retriever.hybrid_retriever import HybridRetriever
from proofgraph.verifier.nli import NLIVerifier
from proofgraph.graph.builder import ProofGraphBuilder
from proofgraph.graph.visualizer import ProofGraphVisualizer
from proofgraph.reporter.report_generator import ReportGenerator

class ProofGraphResult:
    def __init__(
        self,
        input_text: str,
        claims: List[AtomicClaim],
        evidence_map: Dict[str, List[EvidencePassage]],
        verdicts: Dict[str, ClaimVerdict],
        graph,
        report_markdown: str,
        graph_html_path: Optional[str] = None
    ):
        self.input_text = input_text
        self.claims = claims
        self.evidence_map = evidence_map
        self.verdicts = verdicts
        self.graph = graph
        self.report_markdown = report_markdown
        self.graph_html_path = graph_html_path

class ProofGraphPipeline:
    """Claim-review pipeline using heuristic decomposition and rule-based verification."""
    def __init__(
        self,
        decomposer: Optional[ClaimDecomposer] = None,
        retriever: Optional[HybridRetriever] = None,
        verifier: Optional[Any] = None,
        graph_builder: Optional[ProofGraphBuilder] = None,
        visualizer: Optional[ProofGraphVisualizer] = None,
        reporter: Optional[ReportGenerator] = None,
        verifier_backend: Optional[str] = None
    ):
        self.decomposer = decomposer or ClaimDecomposer()
        self.retriever = retriever or HybridRetriever()
        if verifier is not None:
            self.verifier = verifier
        elif verifier_backend is not None:
            from proofgraph.verifier.nli import create_verifier
            self.verifier = create_verifier(backend=verifier_backend)
        else:
            self.verifier = NLIVerifier()
        self.graph_builder = graph_builder or ProofGraphBuilder()
        self.visualizer = visualizer or ProofGraphVisualizer()
        self.reporter = reporter or ReportGenerator()

    def index_corpus(self, passages: List[EvidencePassage]):
        """Indexes local evidence passages into the hybrid retrieval index."""
        self.retriever.index_passages(passages)

    def verify(
        self,
        document_text: str,
        output_dir: Optional[str] = None,
        top_k_evidence: int = 3,
        use_live_web_fallback: bool = True
    ) -> ProofGraphResult:
        """
        Executes complete verification pipeline across a multi-claim document or assertion.
        """
        # 1. Decompose into Atomic Claims
        claims = self.decomposer.decompose(document_text)

        # 2. Retrieve Evidence for each Claim
        evidence_map: Dict[str, List[EvidencePassage]] = {}
        for claim in claims:
            ev_list = self.retriever.retrieve(
                claim,
                top_k=top_k_evidence,
                use_live_fallback=use_live_web_fallback
            )
            evidence_map[claim.id] = ev_list

        # 3. Apply lexical alignment, rule-based stance checks, and a static source prior.
        verdicts: Dict[str, ClaimVerdict] = {}
        for claim in claims:
            verdict = self.verifier.verify_claim(claim, evidence_map.get(claim.id, []))
            verdicts[claim.id] = verdict

        # 4. Construct a PROV-DM-inspired graph with evidence and contradiction edges.
        graph = self.graph_builder.build_graph(claims, evidence_map, verdicts)

        # 5. Export Visualizer & Generate Report
        graph_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            graph_path = os.path.join(output_dir, "graph.html")
            self.visualizer.export_html(graph, graph_path)
            report_path = os.path.join(output_dir, "report.md")
            report_md = self.reporter.generate_markdown(
                document_text, claims, evidence_map, verdicts, graph_html_rel_path="graph.html"
            )
            with open(report_path, "w") as f:
                f.write(report_md)
        else:
            report_md = self.reporter.generate_markdown(
                document_text, claims, evidence_map, verdicts, graph_html_rel_path=None
            )

        return ProofGraphResult(
            input_text=document_text,
            claims=claims,
            evidence_map=evidence_map,
            verdicts=verdicts,
            graph=graph,
            report_markdown=report_md,
            graph_html_path=graph_path
        )

    def verify_citation(
        self,
        claim_or_text: str,
        cited_url: str,
        output_dir: Optional[str] = "outputs",
        top_k_passages: int = 5
    ) -> ProofGraphResult:
        """
        AI Citation & Source Attribution Auditor:
        Directly fetches and inspects the user-specified cited URL via TinyFish Fetch (or HTTP fallback),
        extracts relevant passages, and determines whether the cited page supports, contradicts,
        or fails to establish the asserted claim (detecting citation hallucination).
        """
        # 1. Decompose into Atomic Claims
        claims = self.decomposer.decompose(claim_or_text)

        # 2. Fetch evidence strictly from the cited URL
        evidence_map: Dict[str, List[EvidencePassage]] = {}
        for claim in claims:
            # Fetch passages relevant to this claim from the exact cited URL
            passages = self.retriever.tinyfish_agent.audit_citation_url(
                url=cited_url,
                claim_text=claim.text,
                max_evidence=top_k_passages
            )
            # Filter for relevance to claim
            relevant = self.retriever.filter_relevant_passages(claim, passages)
            evidence_map[claim.id] = relevant

        # 3. Verify claim against the cited page passages
        verdicts: Dict[str, ClaimVerdict] = {}
        for claim in claims:
            verdict = self.verifier.verify_claim(claim, evidence_map.get(claim.id, []))
            verdicts[claim.id] = verdict

        # 4. Construct graph
        graph = self.graph_builder.build_graph(claims, evidence_map, verdicts)

        # 5. Export Visualizer & Report
        graph_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            graph_path = os.path.join(output_dir, "graph.html")
            self.visualizer.export_html(graph, graph_path)
            report_path = os.path.join(output_dir, "report.md")
            report_md = self.reporter.generate_markdown(
                claim_or_text, claims, evidence_map, verdicts, graph_html_rel_path="graph.html"
            )
            with open(report_path, "w") as f:
                f.write(report_md)
        else:
            report_md = self.reporter.generate_markdown(
                claim_or_text, claims, evidence_map, verdicts, graph_html_rel_path=None
            )

        return ProofGraphResult(
            input_text=claim_or_text,
            claims=claims,
            evidence_map=evidence_map,
            verdicts=verdicts,
            graph=graph,
            report_markdown=report_md,
            graph_html_path=graph_path
        )
