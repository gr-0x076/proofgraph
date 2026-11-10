from typing import List, Dict, Optional
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
    """
    End-to-End Autonomous AI Evidence Intelligence Engine:
    Ingestion -> Claimify Decomposition -> Hybrid Retrieval -> RefNLI & CONFACT Verification
    -> W3C PROV-DM Graph with Contradiction Edges -> Interactive Visualization & Audit Report
    """
    def __init__(
        self,
        decomposer: Optional[ClaimDecomposer] = None,
        retriever: Optional[HybridRetriever] = None,
        verifier: Optional[NLIVerifier] = None,
        graph_builder: Optional[ProofGraphBuilder] = None,
        visualizer: Optional[ProofGraphVisualizer] = None,
        reporter: Optional[ReportGenerator] = None
    ):
        self.decomposer = decomposer or ClaimDecomposer()
        self.retriever = retriever or HybridRetriever()
        self.verifier = verifier or NLIVerifier()
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

        # 3. Verify each Claim using RefNLI + Credibility Weighted NLI
        verdicts: Dict[str, ClaimVerdict] = {}
        for claim in claims:
            verdict = self.verifier.verify_claim(claim, evidence_map.get(claim.id, []))
            verdicts[claim.id] = verdict

        # 4. Construct W3C PROV-DM Knowledge Graph with Contradiction Edges
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
                document_text, claims, evidence_map, verdicts, graph_html_rel_path="graph.html"
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
