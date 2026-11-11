from typing import List, Dict, Optional
import datetime
from proofgraph.schemas.claim import AtomicClaim
from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.verdict import ClaimVerdict, VeracityLabel

class ReportGenerator:
    """
    Generates transparent, auditable verification reports in Markdown and HTML format.
    Includes aggregate truthfulness metrics, evidence citations, and credibility breakdowns.
    """
    def generate_markdown(
        self,
        document_text: str,
        claims: List[AtomicClaim],
        evidence_map: Dict[str, List[EvidencePassage]],
        verdicts: Dict[str, ClaimVerdict],
        graph_html_rel_path: Optional[str] = "graph.html"
    ) -> str:
        total = len(claims)
        supported_cnt = sum(1 for v in verdicts.values() if v.label == VeracityLabel.SUPPORTED)
        contradicted_cnt = sum(1 for v in verdicts.values() if v.label == VeracityLabel.CONTRADICTED)
        insufficient_cnt = sum(1 for v in verdicts.values() if v.label == VeracityLabel.INSUFFICIENT_EVIDENCE)

        md = []
        md.append("# ProofGraph: Evidence-Backed Claim Review Report")
        md.append(f"*Generated on: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*\n")
        
        md.append("## 📊 Executive Summary")
        md.append(f"- **Total Atomic Claims Analyzed:** `{total}`")
        md.append(f"- **Supported (Entailed):** `{supported_cnt}` ({(supported_cnt/total*100 if total else 0):.1f}%)")
        md.append(f"- **Contradicted (Refuted):** `{contradicted_cnt}` ({(contradicted_cnt/total*100 if total else 0):.1f}%)")
        md.append(f"- **Insufficient Evidence / Conflicting:** `{insufficient_cnt}` ({(insufficient_cnt/total*100 if total else 0):.1f}%)")
        if graph_html_rel_path:
            md.append(f"\n🔍 **Interactive Knowledge Graph:** [View Interactive Evidence Graph]({graph_html_rel_path})\n")
        else:
            md.append("")

        md.append("## 📜 Analyzed Document")
        md.append(f"> {document_text.strip()}\n")

        md.append("## 🔬 Claim-by-Claim Audit Trail")
        for idx, claim in enumerate(claims, 1):
            verdict = verdicts.get(claim.id)
            status_emoji = "✅" if verdict and verdict.label == VeracityLabel.SUPPORTED else ("❌" if verdict and verdict.label == VeracityLabel.CONTRADICTED else "⚠️")
            label_str = verdict.label.value if verdict else "UNVERIFIED"
            conf_str = f"{verdict.confidence * 100:.1f}%" if verdict else "N/A"

            md.append(f"### Claim {idx}: {status_emoji} `{label_str}` (Uncalibrated rule score: {conf_str})")
            md.append(f"**Assertion:** *\"{claim.text}\"*")
            if claim.entities:
                md.append(f"**Identified Entities:** {', '.join([f'`{e}`' for e in claim.entities])}")
            if verdict:
                md.append(f"**Justification:** {verdict.justification}")

            passages = evidence_map.get(claim.id, [])
            if passages:
                md.append("\n**Retrieved Evidence Passages:**")
                for p in passages:
                    is_supp = verdict and p.id in verdict.supporting_evidence_ids
                    is_contra = verdict and p.id in verdict.contradicting_evidence_ids
                    badge = "🟢 SUPPORTS" if is_supp else ("🔴 CONTRADICTS" if is_contra else "⚪ MENTIONS")
                    
                    link = f"[{p.source_title}]({p.source_url})" if p.source_url else f"`{p.source_title}`"
                    retrieval_method = p.metadata.get("retrieval_method", "local")
                    relevance_method = p.metadata.get("relevance_method", "retriever ranking signal")
                    md.append(f"- **{badge}** via {link} *(Retrieval: `{retrieval_method}` | Ranking signal: {p.relevance_score:.2f} ({relevance_method}) | Source prior: {p.credibility_score:.2f})*")
                    md.append(f"  > \"{p.text}\"")
            else:
                md.append("\n*No supporting or contradicting evidence discovered.*")
            md.append("\n---")

        return "\n".join(md)
