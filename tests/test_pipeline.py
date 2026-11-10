import pytest
import os
import networkx as nx

from proofgraph.schemas import AtomicClaim, EvidencePassage, VeracityLabel, ClaimVerdict
from proofgraph.decomposer import ClaimDecomposer
from proofgraph.retriever import HybridRetriever
from proofgraph.verifier import RefNLIAlignmentGate, SourceCredibilityScorer, NLIVerifier
from proofgraph.graph import ProofGraphBuilder, ProofGraphVisualizer
from proofgraph.reporter import ReportGenerator
from proofgraph.pipeline import ProofGraphPipeline

def test_decomposer_splitting_and_verifiability():
    decomposer = ClaimDecomposer()
    text = (
        "Dr. Jane Smith co-founded Acme Corp in 2018, but the company went bankrupt in 2023. "
        "I feel like this was completely unfair."
    )
    claims = decomposer.decompose(text)
    assert len(claims) == 3
    assert claims[0].is_verifiable is True
    assert "Dr. Jane Smith" in claims[0].text
    assert claims[1].is_verifiable is True
    assert "bankrupt in 2023" in claims[1].text
    assert claims[2].is_verifiable is False  # Opinion flagged

def test_hybrid_retriever_ranking():
    retriever = HybridRetriever()
    passages = [
        EvidencePassage(id="p1", text="Neil Armstrong walked on the Moon on July 20, 1969.", source_title="NASA"),
        EvidencePassage(id="p2", text="The Golden Gate Bridge opened in 1937.", source_title="Britannica")
    ]
    retriever.index_passages(passages)
    claim = AtomicClaim(text="Apollo 11 landed on the moon in 1969", parent_sentence="")
    results = retriever.retrieve(claim, top_k=1, use_live_fallback=False)
    assert len(results) == 1
    assert results[0].id == "p1"
    assert "Neil Armstrong" in results[0].text

def test_refnli_alignment_gate():
    gate = RefNLIAlignmentGate(jaccard_threshold=0.12)
    claim = AtomicClaim(text="Peter was born in 1995.", parent_sentence="", entities=["Peter", "1995"])
    # Irrelevant passage discussing a different person
    irrelevant = EvidencePassage(text="John Doe was born in 1980 in London.", source_title="Bio")
    aligned, score, msg = gate.check_alignment(claim, irrelevant)
    assert aligned is False
    assert "Context Mismatch" in msg

def test_confact_credibility_weighting():
    scorer = SourceCredibilityScorer()
    assert scorer.get_credibility("reuters.com") >= 0.90
    assert scorer.get_credibility("nature.com") >= 0.95
    assert scorer.get_credibility("infowars.com") <= 0.20
    assert scorer.get_credibility("harvard.edu") >= 0.90

def test_nli_verifier_contradiction_detection():
    verifier = NLIVerifier()
    claim = AtomicClaim(text="The Eiffel Tower was constructed in Rome in 1950.", parent_sentence="")
    passages = [
        EvidencePassage(
            id="e_true",
            text="The Eiffel Tower is located in Paris, France and was built between 1887 and 1889.",
            source_domain="britannica.com",
            source_title="Britannica",
            credibility_score=0.95
        ),
        EvidencePassage(
            id="e_rumor",
            text="Conspiracy theorists believe a secret Eiffel Tower was built in Rome in 1950.",
            source_domain="infowars.com",
            source_title="Tabloid",
            credibility_score=0.15
        )
    ]
    verdict = verifier.verify_claim(claim, passages)
    assert verdict.label == VeracityLabel.CONTRADICTED
    assert "e_true" in verdict.contradicting_evidence_ids
    assert verdict.confidence > 0.80

def test_end_to_end_pipeline_execution(tmp_path):
    pipeline = ProofGraphPipeline()
    corpus = [
        EvidencePassage(
            text="Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming.",
            source_title="NobelPrize",
            source_domain="nobelprize.org",
            credibility_score=0.98
        )
    ]
    pipeline.index_corpus(corpus)
    doc = "Alexander Fleming discovered penicillin in 1928."
    out_dir = str(tmp_path / "test_out")
    res = pipeline.verify(doc, output_dir=out_dir, use_live_web_fallback=False)

    assert len(res.claims) == 1
    v = res.verdicts[res.claims[0].id]
    assert v.label == VeracityLabel.SUPPORTED
    assert os.path.exists(os.path.join(out_dir, "report.md"))
    assert os.path.exists(os.path.join(out_dir, "graph.html"))
    assert res.graph.number_of_nodes() >= 3
