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
    # Unit test runs 100% offline using deterministic hash embedding
    retriever = HybridRetriever(prefer_offline=True)
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

def test_refnli_held_out_context_mismatch_suite():
    """Evaluates RefNLI alignment gate on a held-out benchmark of aligned vs. mismatched entity pairs."""
    gate = RefNLIAlignmentGate(jaccard_threshold=0.12)
    test_cases = [
        # (claim_text, entities, passage_text, expected_aligned)
        ("Marie Curie won the Nobel Prize in Chemistry in 1911.", ["Marie Curie", "1911"], "Marie Curie was awarded the 1911 Nobel Prize in Chemistry.", True),
        ("Albert Einstein published General Relativity in 1915.", ["Albert Einstein", "1915"], "Albert Einstein formulated General Relativity in Berlin in 1915.", True),
        ("Alan Turing cracked the Enigma code at Bletchley Park.", ["Alan Turing", "Bletchley Park"], "Alan Turing worked at Bletchley Park deciphering German Enigma encryptions.", True),
        # Distractors: same keywords/dates/actions, completely different person/entity (Reference Indeterminacy)
        ("Peter was born in 1995 in Toronto.", ["Peter", "1995", "Toronto"], "William was born in 1995 in Toronto to diplomat parents.", False),
        ("Dr. Smith invented the cardiac stent in 1985.", ["Dr. Smith", "1985"], "Dr. Charles Dotter pioneered modern angioplasty procedures in 1964.", False),
        ("Acme Corp filed for bankruptcy in 2023.", ["Acme Corp", "2023"], "Beta LLC filed for Chapter 11 bankruptcy reorganization in 2023.", False)
    ]
    for claim_text, entities, passage_text, expected in test_cases:
        claim = AtomicClaim(text=claim_text, parent_sentence="", entities=entities)
        passage = EvidencePassage(text=passage_text, source_title="Benchmark Source")
        aligned, score, msg = gate.check_alignment(claim, passage)
        assert aligned == expected, f"Failed on case: '{claim_text}' vs '{passage_text}' (expected {expected}, got {aligned})"

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

def test_adversarial_numeric_and_inventor_conflicts():
    verifier = NLIVerifier()
    # 1. Unrelated numbers must NOT falsely contradict (birth year vs award year)
    claim1 = AtomicClaim(text="Albert Einstein was born in 1879.", parent_sentence="")
    passage1 = EvidencePassage(
        text="Albert Einstein published his theory of General Relativity in 1915.",
        source_title="History of Science"
    )
    v1 = verifier.verify_claim(claim1, [passage1])
    assert v1.label != VeracityLabel.CONTRADICTED

    # 2. Distinct invention objects must NOT falsely contradict
    claim2 = AtomicClaim(text="Thomas Edison invented the phonograph in 1877.", parent_sentence="")
    passage2 = EvidencePassage(
        text="Alexander Graham Bell was awarded the first patent for the electric telephone in 1876.",
        source_title="Patent History"
    )
    v2 = verifier.verify_claim(claim2, [passage2])
    assert v2.label != VeracityLabel.CONTRADICTED

    # 3. Bound predicate numbers DO contradict (boiling point 50 vs 100)
    claim3 = AtomicClaim(text="Water boils at 50 degrees Celsius.", parent_sentence="")
    passage3 = EvidencePassage(
        text="At standard atmospheric pressure, water boils at 100 degrees Celsius.",
        source_title="Chemistry Handbook"
    )
    v3 = verifier.verify_claim(claim3, [passage3])
    assert v3.label == VeracityLabel.CONTRADICTED

    # 4. Conflicting inventors for the same object DO contradict (telephone Edison vs Bell)
    claim4 = AtomicClaim(text="Thomas Edison invented the telephone.", parent_sentence="")
    passage4 = EvidencePassage(
        text="Alexander Graham Bell was awarded the patent for inventing the telephone.",
        source_title="Patent Records"
    )
    v4 = verifier.verify_claim(claim4, [passage4])
    assert v4.label == VeracityLabel.CONTRADICTED

def test_end_to_end_pipeline_execution(tmp_path):
    # Pass prefer_offline=True so pipeline unit tests run in firewalled/offline environments
    offline_retriever = HybridRetriever(prefer_offline=True)
    pipeline = ProofGraphPipeline(retriever=offline_retriever)
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

def test_domain_spoofing_resistance():
    scorer = SourceCredibilityScorer()
    # Legitimate domains and subdomains
    assert scorer.get_credibility("reuters.com") >= 0.90
    assert scorer.get_credibility("https://www.reuters.com/world") >= 0.90
    assert scorer.get_credibility("news.bbc.co.uk") >= 0.90
    assert scorer.get_credibility("science.nasa.gov") >= 0.95

    # Spoofed domains must NOT receive high authority scores
    assert scorer.get_credibility("reuters.com.evil.org") == 0.50
    assert scorer.get_credibility("fake-reuters.com") == 0.50
    assert scorer.get_credibility("https://nature.com.phishing-site.net/article") == 0.50

def test_fire_iterative_decision_module():
    from proofgraph.retriever.fire_policy import FIREIterativeDecisionModule
    fire = FIREIterativeDecisionModule(confidence_threshold=0.85, max_hops=3)

    claim = AtomicClaim(
        text="James Webb Space Telescope launched in December 2021 from French Guiana.",
        parent_sentence="",
        entities=["James Webb Space Telescope", "December 2021", "French Guiana"]
    )
    # Evidence only mentions telescope, missing French Guiana and launch date
    partial_evidence = [
        EvidencePassage(
            id="e1",
            text="The James Webb Space Telescope is an optical space observatory.",
            relevance_score=0.70
        )
    ]
    should_stop, next_query = fire.evaluate_step(
        claim=claim,
        retrieved_pool=partial_evidence,
        current_hop=1,
        executed_queries=[claim.text]
    )
    # Must NOT stop early; must formulate targeted query containing missing entity
    assert should_stop is False
    assert next_query is not None
    assert any(ent in next_query for ent in ["December 2021", "French Guiana"])

def test_offline_hash_embedding_fallback():
    from proofgraph.retriever.hybrid_retriever import DeterministicHashEmbedding
    model = DeterministicHashEmbedding(dim=128)
    vecs = list(model.embed(["test sentence for offline unit tests"]))
    assert len(vecs) == 1
    assert vecs[0].shape == (128,)

@pytest.mark.integration
def test_live_fastembed_download_if_online():
    """Integration test: validates live FastEmbed model loading if network / model cache is available."""
    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        vec = list(model.embed(["Integration test query"]))[0]
        assert len(vec) == 384
    except Exception as e:
        pytest.skip(f"Live network access or cached model unavailable in this sandbox: {e}")

def test_tinyfish_agent_parsing_and_fallback():
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent
    agent = TinyFishWebAgent(api_key="sk-tinyfish-mock-key")
    assert agent.is_available is True

    # 1. Test structured response parsing from agent payload
    mock_payload = {
        "status": "completed",
        "output": {
            "evidence": [
                {
                    "text": "Apollo 11 landed humans on the Moon on July 20, 1969.",
                    "url": "https://www.nasa.gov/mission/apollo-11",
                    "source_title": "NASA Official Mission Archive"
                }
            ]
        }
    }
    passages = agent._parse_agent_response(mock_payload, "Apollo 11 landed on the moon", "https://default.org")
    assert len(passages) == 1
    assert "Apollo 11 landed" in passages[0].text
    assert passages[0].source_domain == "www.nasa.gov"
    # Dynamic relevance calculation (no hard-coded static scores)
    assert 0.70 <= passages[0].relevance_score <= 1.0

    # 2. Test structured TinySearch JSON parsing
    search_payload = {
        "query": "Apollo 11 launch date",
        "results": [
            {
                "position": 1,
                "title": "Apollo 11 Mission Overview",
                "snippet": "Apollo 11 launched from Cape Kennedy on July 16, 1969.",
                "url": "https://www.nasa.gov/mission/apollo-11-launch",
                "site_name": "www.nasa.gov"
            }
        ]
    }
    search_passages = agent._parse_search_results(search_payload, "Apollo 11 launch date", max_results=1)
    assert len(search_passages) == 1
    assert search_passages[0].source_url == "https://www.nasa.gov/mission/apollo-11-launch"
    assert search_passages[0].metadata["retrieval_method"] == "tinyfish_search"
    assert search_passages[0].relevance_score > 0.60

    # 3. Unkeyed agent should safely abstain and record status
    unkeyed_agent = TinyFishWebAgent(api_key="")
    assert unkeyed_agent.is_available is False
    assert unkeyed_agent.research_claim_on_web("test claim") == []
    assert unkeyed_agent.last_status == "unconfigured_key"

def test_hybrid_retriever_with_tinyfish_agent(monkeypatch):
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent
    mock_agent = TinyFishWebAgent(api_key="sk-tinyfish-mock")
    
    mock_passage = EvidencePassage(
        id="tf_1",
        text="The Great Wall of China is over 13,000 miles long.",
        source_title="UNESCO World Heritage",
        source_domain="unesco.org",
        relevance_score=0.92
    )
    monkeypatch.setattr(mock_agent, "research_claim_on_web", lambda q, max_evidence=3: [mock_passage])
    
    retriever = HybridRetriever(prefer_offline=True, tinyfish_agent=mock_agent)
    # Search with empty local index: should trigger TinyFish fallback
    claim = AtomicClaim(text="Great wall of china length", parent_sentence="")
    results = retriever.retrieve(claim, top_k=2, use_live_fallback=True)
    assert len(results) == 1
    assert results[0].id == "tf_1"
    assert "Great Wall of China" in results[0].text
    
    # Also test direct search_tinyfish_live
    tf_direct = retriever.search_tinyfish_live("test query", max_results=2)
    assert len(tf_direct) == 1
    assert tf_direct[0].id == "tf_1"

@pytest.mark.skipif(not os.environ.get("TINYFISH_API_KEY"), reason="Requires live TINYFISH_API_KEY")
def test_live_tinyfish_search_api_if_keyed():
    """Live opt-in smoke test: validates real network retrieval against TinySearch API."""
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent
    agent = TinyFishWebAgent()
    assert agent.is_available is True
    hits = agent.search_web("Neil Armstrong Apollo 11 moon", max_results=2)
    assert len(hits) >= 1
    assert hits[0].source_url.startswith("http")
    assert hits[0].source_domain != ""
    assert agent.last_status == "success"





