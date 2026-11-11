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

def test_synthetic_context_mismatch_examples():
    """Checks a small hand-authored alignment example set; this is not RefNLI or held out."""
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

def test_tinyfish_search_parser_and_unconfigured_status():
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent
    agent = TinyFishWebAgent(api_key="sk-tinyfish-mock-key")
    assert agent.is_available is True

    # Test structured TinySearch JSON parsing
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
    assert 0.0 <= search_passages[0].relevance_score <= 1.0

    # 3. Unkeyed agent should safely abstain and record status
    unkeyed_agent = TinyFishWebAgent(api_key="")
    assert unkeyed_agent.is_available is False
    assert unkeyed_agent.research_claim_on_web("test claim") == []
    assert unkeyed_agent.last_status == "unconfigured_key"

def test_tinyfish_search_then_fetch_preserves_page_provenance(monkeypatch):
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent
    agent = TinyFishWebAgent(api_key="sk-tinyfish-mock")
    search_hit = EvidencePassage(
        id="search_1",
        text="Apollo 11: official mission timeline and launch details.",
        source_url="https://www.nasa.gov/apollo11",
        source_title="NASA Apollo 11",
        source_domain="www.nasa.gov",
    )
    monkeypatch.setattr(agent, "search_web", lambda query, max_results=3: [search_hit])
    monkeypatch.setattr(agent, "_post_fetch", lambda urls: {
        "results": [{
            "url": "https://www.nasa.gov/apollo11",
            "final_url": "https://science.nasa.gov/mission/apollo-11/",
            "title": "Apollo 11 | NASA",
            "text": "# Apollo 11\n\nApollo 11 launched on July 16, 1969. Neil Armstrong walked on the Moon.\n\nMore mission details.",
        }],
        "errors": [],
    })
    found = agent.research_claim_on_web("When did Apollo 11 launch?", max_evidence=1)
    assert len(found) == 1
    assert found[0].source_url == "https://science.nasa.gov/mission/apollo-11/"
    assert found[0].source_title == "Apollo 11 | NASA"
    assert "July 16, 1969" in found[0].text
    assert found[0].metadata["retrieval_method"] == "tinyfish_fetch"
    assert found[0].metadata["relevance_method"] == "lexical_overlap_and_search_rank_proxy"

def test_tinyfish_rejects_non_public_target_urls():
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent
    valid = [
        "https://www.nasa.gov/mission",
        "https://example.org/page",
    ]
    invalid = [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://127.0.0.1/",
        "http://192.168.1.20/",
        "http://user:pass@example.com/",
    ]
    assert all(TinyFishWebAgent._valid_public_http_url(url) for url in valid)
    assert not any(TinyFishWebAgent._valid_public_http_url(url) for url in invalid)

def test_tinyfish_fetch_request_contract_and_error_status(monkeypatch):
    import io
    import json
    from email.message import Message
    import urllib.error
    import proofgraph.retriever.tinyfish_agent as tinyfish_module
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent

    agent = TinyFishWebAgent(api_key="sk-tinyfish-mock")
    response_body = json.dumps({
        "results": [{"url": "https://example.org/", "final_url": "https://example.org/", "title": "Example", "text": "Example page text."}],
        "errors": [],
    }).encode()

    class FakeResponse(io.BytesIO):
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://api.fetch.tinyfish.ai"
        assert request.get_method() == "POST"
        assert request.get_header("X-api-key") == "sk-tinyfish-mock"
        payload = json.loads(request.data.decode())
        assert payload["urls"] == ["https://example.org/"]
        assert payload["format"] == "markdown"
        assert payload["ttl"] == 0
        assert timeout == agent.timeout
        return FakeResponse(response_body)

    monkeypatch.setattr(tinyfish_module.urllib.request, "urlopen", fake_urlopen)
    data = agent._post_fetch(["https://example.org/"])
    assert data["results"][0]["title"] == "Example"
    assert agent.last_status == "success"

    calls = {"count": 0}
    def throttled_once(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            headers = Message()
            headers["Retry-After"] = "0"
            raise urllib.error.HTTPError(request.full_url, 503, "temporary", headers, io.BytesIO())
        return FakeResponse(response_body)

    monkeypatch.setattr(tinyfish_module.urllib.request, "urlopen", throttled_once)
    monkeypatch.setattr(tinyfish_module.time, "sleep", lambda _delay: None)
    assert agent._post_fetch(["https://example.org/"])["results"]
    assert calls["count"] == 2

    def network_failure(_request, timeout):
        raise urllib.error.URLError("offline test")

    monkeypatch.setattr(tinyfish_module.urllib.request, "urlopen", network_failure)
    assert agent.fetch_page_content("https://example.org/") is None
    assert agent.last_status == "network_exception"
    assert agent.last_error_message

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

@pytest.mark.skipif(
    not (os.environ.get("TINYFISH_API_KEY") and os.environ.get("TINYFISH_LIVE_TEST") == "1"),
    reason="Set TINYFISH_LIVE_TEST=1 and TINYFISH_API_KEY to run the network smoke test",
)
def test_live_tinyfish_search_api_if_keyed():
    """Live opt-in smoke test: validates TinyFish Search followed by page Fetch."""
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent
    agent = TinyFishWebAgent(timeout_seconds=30)
    assert agent.is_available is True
    hits = agent.research_claim_on_web("Neil Armstrong Apollo 11 moon landing", max_evidence=1)
    assert len(hits) >= 1
    assert hits[0].source_url.startswith("http")
    assert hits[0].source_domain != ""
    assert hits[0].metadata.get("retrieval_method") == "tinyfish_fetch"

def test_neural_nli_verifier_offline_fallback():
    from proofgraph.verifier.neural_nli import DebertaNLIVerifier
    verifier = DebertaNLIVerifier(prefer_offline=True)
    claim = AtomicClaim(text="Apollo 11 landed on the Moon", parent_sentence="")
    verdict = verifier.verify_claim(claim, [])
    assert verdict.label == VeracityLabel.INSUFFICIENT_EVIDENCE
    assert verdict.confidence == 0.50
    probs, label = verifier.predict_pair("Evidence premise.", "Claim hypothesis.")
    assert label == "NEUTRAL"
    assert probs.tolist() == [0.0, 0.0, 1.0]

def test_neural_nli_predict_pair_uses_unpadded_sequence():
    import numpy as np
    from proofgraph.verifier.neural_nli import DebertaNLIVerifier

    class Encoding:
        ids = [101, 42, 102, 77, 102]
        attention_mask = [1, 1, 1, 1, 1]

    class Tokenizer:
        def encode(self, premise, hypothesis):
            return Encoding()

    class Session:
        def run(self, _outputs, inputs):
            assert inputs["input_ids"].shape == (1, 5)
            assert inputs["attention_mask"].shape == (1, 5)
            return [np.array([[0.0, 4.0, 0.0]], dtype=np.float32)]

    verifier = DebertaNLIVerifier(prefer_offline=True)
    verifier.tokenizer = Tokenizer()
    verifier.session = Session()
    verifier._is_ready = True
    probs, label = verifier.predict_pair("premise", "hypothesis")
    assert label == "ENTAILMENT"
    assert probs[1] > 0.9

def test_neural_nli_does_not_retry_failed_model_initialization():
    from proofgraph.verifier.neural_nli import DebertaNLIVerifier

    verifier = DebertaNLIVerifier(prefer_offline=True)
    verifier.prefer_offline = False
    verifier._initialization_attempted = True
    assert verifier.is_available is False

@pytest.mark.parametrize(
    ("probabilities", "expected_label", "evidence_field"),
    [
        ([0.01, 0.97, 0.02], VeracityLabel.SUPPORTED, "supporting_evidence_ids"),
        ([0.97, 0.01, 0.02], VeracityLabel.CONTRADICTED, "contradicting_evidence_ids"),
    ],
)
def test_neural_nli_verdict_aggregates_pairwise_stance(probabilities, expected_label, evidence_field):
    import numpy as np
    from proofgraph.verifier.neural_nli import DebertaNLIVerifier

    verifier = DebertaNLIVerifier(prefer_offline=True)
    verifier.predict_pair = lambda _premise, _hypothesis: (np.asarray(probabilities), "unused")
    claim = AtomicClaim(text="Apollo 11 landed on the Moon in 1969.", parent_sentence="")
    evidence = EvidencePassage(
        id="e1",
        text="Apollo 11 landed on the Moon in 1969.",
        source_domain="nasa.gov",
        relevance_score=0.95,
    )
    verdict = verifier.verify_claim(claim, [evidence])
    assert verdict.label == expected_label
    assert getattr(verdict, evidence_field) == ["e1"]

def test_benchmark_verifier_receives_only_the_current_candidate_pool():
    from types import SimpleNamespace
    from evaluate import verify_claim_against_pool

    evidence = [EvidencePassage(id="current", text="Current pool passage.")]
    seen = {}

    class Decomposer:
        def decompose(self, text):
            return [AtomicClaim(text=text, parent_sentence=text)]

    class Verifier:
        def verify_claim(self, claim, passages):
            seen["claim"] = claim.text
            seen["passages"] = passages
            return SimpleNamespace(label=VeracityLabel.SUPPORTED)

    prediction = verify_claim_against_pool(
        SimpleNamespace(decomposer=Decomposer(), verifier=Verifier()),
        "A test claim.",
        evidence,
    )
    assert prediction == VeracityLabel.SUPPORTED.value
    assert seen["claim"] == "A test claim."
    assert seen["passages"] is evidence

def test_neural_nli_inference():
    from proofgraph.verifier.neural_nli import DebertaNLIVerifier
    verifier = DebertaNLIVerifier()
    if not verifier.is_available:
        pytest.skip("Neural NLI model weights not yet cached locally")

    # Clear entailment
    probs_entail, label_entail = verifier.predict_pair(
        "Commander Neil Armstrong stepped onto the lunar surface on July 20, 1969.",
        "Neil Armstrong walked on the Moon in 1969."
    )
    assert label_entail == "ENTAILMENT"
    assert probs_entail[1] > 0.80

    # Clear contradiction
    probs_contra, label_contra = verifier.predict_pair(
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower is located in Rome, Italy."
    )
    assert label_contra == "CONTRADICTION"
    assert probs_contra[0] > 0.80

def test_hybrid_nli_verifier_ensemble():
    from proofgraph.verifier.nli import HybridNLIVerifier
    hybrid = HybridNLIVerifier()

    # 1. Deterministic rule check: numeric conflict
    claim = AtomicClaim(text="Water boils at 50 degrees Celsius at sea level.", parent_sentence="")
    evid = EvidencePassage(id="e1", text="Pure water boils at 100 degrees Celsius at sea level.", source_url="https://britannica.com", relevance_score=0.95)
    v1 = hybrid.verify_claim(claim, [evid])
    assert v1.label == VeracityLabel.CONTRADICTED

    # 2. RefNLI context mismatch defense: distinct individuals
    claim2 = AtomicClaim(text="Alexander Graham Bell invented the telephone.", parent_sentence="", entities=["Alexander Graham Bell"])
    evid2 = EvidencePassage(id="e2", text="Thomas Edison invented the phonograph.", source_url="https://wikipedia.org", relevance_score=0.90)
    v2 = hybrid.verify_claim(claim2, [evid2])
    assert v2.label == VeracityLabel.INSUFFICIENT_EVIDENCE

def test_hybrid_retriever_multimodal_relevance():
    retriever = HybridRetriever(prefer_offline=True)
    passages = [
        EvidencePassage(id="p1", text="The Apollo 11 lunar module landed on July 20, 1969.", source_title="NASA"),
        EvidencePassage(id="p2", text="The Golden Gate Bridge is an iconic suspension bridge in California.", source_title="Britannica"),
        EvidencePassage(id="p3", text="Mount Everest is the highest mountain on Earth above sea level.", source_title="National Geographic")
    ]
    retriever.index_passages(passages)
    results = retriever.search_local("Apollo 11 lunar module 1969", top_k=2)
    assert len(results) == 2
    assert results[0].id == "p1"
    # Multimodal relevance score must be within valid bounds and reflect both BM25 and dense
    assert 0.40 <= results[0].relevance_score <= 1.0

def test_report_generator_without_graph_path():
    from proofgraph.reporter import ReportGenerator
    reporter = ReportGenerator()
    claim = AtomicClaim(text="Test assertion", parent_sentence="")
    verdict = ClaimVerdict(
        claim_id=claim.id,
        label=VeracityLabel.SUPPORTED,
        confidence=0.95,
        justification="Verified by evidence"
    )
    md = reporter.generate_markdown("Test document", [claim], {}, {claim.id: verdict}, graph_html_rel_path=None)
    assert "View Interactive Evidence Graph" not in md
    assert "**Supported (Entailed):** `1`" in md

def test_tinyfish_domain_parsing_fallback():
    from proofgraph.retriever.tinyfish_agent import TinyFishWebAgent
    agent = TinyFishWebAgent(api_key="sk-tinyfish-mock")
    # Simulate official TinySearch schema with 'domain'
    data_with_domain = {
        "results": [
            {
                "url": "https://www.nasa.gov/mission_pages/apollo/apollo-11.html",
                "title": "Apollo 11",
                "snippet": "Neil Armstrong and Buzz Aldrin landed on the Moon in July 1969.",
                "domain": "www.nasa.gov"
            }
        ]
    }
    passages = agent._parse_search_results(data_with_domain, "Apollo 11", max_results=1)
    assert len(passages) == 1
    assert passages[0].source_domain == "www.nasa.gov"
    assert passages[0].metadata.get("domain") == "www.nasa.gov"

def test_hybrid_calibrated_fallback_threshold():
    from proofgraph.verifier.nli import HybridNLIVerifier
    import warnings

    hybrid = HybridNLIVerifier(prefer_offline=True, neural_fallback_threshold=0.90)
    assert hybrid.neural_fallback_threshold == 0.90
    assert hybrid.neural_active is False

    # Simulate mock neural verifier that emits a medium-confidence verdict (0.75)
    class MockNeuralVerifier:
        is_available = True
        def verify_claim(self, claim, passages):
            return ClaimVerdict(
                claim_id=claim.id,
                label=VeracityLabel.CONTRADICTED,
                confidence=0.75, # below 0.90
                justification="Medium confidence refutation"
            )

    hybrid._neural_verifier = MockNeuralVerifier()
    assert hybrid.neural_active is True

    claim = AtomicClaim(text="Ambiguous claim where rules abstain", parent_sentence="")
    # Because confidence 0.75 < 0.90, hybrid must NOT overturn rule-based abstention!
    v = hybrid.verify_claim(claim, [])
    assert v.label == VeracityLabel.INSUFFICIENT_EVIDENCE

    # Now simulate high-confidence neural verdict (0.94 >= 0.90)
    class HighConfNeuralVerifier:
        is_available = True
        def verify_claim(self, claim, passages):
            return ClaimVerdict(
                claim_id=claim.id,
                label=VeracityLabel.CONTRADICTED,
                confidence=0.94, # above 0.90
                justification="High confidence refutation"
            )

    hybrid._neural_verifier = HighConfNeuralVerifier()
    v_high = hybrid.verify_claim(claim, [])
    assert v_high.label == VeracityLabel.CONTRADICTED
    assert v_high.confidence == 0.94

def test_hybrid_warning_on_neural_init_failure():
    from proofgraph.verifier.nli import HybridNLIVerifier
    import warnings

    class BadNeuralVerifier:
        def __init__(self, **kwargs):
            raise RuntimeError("Simulated missing ONNX runtime or network failure")

    hybrid = HybridNLIVerifier(prefer_offline=False)
    # Monkeypatch import
    import proofgraph.verifier.nli as nli_mod
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        # trigger property access with broken constructor
        hybrid._neural_verifier = None
        orig = nli_mod.DebertaNLIVerifier if hasattr(nli_mod, "DebertaNLIVerifier") else None
        try:
            import proofgraph.verifier.neural_nli as nnli
            old_cls = nnli.DebertaNLIVerifier
            nnli.DebertaNLIVerifier = BadNeuralVerifier
            nv = hybrid.neural_verifier
            assert nv is None
            assert hybrid.neural_active is False
            assert any("Neural NLI could not be initialized" in str(w.message) for w in record)
        finally:
            nnli.DebertaNLIVerifier = old_cls

def test_neural_nli_runtime_label_schema_verification():
    from proofgraph.verifier.neural_nli import DebertaNLIVerifier
    verifier = DebertaNLIVerifier(prefer_offline=True)
    # Verify authoritative label ordering
    assert verifier.label_mapping[0] == "CONTRADICTION"
    assert verifier.label_mapping[1] == "ENTAILMENT"
    assert verifier.label_mapping[2] == "NEUTRAL"
    assert verifier.MODEL_REVISION == "2a4f614a701367a02d51389039afc998faeda637"

def test_get_stratified_dev_test_split_proportions():
    from evaluate import get_stratified_dev_test_split
    sample_data = [
        {"claim": f"claim_r_{i}", "label": "Refuted"} for i in range(15)
    ] + [
        {"claim": f"claim_s_{i}", "label": "Supported"} for i in range(15)
    ] + [
        {"claim": f"claim_c_{i}", "label": "Conflicting Evidence/Cherrypicking"} for i in range(15)
    ] + [
        {"claim": f"claim_n_{i}", "label": "Not Enough Evidence"} for i in range(15)
    ]
    dev, test = get_stratified_dev_test_split(sample_data, dev_per_class=10, test_per_class=5)
    assert len(dev) == 40
    assert len(test) == 20
    # Check that each class has exactly 10 in dev and 5 in test
    for label in ["Refuted", "Supported", "Conflicting Evidence/Cherrypicking", "Not Enough Evidence"]:
        assert sum(1 for d in dev if d["label"] == label) == 10
        assert sum(1 for d in test if d["label"] == label) == 5

def test_compute_bootstrap_confidence_intervals_bounds():
    from evaluate import compute_bootstrap_confidence_intervals
    y_true = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"] * 10
    y_pred = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"] * 10
    labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    ci = compute_bootstrap_confidence_intervals(y_true, y_pred, labels, n_bootstraps=100)
    assert "accuracy_ci_95" in ci
    assert "macro_f1_ci_95" in ci
    assert ci["accuracy_ci_95"][0] <= 1.0 <= ci["accuracy_ci_95"][1]

def test_hybrid_cross_verifier_diagnostics():
    from proofgraph.verifier.nli import HybridNLIVerifier
    hybrid = HybridNLIVerifier(prefer_offline=True)
    claim = AtomicClaim(text="Water boils at 100 degrees Celsius.", parent_sentence="")
    evidence = EvidencePassage(id="e1", text="Water boils at 100 degrees Celsius at sea level.", relevance_score=0.95)
    hybrid.verify_claim(claim, [evidence])
    assert hasattr(hybrid, "last_diagnostics")
    assert "rule_label" in hybrid.last_diagnostics

def test_demo_case_study_execution(tmp_path):
    from demo_case_study import run_case_study
    out_file = str(tmp_path / "test_case_study.md")
    res = run_case_study(output_path=out_file)
    assert os.path.exists(res)
    with open(res, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Eiffel Tower" in content
    assert "RefNLI Context Alignment Gating" in content
