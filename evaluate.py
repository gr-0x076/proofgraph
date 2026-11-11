#!/usr/bin/env python3
"""
ProofGraph Rigorous Benchmark & Diagnostic Evaluation Suite:
1. Diagnostic Unit Sanity Suite (10 hand-authored stress cases)
2. In-Batch Evidence Retrieval Quality Diagnostic on Pooled AVeriTeC Corpus (N=60 Claims, 170 Passages)
3. AVeriTeC Dev Sample Verdict Classification on Candidate Evidence Pools (N=60 Claims, 3-Way & 4-Way Analysis)
4. Synthetic Context-Mismatch Stress Diagnostic (N=20 Hand-Authored Claim-Passage Pairs, inspired by RefNLI)
Note: For full dataset provenance, task formulation, and protocol distinctions (e.g. custom classification vs. official Hungarian METEOR shared task), see docs/DATA_CARD.md.
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import numpy as np

import argparse
from proofgraph.pipeline import ProofGraphPipeline
from proofgraph.schemas import EvidencePassage, VeracityLabel, AtomicClaim
from proofgraph.verifier.nli import NLIVerifier, RuleStanceVerifier, HybridNLIVerifier, create_verifier
from proofgraph.verifier.neural_nli import DebertaNLIVerifier
from proofgraph.verifier.alignment import RefNLIAlignmentGate
from proofgraph.verifier.credibility import SourceCredibilityScorer
from proofgraph.retriever.hybrid_retriever import HybridRetriever

# =====================================================================
# PART 1: DIAGNOSTIC UNIT SANITY DATASET (10 Stress Cases)
# =====================================================================
DIAGNOSTIC_DATASET = [
    {
        "claim": "Apollo 11 landed on the moon on July 20, 1969.",
        "ground_truth": VeracityLabel.SUPPORTED,
        "type": "temporal_exact_support",
        "evidence": [
            EvidencePassage(text="Apollo 11 commander Neil Armstrong stepped onto the lunar surface on July 20, 1969.", source_title="NASA", source_domain="nasa.gov", credibility_score=0.99)
        ]
    },
    {
        "claim": "The Eiffel Tower was constructed in Rome in 1950.",
        "ground_truth": VeracityLabel.CONTRADICTED,
        "type": "geo_and_temporal_conflict",
        "evidence": [
            EvidencePassage(text="The Eiffel Tower was erected in Paris between 1887 and 1889.", source_title="Britannica", source_domain="britannica.com", credibility_score=0.95),
            EvidencePassage(text="Rumors claim a secret Eiffel Tower was built in Rome in 1950.", source_title="TabloidBlog", source_domain="infowars.com", credibility_score=0.15)
        ]
    },
    {
        "claim": "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital.",
        "ground_truth": VeracityLabel.SUPPORTED,
        "type": "factual_assertion_support",
        "evidence": [
            EvidencePassage(text="In September 1928, Scottish physician Alexander Fleming discovered penicillin at St Mary's Hospital in London.", source_title="NobelPrize.org", source_domain="nobelprize.org", credibility_score=0.98)
        ]
    },
    {
        "claim": "Water boils at 50 degrees Celsius at standard sea-level atmospheric pressure.",
        "ground_truth": VeracityLabel.CONTRADICTED,
        "type": "numerical_contradiction",
        "evidence": [
            EvidencePassage(text="At standard atmospheric pressure (1 atm), pure water boils at 100 degrees Celsius.", source_title="NIST Chemistry", source_domain="nist.gov", credibility_score=0.99)
        ]
    },
    {
        "claim": "Albert Einstein was awarded the Nobel Prize in Physics for his discovery of the law of the photoelectric effect in 1921.",
        "ground_truth": VeracityLabel.SUPPORTED,
        "type": "complex_predicate_support",
        "evidence": [
            EvidencePassage(text="The 1921 Nobel Prize in Physics was awarded to Albert Einstein for his services to Theoretical Physics, and especially for his discovery of the law of the photoelectric effect.", source_title="Nobel Foundation", source_domain="nobelprize.org", credibility_score=0.98)
        ]
    },
    {
        "claim": "Peter was born in 1995.",
        "ground_truth": VeracityLabel.INSUFFICIENT_EVIDENCE,
        "type": "refnli_context_mismatch",
        "evidence": [
            EvidencePassage(text="John Doe was born in 1980 in London and became an author.", source_title="Author Bio", source_domain="wikipedia.org", credibility_score=0.85)
        ]
    },
    {
        "claim": "The Great Wall of China is visible from low Earth orbit without magnification.",
        "ground_truth": VeracityLabel.CONTRADICTED,
        "type": "explicit_refutation_text",
        "evidence": [
            EvidencePassage(text="NASA astronauts have repeatedly confirmed that the Great Wall of China is not visible from low Earth orbit with the naked eye.", source_title="NASA", source_domain="nasa.gov", credibility_score=0.98),
            EvidencePassage(text="Urban legends frequently state that the Great Wall is visible from space.", source_title="BlogRumor", source_domain="reddit.com", credibility_score=0.30)
        ]
    },
    {
        "claim": "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
        "ground_truth": VeracityLabel.SUPPORTED,
        "type": "geographical_support",
        "evidence": [
            EvidencePassage(text="The Pacific Ocean is the largest and deepest of the world ocean basins, covering approximately 63 million square miles.", source_title="NOAA", source_domain="noaa.gov", credibility_score=0.98)
        ]
    },
    {
        "claim": "Thomas Edison invented the first commercial telephone.",
        "ground_truth": VeracityLabel.CONTRADICTED,
        "type": "entity_inventor_conflict",
        "evidence": [
            EvidencePassage(text="Alexander Graham Bell was awarded the first US patent for the telephone in 1876.", source_title="Library of Congress", source_domain="loc.gov", credibility_score=0.97)
        ]
    },
    {
        "claim": "Quantum computers have officially proven that P equals NP in 2024.",
        "ground_truth": VeracityLabel.INSUFFICIENT_EVIDENCE,
        "type": "unproven_premise_abstention",
        "evidence": [
            EvidencePassage(text="The P versus NP problem remains one of the seven Millennium Prize Problems with no generally accepted proof as of 2024.", source_title="Clay Mathematics Institute", source_domain="claymath.org", credibility_score=0.98)
        ]
    }
]

# =====================================================================
# BASELINES FOR SCIENTIFIC COMPARISON
# =====================================================================
class LexicalHeuristicBaseline:
    """
    Unweighted Lexical Heuristic Baseline:
    Simple keyword overlap and lexical negation search without context alignment gates or credibility weighting.
    """
    def verify(self, claim_text: str, evidence_list: List[EvidencePassage]) -> VeracityLabel:
        if not evidence_list:
            return VeracityLabel.INSUFFICIENT_EVIDENCE
        c_words = set(re.findall(r'\b\w{3,}\b', claim_text.lower()))
        votes = {"SUPPORT": 0, "CONTRADICT": 0, "NEUTRAL": 0}
        for ev in evidence_list:
            e_text = ev.text.lower()
            e_words = set(re.findall(r'\b\w{3,}\b', e_text))
            overlap = len(c_words.intersection(e_words)) / max(1, len(c_words))
            has_negation = any(neg in e_text for neg in ["not", "never", "disproven", "false", "disputed"])
            if overlap >= 0.35:
                if has_negation:
                    votes["CONTRADICT"] += 1
                else:
                    votes["SUPPORT"] += 1
            else:
                votes["NEUTRAL"] += 1

        if votes["CONTRADICT"] > votes["SUPPORT"]:
            return VeracityLabel.CONTRADICTED
        elif votes["SUPPORT"] > votes["CONTRADICT"]:
            return VeracityLabel.SUPPORTED
        else:
            return VeracityLabel.INSUFFICIENT_EVIDENCE

class DenseSimilarityBaseline:
    """
    Standard Dense Similarity Baseline:
    Cosine similarity over dense embeddings + unweighted majority voting
    (No RefNLI alignment gate, no source credibility weighting).
    """
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def verify(self, claim_text: str, evidence_list: List[EvidencePassage]) -> VeracityLabel:
        if not evidence_list:
            return VeracityLabel.INSUFFICIENT_EVIDENCE
        # Compute cosine similarities
        q_emb = np.array(list(self.retriever.embed_model.embed([claim_text])))
        e_embs = np.array(list(self.retriever.embed_model.embed([e.text for e in evidence_list])))
        sims = self.retriever._cosine_similarity(q_emb, e_embs)[0]
        
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])
        top_ev_text = evidence_list[top_idx].text.lower()

        has_neg = any(neg in top_ev_text for neg in ["not", "never", "disproven", "false", "fake", "refuted"])
        if top_sim >= 0.70:
            return VeracityLabel.CONTRADICTED if has_neg else VeracityLabel.SUPPORTED
        return VeracityLabel.INSUFFICIENT_EVIDENCE

# =====================================================================
# EVALUATION METRICS HELPERS
# =====================================================================
def compute_classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0

    cm = {l: {pred_l: 0 for pred_l in labels} for l in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    per_class = {}
    f1_list = []
    for l in labels:
        tp = cm[l][l]
        fp = sum(cm[other][l] for other in labels if other != l)
        fn = sum(cm[l][other] for other in labels if other != l)
        support = sum(cm[l].values())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[l] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": cm, "total": total}

# =====================================================================
# RUNNERS
# =====================================================================
def run_diagnostic_suite(pipeline: ProofGraphPipeline, baseline: LexicalHeuristicBaseline):
    print("=" * 78)
    print("🔬 SECTION 1: DIAGNOSTIC UNIT SANITY SUITE (10 Hand-Authored Stress Cases)")
    print("=" * 78)
    print(f"{'Claim Summary':<40} | {'Failure Mode Tested':<22} | {'GT':<11} | {'Baseline':<11} | {'PG':<11}")
    print("-" * 105)

    y_true, y_base, y_pg = [], [], []
    for item in DIAGNOSTIC_DATASET:
        c_text = item["claim"]
        gt = item["ground_truth"].value
        pipeline.index_corpus(item["evidence"])
        res = pipeline.verify(c_text, use_live_web_fallback=False)
        pg_pred = res.verdicts[res.claims[0].id].label.value if res.claims else "INSUFFICIENT_EVIDENCE"
        base_pred = baseline.verify(c_text, item["evidence"]).value

        y_true.append(gt)
        y_base.append(base_pred)
        y_pg.append(pg_pred)

        print(f"{c_text[:38] + '..':<40} | {item['type']:<22} | {gt:<11} | {base_pred:<11} | {pg_pred:<11}")

    labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    m_base = compute_classification_metrics(y_true, y_base, labels)
    m_pg = compute_classification_metrics(y_true, y_pg, labels)

    print("\n📊 Diagnostic Summary Results:")
    print(f"   • Lexical Heuristic Baseline Accuracy: {m_base['accuracy']*100:.1f}% | Macro-F1: {m_base['macro_f1']:.3f}")
    print(f"   • ProofGraph Engine Accuracy:          {m_pg['accuracy']*100:.1f}% | Macro-F1: {m_pg['macro_f1']:.3f}")

def run_evidence_retrieval_benchmark(sample_path: str):
    print("\n" + "=" * 78)
    print("🔍 SECTION 2: IN-BATCH EVIDENCE RETRIEVAL QUALITY DIAGNOSTIC (POOLED AVeriTeC, N=60, 170 PASSAGES)")
    print("=" * 78)
    print("Testing retrieval precision when all 170 gold evidence passages from 60 claims")
    print("are pooled into a single global index with cross-claim distractor passages.")

    with open(sample_path, "r") as f:
        data = json.load(f)

    pooled_passages = []
    gold_ids_per_claim = {}
    p_idx = 0
    for c_idx, d in enumerate(data):
        gold_ids = []
        for p in d.get("evidence_passages", []):
            p_id = f"p_{c_idx}_{p_idx}"
            p_idx += 1
            gold_ids.append(p_id)
            pooled_passages.append(EvidencePassage(
                id=p_id,
                text=p["text"],
                source_url=p.get("source_url", ""),
                source_title=p.get("source_title", "AVeriTeC Gold Evidence")
            ))
        gold_ids_per_claim[c_idx] = gold_ids

    retriever = HybridRetriever(prefer_offline=False)
    retriever.index_passages(pooled_passages)

    def eval_ir(search_fn, name):
        h1, h3, h5 = 0, 0, 0
        mrr_list = []
        for c_idx, d in enumerate(data):
            golds = set(gold_ids_per_claim[c_idx])
            retrieved_ids = search_fn(d["claim"], top_k=5)
            if any(r in golds for r in retrieved_ids[:1]): h1 += 1
            if any(r in golds for r in retrieved_ids[:3]): h3 += 1
            if any(r in golds for r in retrieved_ids[:5]): h5 += 1
            rr = 0.0
            for rank, r in enumerate(retrieved_ids, 1):
                if r in golds:
                    rr = 1.0 / rank
                    break
            mrr_list.append(rr)
        n = len(data)
        print(f"| {name:<30} | {h1/n*100:>6.1f}% | {h3/n*100:>6.1f}% | {h5/n*100:>6.1f}% | {np.mean(mrr_list):>6.3f} |")

    # 1. BM25 alone
    def bm25_search(q, top_k=5):
        tokens = retriever.tokenize(q)
        scores = retriever.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [retriever.passages[i].id for i in ranked]

    # 2. Pure Dense alone
    def dense_search(q, top_k=5):
        q_emb = np.array(list(retriever.embed_model.embed([q])))
        sims = retriever._cosine_similarity(q_emb, retriever.dense_embeddings)[0]
        ranked = np.argsort(sims)[::-1][:top_k]
        return [retriever.passages[i].id for i in ranked]

    # 3. Hybrid RRF
    def hybrid_search(q, top_k=5):
        res = retriever.search_local(q, top_k=top_k)
        return [r.id for r in res]

    print(f"\n| {'Retriever Configuration':<30} | {'Hits@1':<7} | {'Hits@3':<7} | {'Hits@5':<7} | {'MRR':<7} |")
    print(f"| :{'-'*29} | :{'-'*6} | :{'-'*6} | :{'-'*6} | :{'-'*6} |")
    eval_ir(bm25_search, "BM25 Lexical Alone")
    eval_ir(dense_search, "Pure Dense Semantic Alone")
    eval_ir(hybrid_search, "ProofGraph Hybrid Fusion (RRF)")

def run_averitec_benchmark(pipeline: ProofGraphPipeline, sample_path: str):
    print("\n" + "=" * 78)
    print("📚 SECTION 3: AVeriTeC DEV SAMPLE VERDICT CLASSIFICATION (N=60 CLAIMS, CANDIDATE POOLS)")
    print("=" * 78)
    print("Evaluation Setup: Verdict classification given claim-specific candidate evidence pools.")
    print("Protocol Note: Custom classification protocol (see docs/DATA_CARD.md); does not evaluate")
    print("full-corpus open-web document retrieval or official Hungarian METEOR overlap score.")
    print("Dataset: 60 real-world claims fact-checked across 50 professional organizations.")

    with open(sample_path, "r") as f:
        data = json.load(f)

    def map_label(raw: str) -> str:
        # Standard FEVER/NLI 3-way mapping:
        # 'Not Enough Evidence' & 'Conflicting Evidence' map to INSUFFICIENT_EVIDENCE
        if raw == "Refuted":
            return "CONTRADICTED"
        elif raw == "Supported":
            return "SUPPORTED"
        else:
            return "INSUFFICIENT_EVIDENCE"

    labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    y_true, y_pred = [], []
    raw_breakdown = defaultdict(lambda: {"SUPPORTED": 0, "CONTRADICTED": 0, "INSUFFICIENT_EVIDENCE": 0})

    for item in data:
        raw_label = item["label"]
        gt_mapped = map_label(raw_label)
        claim_text = item["claim"]
        evidence_objs = [
            EvidencePassage(
                text=p["text"],
                source_url=p.get("source_url", ""),
                source_title=p.get("source_title", "AVeriTeC Gold Evidence")
            )
            for p in item.get("evidence_passages", [])
        ]

        pipeline.index_corpus(evidence_objs)
        res = pipeline.verify(claim_text, use_live_web_fallback=False)
        pred = res.verdicts[res.claims[0].id].label.value if res.claims else "INSUFFICIENT_EVIDENCE"

        y_true.append(gt_mapped)
        y_pred.append(pred)
        raw_breakdown[raw_label][pred] += 1

    metrics = compute_classification_metrics(y_true, y_pred, labels)

    print(f"\n📊 Benchmark Metrics on AVeriTeC (N={metrics['total']} Claims):")
    print(f"   • Overall Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"   • Macro-F1:         {metrics['macro_f1']:.3f}")
    
    print("\n📈 Per-Class Breakdown (Standard 3-Way FEVER Mapping):")
    for l in labels:
        pc = metrics["per_class"][l]
        print(f"   • {l:<22}: Precision: {pc['precision']:.3f} | Recall: {pc['recall']:.3f} | F1: {pc['f1']:.3f} (N={pc['support']})")

    print("\n🧮 3x3 Confusion Matrix (Rows = Ground Truth, Columns = Predicted):")
    print(f"{'':<24} | {'Pred: SUPP':<12} | {'Pred: CONTRA':<12} | {'Pred: INSUFF':<12}")
    print("-" * 65)
    for l in labels:
        row = metrics["confusion_matrix"][l]
        print(f"{'True: ' + l:<24} | {row['SUPPORTED']:<12} | {row['CONTRADICTED']:<12} | {row['INSUFFICIENT_EVIDENCE']:<12}")

    print("\n🔍 4-Class Breakdown (Official AVeriTeC Raw Classes vs. 3-Way Predicted Stance):")
    print(f"{'Official AVeriTeC Label':<35} | {'N':<4} | {'Pred: SUPP':<10} | {'Pred: CONTRA':<12} | {'Pred: INSUFF':<12}")
    print("-" * 80)
    for raw_l, counts in raw_breakdown.items():
        total_raw = sum(counts.values())
        print(f"{raw_l:<35} | {total_raw:<4} | {counts['SUPPORTED']:<10} | {counts['CONTRADICTED']:<12} | {counts['INSUFFICIENT_EVIDENCE']:<12}")

def run_ablation_and_baselines(sample_path: str):
    print("\n" + "=" * 78)
    print("🔬 COMPONENT-BY-COMPONENT ABLATION & BASELINE COMPARISON ON AVeriTeC")
    print("=" * 78)

    with open(sample_path, "r") as f:
        data = json.load(f)

    def map_label(raw: str) -> str:
        return "CONTRADICTED" if raw == "Refuted" else ("SUPPORTED" if raw == "Supported" else "INSUFFICIENT_EVIDENCE")

    labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]

    pipe_rules = ProofGraphPipeline(verifier_backend="rules")
    pipe_hybrid = ProofGraphPipeline(verifier_backend="hybrid")
    pipe_no_align = ProofGraphPipeline(verifier=NLIVerifier(refnli_gate=RefNLIAlignmentGate(jaccard_threshold=0.0)))
    base_lexical = LexicalHeuristicBaseline()
    base_dense = DenseSimilarityBaseline(retriever=pipe_rules.retriever)

    y_true = [map_label(d["label"]) for d in data]
    preds_rules, preds_hybrid, preds_no_align, preds_lex, preds_dense = [], [], [], [], []

    for item in data:
        claim_text = item["claim"]
        evidence_objs = [EvidencePassage(text=p["text"], source_title="AVeriTeC") for p in item.get("evidence_passages", [])]

        pipe_rules.index_corpus(evidence_objs)
        r1 = pipe_rules.verify(claim_text, use_live_web_fallback=False)
        preds_rules.append(r1.verdicts[r1.claims[0].id].label.value if r1.claims else "INSUFFICIENT_EVIDENCE")

        pipe_hybrid.index_corpus(evidence_objs)
        r_hyb = pipe_hybrid.verify(claim_text, use_live_web_fallback=False)
        preds_hybrid.append(r_hyb.verdicts[r_hyb.claims[0].id].label.value if r_hyb.claims else "INSUFFICIENT_EVIDENCE")

        pipe_no_align.index_corpus(evidence_objs)
        r2 = pipe_no_align.verify(claim_text, use_live_web_fallback=False)
        preds_no_align.append(r2.verdicts[r2.claims[0].id].label.value if r2.claims else "INSUFFICIENT_EVIDENCE")

        preds_lex.append(base_lexical.verify(claim_text, evidence_objs).value)
        preds_dense.append(base_dense.verify(claim_text, evidence_objs).value)

    m_rules = compute_classification_metrics(y_true, preds_rules, labels)
    m_hybrid = compute_classification_metrics(y_true, preds_hybrid, labels)
    m_no_align = compute_classification_metrics(y_true, preds_no_align, labels)
    m_lex = compute_classification_metrics(y_true, preds_lex, labels)
    m_dense = compute_classification_metrics(y_true, preds_dense, labels)

    print("\n| Configuration                          | Accuracy | Macro-F1 | Prec (Supp) | Prec (Contra) | Rec (Contra) |")
    print("| :------------------------------------- | :------: | :------: | :---------: | :-----------: | :----------: |")
    print(f"| **ProofGraph Hybrid (Rules + Neural)** | **{m_hybrid['accuracy']*100:.1f}%** |  **{m_hybrid['macro_f1']:.3f}**  |    {m_hybrid['per_class']['SUPPORTED']['precision']:.3f}    |     {m_hybrid['per_class']['CONTRADICTED']['precision']:.3f}     |    **{m_hybrid['per_class']['CONTRADICTED']['recall']:.3f}**   |")
    print(f"| ProofGraph Rules Alone                 |  {m_rules['accuracy']*100:.1f}%  |   {m_rules['macro_f1']:.3f}  |    {m_rules['per_class']['SUPPORTED']['precision']:.3f}    |     {m_rules['per_class']['CONTRADICTED']['precision']:.3f}     |    {m_rules['per_class']['CONTRADICTED']['recall']:.3f}   |")
    print(f"| Ablation: Without Alignment Gate       |  {m_no_align['accuracy']*100:.1f}%  |   {m_no_align['macro_f1']:.3f}  |    {m_no_align['per_class']['SUPPORTED']['precision']:.3f}    |     {m_no_align['per_class']['CONTRADICTED']['precision']:.3f}     |    {m_no_align['per_class']['CONTRADICTED']['recall']:.3f}   |")
    print(f"| Baseline 1: Dense Similarity (No Gate) |  {m_dense['accuracy']*100:.1f}%  |   {m_dense['macro_f1']:.3f}  |    {m_dense['per_class']['SUPPORTED']['precision']:.3f}    |     {m_dense['per_class']['CONTRADICTED']['precision']:.3f}     |    {m_dense['per_class']['CONTRADICTED']['recall']:.3f}   |")
    print(f"| Baseline 2: Lexical Heuristic Baseline |  {m_lex['accuracy']*100:.1f}%  |   {m_lex['macro_f1']:.3f}  |    {m_lex['per_class']['SUPPORTED']['precision']:.3f}    |     {m_lex['per_class']['CONTRADICTED']['precision']:.3f}     |    {m_lex['per_class']['CONTRADICTED']['recall']:.3f}   |")
    print("=" * 78)

def run_held_out_context_mismatch_benchmark():
    print("\n" + "=" * 78)
    print("🔬 SECTION 4: SYNTHETIC CONTEXT-MISMATCH STRESS DIAGNOSTIC (N=20, Hand-Authored)")
    print("=" * 78)
    print("Unit diagnostic inspired by RefNLI reference indeterminacy failure modes (N=20 pairs).")
    print("Disclosure: This is a synthetic stress test, not the full 1,143-pair RefNLI corpus (see docs/DATA_CARD.md).")
    print("Evaluating the alignment gate on this hand-authored synthetic suite (not held out or RefNLI):")

    CONTEXT_MISMATCH_BENCHMARK = [
        # 10 Aligned Pairs (Ground Truth: Supported)
        {"c": "Marie Curie won the Nobel Prize in Chemistry in 1911.", "ents": ["Marie Curie", "1911"], "p": "Marie Curie was awarded the 1911 Nobel Prize in Chemistry for discovering radium.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "Albert Einstein published General Relativity in 1915.", "ents": ["Albert Einstein", "1915"], "p": "Albert Einstein formulated General Relativity in Berlin in 1915.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "Alan Turing cracked the Enigma code at Bletchley Park.", "ents": ["Alan Turing", "Bletchley Park"], "p": "Alan Turing worked at Bletchley Park deciphering German Enigma encryptions.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "Apollo 11 landed on the moon on July 20, 1969.", "ents": ["Apollo 11", "1969"], "p": "The Apollo 11 mission landed humans on the Moon on July 20, 1969.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "Alexander Fleming discovered penicillin in 1928.", "ents": ["Alexander Fleming", "1928"], "p": "Alexander Fleming discovered penicillin at St. Marys Hospital in 1928.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "James Watson and Francis Crick proposed the double helix structure of DNA in 1953.", "ents": ["James Watson", "Francis Crick", "1953"], "p": "In 1953, James Watson and Francis Crick published the double helix structure of DNA.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "Charles Darwin published On the Origin of Species in 1859.", "ents": ["Charles Darwin", "1859"], "p": "Charles Darwin published On the Origin of Species in November 1859.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "Ada Lovelace wrote the first computer algorithm in 1843.", "ents": ["Ada Lovelace", "1843"], "p": "Ada Lovelace is regarded as the first computer programmer after publishing an algorithm in 1843.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "Johannes Gutenberg invented the movable-type printing press around 1440.", "ents": ["Johannes Gutenberg", "1440"], "p": "Johannes Gutenberg introduced movable type to Europe around 1440 in Mainz.", "aligned": True, "gt": VeracityLabel.SUPPORTED},
        {"c": "Wright brothers conducted the first powered flight in 1903 at Kitty Hawk.", "ents": ["Wright brothers", "1903", "Kitty Hawk"], "p": "Orville and Wilbur Wright achieved the first sustained powered flight at Kitty Hawk in 1903.", "aligned": True, "gt": VeracityLabel.SUPPORTED},

        # 10 Reference Indeterminacy Distractors (Ground Truth: Insufficient Evidence)
        {"c": "Peter was born in 1995 in Toronto.", "ents": ["Peter", "1995", "Toronto"], "p": "William was born in 1980 in Toronto to diplomat parents.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "Dr. Smith invented the cardiac stent in 1985.", "ents": ["Dr. Smith", "1985"], "p": "Dr. Charles Dotter pioneered modern angioplasty procedures in 1964.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "Acme Corp filed for Chapter 11 bankruptcy in 2023.", "ents": ["Acme Corp", "2023"], "p": "Beta LLC filed for Chapter 11 bankruptcy reorganization in 2023.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "Sarah Jenkins won the Boston Marathon in 2018.", "ents": ["Sarah Jenkins", "2018"], "p": "Desiree Linden won the Boston Marathon in 2018 amid heavy rains.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "Apex Motors released their electric truck in 2022.", "ents": ["Apex Motors", "2022"], "p": "Rivian Automotive delivered their first electric truck in 2021.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "Professor Vance published a study on dark matter in 2020.", "ents": ["Professor Vance", "2020"], "p": "Dr. Katherine Freese published pioneering research on dark matter candidates in 2020.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "Delta Air Flight 402 crashed near Denver in 1999.", "ents": ["Delta Air", "1999", "Denver"], "p": "United Airlines Flight 585 crashed near Colorado Springs in 1991.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "David Miller was elected mayor of Chicago in 2019.", "ents": ["David Miller", "2019", "Chicago"], "p": "Lori Lightfoot was elected mayor of Chicago in 2019.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "Zenith Biotech received FDA approval for their vaccine in 2021.", "ents": ["Zenith Biotech", "2021"], "p": "Pfizer and BioNTech received full FDA approval for their COVID-19 vaccine in 2021.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE},
        {"c": "Marcus Aurelius completed the Meditations in 175 AD in Rome.", "ents": ["Marcus Aurelius", "175", "Rome"], "p": "Emperor Septimius Severus campaigned across Britannia in 208 AD.", "aligned": False, "gt": VeracityLabel.INSUFFICIENT_EVIDENCE}
    ]

    v_with_gate = NLIVerifier(refnli_gate=RefNLIAlignmentGate(jaccard_threshold=0.12))
    
    class UnGatedFilter:
        def check_alignment(self, c, e):
            return True, 0.5, "Ablated: always aligned"
    v_without_gate = NLIVerifier()
    v_without_gate.alignment_gate = UnGatedFilter()

    correct_with, false_contra_with = 0, 0
    correct_without, false_contra_without = 0, 0

    for item in CONTEXT_MISMATCH_BENCHMARK:
        c = AtomicClaim(text=item["c"], parent_sentence="", entities=item["ents"])
        p = EvidencePassage(text=item["p"], source_title="Test Source")
        gt = item["gt"]

        vw = v_with_gate.verify_claim(c, [p])
        if vw.label == gt: correct_with += 1
        if not item["aligned"] and vw.label == VeracityLabel.CONTRADICTED: false_contra_with += 1

        vwo = v_without_gate.verify_claim(c, [p])
        if vwo.label == gt: correct_without += 1
        if not item["aligned"] and vwo.label == VeracityLabel.CONTRADICTED: false_contra_without += 1

    n = len(CONTEXT_MISMATCH_BENCHMARK)
    n_mismatches = sum(1 for item in CONTEXT_MISMATCH_BENCHMARK if not item["aligned"])

    print(f"\n| Gate Configuration                      | Accuracy (N=20) | False Refutations (Mismatches N=10) |")
    print(f"| :-------------------------------------- | :-------------: | :----------------------------------: |")
    print(f"| **ProofGraph (With RefNLI Gate)**       | **{correct_with/n*100:.1f}% ({correct_with}/{n})** | **{false_contra_with}/{n_mismatches} ({false_contra_with/n_mismatches*100:.1f}%)**           |")
    print(f"| Ablation (WITHOUT RefNLI Gate)          | {correct_without/n*100:.1f}% ({correct_without}/{n})  | {false_contra_without}/{n_mismatches} ({false_contra_without/n_mismatches*100:.1f}%)             |")
    print("=" * 78)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProofGraph Benchmark & Diagnostic Evaluation Suite")
    parser.add_argument("--backend", choices=["rules", "hybrid", "neural", "all"], default="all", help="Verifier backend to evaluate")
    parser.add_argument("--sample-file", default=None, help="Path to AVeriTeC sample JSON")
    args = parser.parse_args()

    sample_file = args.sample_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "benchmarks", "averitec_dev_sample.json")
    selected_backend = "hybrid" if args.backend in ("hybrid", "all") else args.backend
    eval_pipe = ProofGraphPipeline(verifier_backend=selected_backend)
    base_lexical = LexicalHeuristicBaseline()

    run_diagnostic_suite(eval_pipe, base_lexical)
    run_evidence_retrieval_benchmark(sample_file)
    run_averitec_benchmark(eval_pipe, sample_file)
    run_ablation_and_baselines(sample_file)
    run_held_out_context_mismatch_benchmark()
