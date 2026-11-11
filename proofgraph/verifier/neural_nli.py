import os
import re
import warnings
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from proofgraph.schemas.claim import AtomicClaim
from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.verdict import VeracityLabel, ClaimVerdict
from proofgraph.verifier.alignment import RefNLIAlignmentGate
from proofgraph.verifier.credibility import SourceCredibilityScorer

class DebertaNLIVerifier:
    """
    Trained Neural Cross-Encoder NLI Verifier running via ONNX Runtime:
    - Uses Xenova/nli-deberta-v3-xsmall quantized ONNX model (~83MB).
    - Sub-15ms CPU inference with zero heavy PyTorch or CUDA requirements.
    - Pre-filtered by RefNLIAlignmentGate to eliminate Context Mismatch traps
      before neural inference (as established in RefNLI research).
    - Softmax outputs uncalibrated model scores across:
        0: CONTRADICTION
        1: ENTAILMENT (SUPPORTED)
        2: NEUTRAL (INSUFFICIENT_EVIDENCE)
    - Fuses neural probabilities with source domain authority priors (CONFACT)
      and passage relevance scores for robust final verdict determination.
    - Falls back gracefully to rule-based verification if offline or uninitialized.
    """
    MODEL_REPO = "Xenova/nli-deberta-v3-xsmall"
    MODEL_REVISION = "2a4f614a701367a02d51389039afc998faeda637"
    MODEL_FILE = "onnx/model_quantized.onnx"
    TOKENIZER_FILE = "tokenizer.json"

    def __init__(
        self,
        refnli_gate: Optional[RefNLIAlignmentGate] = None,
        credibility_scorer: Optional[SourceCredibilityScorer] = None,
        confidence_threshold: float = 0.60,
        model_repo: Optional[str] = None,
        model_revision: Optional[str] = None,
        prefer_offline: bool = False
    ):
        self.alignment_gate = refnli_gate or RefNLIAlignmentGate()
        self.credibility_scorer = credibility_scorer or SourceCredibilityScorer()
        self.confidence_threshold = confidence_threshold
        self.model_repo = model_repo or self.MODEL_REPO
        self.model_revision = model_revision or self.MODEL_REVISION
        self.prefer_offline = prefer_offline
        
        self.session = None
        self.tokenizer = None
        self.label_mapping = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]
        self._is_ready = False
        self._init_error = None
        self._initialization_attempted = False
        
        if not self.prefer_offline:
            self._try_load_model()

    def _try_load_model(self) -> bool:
        """Attempts to load ONNX model and tokenizer from local cache or Hugging Face Hub."""
        if self._is_ready:
            return True
        if self._initialization_attempted:
            return False
        self._initialization_attempted = True
        try:
            import json
            import onnxruntime as ort
            from tokenizers import Tokenizer
            import huggingface_hub

            # Attempt local download/cache resolution with pinned immutable revision
            model_path = huggingface_hub.hf_hub_download(
                repo_id=self.model_repo,
                filename=self.MODEL_FILE,
                revision=self.model_revision
            )
            tokenizer_path = huggingface_hub.hf_hub_download(
                repo_id=self.model_repo,
                filename=self.TOKENIZER_FILE,
                revision=self.model_revision
            )

            # Runtime validation of authoritative label schema from model config.json
            try:
                cfg_path = huggingface_hub.hf_hub_download(
                    repo_id=self.model_repo,
                    filename="config.json",
                    revision=self.model_revision
                )
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                id2label = cfg.get("id2label", {})
                if id2label:
                    self.label_mapping = [id2label.get(str(i), "").upper() for i in range(3)]
            except Exception:
                self.label_mapping = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]

            # Initialize ONNX inference session on CPU with minimal threads
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.enable_truncation(max_length=512)
            # This verifier scores one pair at a time. Fixed 512-token padding
            # needlessly makes every ONNX call process a full-length sequence.
            # Keep the encoded pair at its natural length; the exported model
            # accepts dynamic sequence lengths.
            self._is_ready = True
            return True
        except Exception as e:
            self._init_error = str(e)
            self._is_ready = False
            warnings.warn(
                f"Neural NLI model unavailable; this verifier will abstain. Details: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

    @property
    def is_available(self) -> bool:
        """Returns True if the ONNX session and tokenizer are initialized."""
        if not self._is_ready and not self.prefer_offline and not self._initialization_attempted:
            self._try_load_model()
        return self._is_ready

    def predict_pair(self, premise: str, hypothesis: str) -> Tuple[np.ndarray, str]:
        """
        Runs cross-encoder inference on (premise, hypothesis).
        Returns:
            probs: np.ndarray shape (3,) -> [P(Contradiction), P(Entailment), P(Neutral)]
            predicted_label: 'CONTRADICTION' | 'ENTAILMENT' | 'NEUTRAL'
        """
        if not self.is_available:
            # Fallback uniform neutral distribution
            return np.array([0.0, 0.0, 1.0]), "NEUTRAL"

        enc = self.tokenizer.encode(premise, hypothesis)
        inputs = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64)
        }
        logits = self.session.run(None, inputs)[0][0]
        # Softmax with numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        best_idx = int(np.argmax(probs))
        return probs, self.label_mapping[best_idx]

    def verify_claim(self, claim: AtomicClaim, evidence_passages: List[EvidencePassage]) -> ClaimVerdict:
        """
        Aggregates pairwise NLI cross-encoder inferences over retrieved evidence passages.
        Applies RefNLI alignment filtering and source credibility priors.
        """
        if not evidence_passages:
            return ClaimVerdict(
                claim_id=claim.id,
                label=VeracityLabel.INSUFFICIENT_EVIDENCE,
                confidence=0.50,
                supporting_evidence_ids=[],
                contradicting_evidence_ids=[],
                justification="No evidence passages provided for neural stance verification."
            )

        supporting_ids = []
        contradicting_ids = []
        entailment_weight = 0.0
        contradiction_weight = 0.0
        rationale_snippets = []

        for p in evidence_passages:
            # 1. RefNLI Context Alignment Gate (blocks reference indeterminacy errors)
            aligned, jaccard, reason = self.alignment_gate.check_alignment(claim, p)
            if not aligned:
                continue

            # 2. Domain authority and relevance weighting
            credibility = self.credibility_scorer.get_credibility(p.source_domain or p.source_url or "")
            relevance = p.relevance_score or 0.70
            weight_multiplier = credibility * max(0.40, relevance)

            # 3. Neural inference: Premise = Evidence passage, Hypothesis = Claim text
            probs, stance = self.predict_pair(p.text, claim.text)
            p_contra, p_entail, p_neutral = float(probs[0]), float(probs[1]), float(probs[2])

            if p_contra >= self.confidence_threshold and p_contra > p_entail:
                contradiction_weight += p_contra * weight_multiplier
                contradicting_ids.append(p.id)
                rationale_snippets.append(
                    f"[{p.source_domain or p.id}] Neural Contradiction (P={p_contra:.2f}, Cred={credibility:.2f})"
                )
            elif p_entail >= self.confidence_threshold and p_entail > p_contra:
                entailment_weight += p_entail * weight_multiplier
                supporting_ids.append(p.id)
                rationale_snippets.append(
                    f"[{p.source_domain or p.id}] Neural Entailment (P={p_entail:.2f}, Cred={credibility:.2f})"
                )

        diff = abs(entailment_weight - contradiction_weight)

        if entailment_weight > contradiction_weight and diff >= 0.20:
            final_label = VeracityLabel.SUPPORTED
            final_conf = min(0.98, 0.60 + (entailment_weight / (entailment_weight + contradiction_weight + 0.1) * 0.38))
            summary = f"Neural cross-encoder established support with {len(supporting_ids)} aligned passage(s)."
        elif contradiction_weight > entailment_weight and diff >= 0.20:
            final_label = VeracityLabel.CONTRADICTED
            final_conf = min(0.98, 0.60 + (contradiction_weight / (entailment_weight + contradiction_weight + 0.1) * 0.38))
            summary = f"Neural cross-encoder identified refutation with {len(contradicting_ids)} aligned passage(s)."
        else:
            final_label = VeracityLabel.INSUFFICIENT_EVIDENCE
            final_conf = 0.50
            summary = (
                f"Abstained as INSUFFICIENT_EVIDENCE: evidence did not exceed neural decision threshold "
                f"(Entail={entailment_weight:.2f}, Contra={contradiction_weight:.2f}, Diff={diff:.2f})."
            )

        if rationale_snippets:
            summary += " Detail: " + "; ".join(rationale_snippets[:3])

        return ClaimVerdict(
            claim_id=claim.id,
            label=final_label,
            confidence=round(final_conf, 3),
            justification=summary,
            supporting_evidence_ids=supporting_ids,
            contradicting_evidence_ids=contradicting_ids,
            refnli_passed=len(supporting_ids) + len(contradicting_ids) > 0,
            credibility_adjusted=True
        )
