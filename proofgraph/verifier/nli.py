import re
from typing import List, Tuple, Set
from proofgraph.schemas.claim import AtomicClaim
from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.verdict import VeracityLabel, ClaimVerdict
from proofgraph.verifier.alignment import RefNLIAlignmentGate
from proofgraph.verifier.credibility import SourceCredibilityScorer

class RuleStanceVerifier:
    """
    Deterministic Rule-Based Stance Engine:
    Performs 3-way claim verification (SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE)
    using:
    1. Lexical-entity context alignment filtering (RefNLI-inspired gate)
    2. Canonical domain authority prior (CONFACT-inspired source weighting)
    3. Predicate-bound numerical, temporal, and entity contradiction checks
    4. Calibrated abstention for ambiguous, speculative, or unaligned evidence
    """
    def __init__(self, refnli_gate: RefNLIAlignmentGate = None, credibility_scorer: SourceCredibilityScorer = None):
        self.alignment_gate = refnli_gate or RefNLIAlignmentGate()
        self.credibility_scorer = credibility_scorer or SourceCredibilityScorer()

    # Negation and refutation keywords
    REFUTATION_MARKERS = [

        r"\b(?:not|never|neither|nor|falsely|incorrectly|debunked|denied|refuted|untrue|fake)\b",
        r"\b(?:failed to|did not|was not|were not|is not|are not|no evidence)\b",
        r"\b(?:contrary to|disproven|disputed|misleading|hoax|myth)\b"
    ]

    # Hedging / speculation markers (reduces entailment confidence)
    HEDGE_MARKERS = [
        r"\b(?:allegedly|rumored|supposedly|purportedly|claimed by some|secretly|replica|conspiracy)\b"
    ]

    # Common location pairs known to be distinct
    KNOWN_DISTINCT_LOCATIONS = [
        {"rome", "paris"},
        {"london", "paris"},
        {"new york", "los angeles"},
        {"tokyo", "beijing"},
        {"germany", "france"},
        {"italy", "france"},
        {"russia", "ukraine"},
        {"india", "china"}
    ]

    def _has_location_conflict(self, text_a: str, text_b: str) -> bool:
        """
        Detects if two texts assign conflicting primary geographical locations to the same subject.
        Requires presence of location prepositions ('in', 'at', 'located in', 'built in')
        to avoid flagging transit or co-mentioned locations.
        """
        words_a = set(re.findall(r'\b[a-z]+\b', text_a.lower()))
        words_b = set(re.findall(r'\b[a-z]+\b', text_b.lower()))
        for loc_pair in self.KNOWN_DISTINCT_LOCATIONS:
            if len(loc_pair.intersection(words_a)) == 1 and len(loc_pair.intersection(words_b)) == 1:
                if loc_pair.intersection(words_a) != loc_pair.intersection(words_b):
                    # Check for location binding context in both texts
                    loc_a = list(loc_pair.intersection(words_a))[0]
                    loc_b = list(loc_pair.intersection(words_b))[0]
                    pat_a = rf'\b(?:in|at|near|located|built|erected)\s+(?:\w+\s+)?{loc_a}\b'
                    pat_b = rf'\b(?:in|at|near|located|built|erected)\s+(?:\w+\s+)?{loc_b}\b'
                    if re.search(pat_a, text_a.lower()) or re.search(pat_b, text_b.lower()):
                        return True
        return False

    def _has_numeric_conflict(self, text_a: str, text_b: str) -> bool:
        """
        Detects conflicting numerical quantities or temperatures.
        Requires shared predicate/event context (e.g. both describe boiling, birth, construction, or patent)
        to prevent unrelated numbers (e.g. birth year vs award year) from triggering false contradictions.
        """
        predicates = [
            "boil", "freez", "born", "died", "construct", "built", "open", 
            "land", "discover", "invent", "found", "launch", "elect", "crash",
            "patent", "temperat", "celsius", "fahrenheit", "meter", "foot", "feet"
        ]
        # Check if any shared event/unit predicate exists between both texts
        shared_pred = any(p in text_a.lower() and p in text_b.lower() for p in predicates)
        if not shared_pred:
            return False

        nums_a = set(re.findall(r'\b\d+(?:\.\d+)?\b', text_a))
        nums_b = set(re.findall(r'\b\d+(?:\.\d+)?\b', text_b))
        if nums_a and nums_b and nums_a.isdisjoint(nums_b):
            return True
        return False


    def _has_named_inventor_conflict(self, text_a: str, text_b: str) -> bool:
        """Detects if claim attributes invention/discovery to Person A while evidence attributes to Person B."""
        inventor_markers = ["invent", "discover", "found", "creat", "patent", "award", "pioneer"]
        if any(m in text_a.lower() for m in inventor_markers) and any(m in text_b.lower() for m in inventor_markers):
            # Check capitalized proper names
            names_a = set(re.findall(r'\b[A-Z][a-z]+\b', text_a)) - {"The", "In", "On", "At", "First"}
            names_b = set(re.findall(r'\b[A-Z][a-z]+\b', text_b)) - {"The", "In", "On", "At", "First", "US"}
            if names_a and names_b and names_a.isdisjoint(names_b):
                return True
        return False

    def _assess_pairwise(self, claim: AtomicClaim, evidence: EvidencePassage) -> Tuple[str, float, str]:
        """
        Assesses pairwise relationship between claim and evidence passage.
        Returns: (stance: 'SUPPORT' | 'CONTRADICT' | 'NEUTRAL', raw_confidence, reason)
        """
        # 1. RefNLI Context Alignment Gate
        is_aligned, jaccard, reason = self.alignment_gate.check_alignment(claim, evidence)
        if not is_aligned:
            return "NEUTRAL", 0.0, f"Filtered by RefNLI ({reason})"

        c_text = claim.text
        e_text = evidence.text

        # Check for explicit refutation markers
        has_refutation = any(re.search(pat, e_text.lower()) for pat in self.REFUTATION_MARKERS)
        
        # Check for numeric / date conflicts
        numeric_conflict = self._has_numeric_conflict(c_text, e_text)

        # Check for geographical location conflicts (e.g. Rome vs Paris)
        location_conflict = self._has_location_conflict(c_text, e_text)

        # Check for entity / inventor conflicts (e.g. Edison vs Bell)
        inventor_conflict = self._has_named_inventor_conflict(c_text, e_text)

        # Semantic keywords overlap
        c_words = set(re.findall(r'\b[a-z]{3,}\b', c_text.lower()))
        e_words = set(re.findall(r'\b[a-z]{3,}\b', e_text.lower()))
        overlap = c_words.intersection(e_words)
        overlap_ratio = len(overlap) / len(c_words) if c_words else 0.0

        # Contradiction Resolution
        if (has_refutation and overlap_ratio >= 0.25) or numeric_conflict or location_conflict or inventor_conflict:
            conf = 0.92 if (numeric_conflict or location_conflict or inventor_conflict) else 0.82
            details = []
            if numeric_conflict: details.append("numerical / quantitative discrepancy")
            if location_conflict: details.append("geographical conflict")
            if inventor_conflict: details.append("conflicting entity attribution")
            if has_refutation: details.append("explicit refuting language")
            return "CONTRADICT", conf, f"Contradiction detected: {', '.join(details)}"

        # Check for hedging / unverified speculation in low-credibility sources
        is_hedged = any(re.search(pat, e_text.lower()) for pat in self.HEDGE_MARKERS)

        if overlap_ratio >= 0.40:
            conf = min(0.95, 0.55 + overlap_ratio * 0.40)
            if is_hedged:
                conf *= 0.5
                return "NEUTRAL", conf, f"Passage contains speculative phrasing (overlap: {overlap_ratio:.2f})"
            return "SUPPORT", conf, f"Direct semantic alignment (overlap: {overlap_ratio:.2f})"

        return "NEUTRAL", 0.3, "Insufficient direct semantic entailment in passage"

    def verify_claim(self, claim: AtomicClaim, evidence_list: List[EvidencePassage]) -> ClaimVerdict:
        """
        Aggregates evidence for a single atomic claim, applying source credibility weights
        and RefNLI gates to produce a definitive calibrated ClaimVerdict.
        """
        if not claim.is_verifiable:
            return ClaimVerdict(
                claim_id=claim.id,
                label=VeracityLabel.INSUFFICIENT_EVIDENCE,
                confidence=1.0,
                justification="Claim was classified as subjective or non-verifiable opinion.",
                refnli_passed=True
            )

        if not evidence_list:
            return ClaimVerdict(
                claim_id=claim.id,
                label=VeracityLabel.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                justification="No relevant evidence was retrieved from any source.",
                refnli_passed=False
            )

        support_weight = 0.0
        contradict_weight = 0.0
        supporting_ids = []
        contradicting_ids = []
        reasons = []

        for ev in evidence_list:
            cred = self.credibility_scorer.get_credibility(ev.source_domain or ev.source_url or "")
            ev.credibility_score = cred
            
            stance, raw_conf, reason = self._assess_pairwise(claim, ev)
            reasons.append(f"[{ev.source_title}]: {stance} (cred: {cred:.2f}) - {reason}")
            
            # CONFACT Weighted Vote: confidence * source_credibility * retrieval_relevance
            weighted_score = raw_conf * cred * max(0.4, ev.relevance_score)
            
            if stance == "SUPPORT":
                support_weight += weighted_score
                supporting_ids.append(ev.id)
            elif stance == "CONTRADICT":
                contradict_weight += weighted_score
                contradicting_ids.append(ev.id)

        # Verdict Resolution Logic
        if support_weight == 0 and contradict_weight == 0:
            final_label = VeracityLabel.INSUFFICIENT_EVIDENCE
            confidence = 0.3
            justification = f"Evidence was retrieved but failed contextual alignment gates: {'; '.join(reasons[:2])}"
        elif contradict_weight > support_weight:
            final_label = VeracityLabel.CONTRADICTED
            confidence = min(0.95, contradict_weight / (support_weight + contradict_weight + 1e-6))
            justification = f"Refuted by authoritative evidence ({len(contradicting_ids)} contradicting sources vs {len(supporting_ids)} supporting)."
        elif support_weight > contradict_weight:
            final_label = VeracityLabel.SUPPORTED
            confidence = min(0.95, support_weight / (support_weight + contradict_weight + 1e-6))
            justification = f"Supported by authoritative evidence ({len(supporting_ids)} supporting sources vs {len(contradicting_ids)} contradicting)."
        else:
            final_label = VeracityLabel.INSUFFICIENT_EVIDENCE
            confidence = 0.5
            justification = f"Conflicting evidence from competing sources with comparable weight (Support: {support_weight:.2f}, Contradict: {contradict_weight:.2f})."

        return ClaimVerdict(
            claim_id=claim.id,
            label=final_label,
            confidence=float(confidence),
            justification=justification,
            supporting_evidence_ids=supporting_ids,
            contradicting_evidence_ids=contradicting_ids,
            refnli_passed=len(supporting_ids) + len(contradicting_ids) > 0,
            credibility_adjusted=True
        )

# Backward-compatibility alias
NLIVerifier = RuleStanceVerifier

