import re
from typing import Set, Tuple
from proofgraph.schemas.claim import AtomicClaim
from proofgraph.schemas.evidence import EvidencePassage

class RefNLIAlignmentGate:
    """
    Implements the RefNLI (arXiv:2502.05793) reference determinacy filter:
    Prevents false contradictions caused by NLI models assuming premise and hypothesis
    discuss the same entity when retrieved evidence actually refers to a different context.
    """
    def __init__(self, jaccard_threshold: float = 0.15):
        self.jaccard_threshold = jaccard_threshold

    def _tokenize_set(self, text: str) -> Set[str]:
        # Filter out common stop words and generic biographical verbs
        stopwords = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", 
            "is", "was", "are", "were", "born", "died", "became", "had", "has", "been"
        }
        tokens = set(re.findall(r'\b[a-z0-9]+\b', text.lower()))
        return tokens - stopwords

    def check_alignment(self, claim: AtomicClaim, evidence: EvidencePassage) -> Tuple[bool, float, str]:
        """
        Evaluates whether evidence is contextually aligned with the claim.
        Returns: (is_aligned, jaccard_score, explanation)
        """
        claim_tokens = self._tokenize_set(claim.text)
        evidence_tokens = self._tokenize_set(evidence.text)
        
        if not claim_tokens or not evidence_tokens:
            return False, 0.0, "Empty token sets"

        intersection = claim_tokens.intersection(evidence_tokens)
        union = claim_tokens.union(evidence_tokens)
        jaccard = len(intersection) / len(union) if union else 0.0

        # Check entity overlap
        entity_match = False
        if claim.entities:
            for ent in claim.entities:
                if ent.lower() in evidence.text.lower():
                    entity_match = True
                    break
            # If claim has explicit entities and none match in evidence, enforce stricter boundary
            if not entity_match and jaccard < 0.25:
                return False, jaccard, f"Context Mismatch: Entities {claim.entities} absent and Jaccard ({jaccard:.3f}) < 0.25"
        else:
            entity_match = True

        if entity_match or jaccard >= self.jaccard_threshold:
            return True, jaccard, "Aligned: entity or lexical threshold satisfied"
        
        return False, jaccard, f"Context Mismatch: Jaccard ({jaccard:.3f}) below threshold ({self.jaccard_threshold}) and no entity match"
