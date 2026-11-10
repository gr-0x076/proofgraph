from typing import List
from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.claim import AtomicClaim

class FIREStoppingPolicy:
    """
    Implements FIRE-style (NAACL 2025) dynamic stopping logic:
    Instead of blindly retrieving a fixed N passages, evaluates whether current
    evidence is sufficient to terminate search, reducing API cost and noise.
    """
    def __init__(self, confidence_threshold: float = 0.75, max_hops: int = 3):
        self.confidence_threshold = confidence_threshold
        self.max_hops = max_hops

    def should_terminate(self, claim: AtomicClaim, retrieved: List[EvidencePassage], current_hop: int) -> bool:
        """Determines if the search loop should terminate."""
        if current_hop >= self.max_hops:
            return True
        if not retrieved:
            return False
            
        top_passage = retrieved[0]
        # Check if top passage exceeds confidence threshold and covers at least one key entity
        has_entity_match = False
        if claim.entities:
            for ent in claim.entities:
                if ent.lower() in top_passage.text.lower():
                    has_entity_match = True
                    break
        else:
            has_entity_match = True

        if top_passage.relevance_score >= self.confidence_threshold and has_entity_match:
            return True
            
        return False
