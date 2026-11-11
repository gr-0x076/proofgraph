import re
from typing import List, Tuple, Set, Optional
from proofgraph.schemas.evidence import EvidencePassage
from proofgraph.schemas.claim import AtomicClaim

class FIREIterativeDecisionModule:
    """
    Implements a lightweight FIRE-inspired iterative retrieval heuristic:
    - Analyzes current evidence coverage against claim entities and predicates.
    - Emits a binary decision: (should_terminate: bool, next_query: Optional[str])
    - If evidence is insufficient or missing key entities, generates a targeted follow-up query.
    - Implements diversity prompting / negative query constraints to prevent repetitive search loops.
    """
    def __init__(self, confidence_threshold: float = 0.75, max_hops: int = 3):
        self.confidence_threshold = confidence_threshold
        self.max_hops = max_hops

    def _get_covered_entities(self, claim: AtomicClaim, evidence_list: List[EvidencePassage]) -> Set[str]:
        """Finds which claim entities are mentioned in the pooled evidence text."""
        combined_text = " ".join([e.text.lower() for e in evidence_list])
        covered = set()
        for ent in claim.entities:
            if ent.lower() in combined_text:
                covered.add(ent)
        return covered

    def evaluate_step(
        self,
        claim: AtomicClaim,
        retrieved_pool: List[EvidencePassage],
        current_hop: int,
        executed_queries: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether current evidence pool is sufficient to terminate,
        or generates the refined next query for the next iteration.
        """
        if current_hop >= self.max_hops:
            return True, None

        if not retrieved_pool:
            # First hop failed to find anything: query key entities directly
            if claim.entities:
                next_q = " ".join([f'"{ent}"' for ent in claim.entities[:2]])
                return False, next_q
            return True, None

        top_evidence = retrieved_pool[0]
        covered_entities = self._get_covered_entities(claim, retrieved_pool)
        missing_entities = [e for e in claim.entities if e not in covered_entities]

        # FIRE Stopping Condition:
        # High relevance score AND either no missing entities or all key entities covered
        if top_evidence.relevance_score >= self.confidence_threshold and not missing_entities:
            return True, None

        # If missing critical entities (e.g. date, location, or subject), generate targeted follow-up query
        if missing_entities:
            # Query formulation: combine covered subject with missing entity to find the missing link
            key_subject = list(covered_entities)[0] if covered_entities else (claim.entities[0] if claim.entities else "")
            missing_focus = " ".join([f'"{m}"' for m in missing_entities[:2]])
            next_q = f'{key_subject} {missing_focus}'.strip()
            
            # Diversity check: ensure query is not identical to previously executed queries
            if next_q.lower() not in [q.lower() for q in executed_queries]:
                return False, next_q

        # If similarity is moderate (0.50 - 0.70), try an exact-phrase query on the core predicate
        words = claim.text.split()
        if len(words) >= 4:
            core_phrase = " ".join(words[:4])
            next_q = f'"{core_phrase}"'
            if next_q.lower() not in [q.lower() for q in executed_queries]:
                return False, next_q

        # Fallback: terminate if no meaningful refinement can be generated
        return True, None
