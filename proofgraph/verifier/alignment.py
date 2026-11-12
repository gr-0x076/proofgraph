import re
from typing import Set, Tuple
from proofgraph.schemas.claim import AtomicClaim
from proofgraph.schemas.evidence import EvidencePassage

class RefNLIAlignmentGate:
    """
    Applies a lexical context-mismatch filter inspired by RefNLI research:
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

    NATIONALITIES = {
        "american", "french", "german", "british", "russian", "chinese",
        "pakistani", "indian", "japanese", "korean", "italian", "spanish",
        "canadian", "australian"
    }

    TITLES = {"dr", "mr", "mrs", "ms", "prof", "sir", "lord"}
    SUBJECT_STOPWORDS = TITLES | {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "by", "as", "is", "was", "are", "were", "it", "its"
    }

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

        # Check entity and primary subject overlap (RefNLI Reference Indeterminacy Defense)
        entity_match = False
        if claim.entities:
            non_numeric = [e for e in claim.entities if not re.match(r'^\d+$', e)]
            primary_subject = non_numeric[0] if non_numeric else claim.entities[0]
            is_attr_or_nationality = primary_subject.lower() in self.NATIONALITIES
            
            # Primary subject verification: if claim has a named subject entity,
            # verify that core subject stems (excluding titles and short words) are present in evidence
            subject_parts = [
                p.lower().strip(".,") for p in primary_subject.split() 
                if p.lower().strip(".,") not in self.SUBJECT_STOPWORDS and len(p.strip(".,")) >= 3
            ]

            has_primary = bool(subject_parts) and any(
                p in evidence.text.lower() for p in subject_parts
            )

            # RefNLI Reference Indeterminacy Defense: If claim specifies only a single given name,
            # but evidence refers to that name followed by an explicit surname not present in the claim,
            # reject alignment as referencing an indeterminate or different individual.
            if len(subject_parts) == 1 and not is_attr_or_nationality and has_primary:
                name_stem = subject_parts[0]
                surname_match = re.search(rf"\b{re.escape(name_stem)}\s+([a-z]+)\b", evidence.text.lower())
                if surname_match:
                    found_surname = surname_match.group(1)
                    if (
                        found_surname not in self.SUBJECT_STOPWORDS
                        and found_surname not in self.NATIONALITIES
                        and found_surname not in claim_tokens
                        and len(found_surname) >= 3
                    ):
                        return False, jaccard, (
                            f"Context Mismatch (RefNLI Reference Indeterminacy): "
                            f"Evidence refers to specific entity '{name_stem.capitalize()} {found_surname.capitalize()}', "
                            f"which was not identified in the claim."
                        )

            other_entity_match = any(
                e.lower() in evidence.text.lower()
                for e in non_numeric[1:]
                if len(e.strip()) >= 3 and e.lower().strip() not in self.SUBJECT_STOPWORDS
            )

            is_attr_or_nationality = primary_subject.lower() in self.NATIONALITIES

            if subject_parts and not has_primary:
                # If primary subject is an adjective/nationality (e.g. French in 'French embassy')
                # and another substantive entity matches (e.g. Sudan): allow alignment
                if is_attr_or_nationality and other_entity_match:
                    entity_match = True
                else:
                    return False, jaccard, f"Context Mismatch: Primary subject '{primary_subject}' absent in evidence (RefNLI Reference Indeterminacy)"
            else:
                entity_match = has_primary or other_entity_match
        else:
            entity_match = False

        claim_coverage = len(intersection) / len(claim_tokens)
        lexical_match = jaccard >= self.jaccard_threshold or (not claim.entities and claim_coverage >= 0.20)

        if entity_match or lexical_match:
            return True, jaccard, "Aligned: primary subject or lexical overlap threshold satisfied"
        
        return False, jaccard, f"Context Mismatch: Jaccard ({jaccard:.3f}) below threshold ({self.jaccard_threshold}) and no primary-subject match"
