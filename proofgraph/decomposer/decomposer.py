import re
from typing import List, Optional
from proofgraph.schemas.claim import AtomicClaim

class ClaimDecomposer:
    """
    Implements a lightweight Claimify-inspired heuristic proposition splitter:
    1. Sentence segmentation with abbreviation protection
    2. Verifiability filtering
    3. Referential ambiguity detection
    4. Decontextualization and atomic proposition extraction
    """

    # Common subjective/opinion prefixes that mark non-verifiable claims
    OPINION_MARKERS = [
        r"^i (think|feel|believe|suppose|guess|hope)",
        r"^(in my opinion|in my view|arguably|personally)",
        r"^(it seems|it feels like|it is widely believed that)",
        r"(best|worst|greatest|most beautiful|nicest) ever"
    ]

    # Ambiguity triggers (unresolved pronouns at start of clauses)
    AMBIGUITY_PRONOUNS = ["he", "she", "they", "it", "this", "these", "those", "that"]

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def split_sentences(self, text: str) -> List[str]:
        """Splits paragraphs into clean sentences while preserving abbreviations and initials."""
        # Normalize whitespace
        text = text.strip().replace("\r\n", "\n")
        # Protect common abbreviations: e.g., i.e., U.S., Dr., Prof., Jan., etc.
        protected = re.sub(r'\b(U\.S\.|e\.g\.|i\.e\.|Dr\.|Mr\.|Mrs\.|Prof\.|vs\.)', lambda m: m.group(0).replace('.', '§'), text)
        # Protect single letter initials like J. K. Rowling
        protected = re.sub(r'\b([A-Z])\.\s*', r'\1§ ', protected)
        
        # Split on sentence terminals
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])', protected)
        
        # Restore periods
        sentences = [s.replace('§', '.').strip() for s in raw_sentences if s.strip()]
        return sentences

    def is_verifiable(self, sentence: str) -> bool:
        """Determines if a statement makes an objective assertion verifiable against evidence."""
        s_lower = sentence.lower()
        for pattern in self.OPINION_MARKERS:
            if re.search(pattern, s_lower):
                return False
        # Very short fragments (< 3 words) are usually non-verifiable
        words = [w for w in sentence.split() if re.search(r'\w', w)]
        return len(words) >= 3

    def detect_ambiguity(self, sentence: str, previous_context: Optional[str] = None) -> bool:
        """Detects referential ambiguity when key subjects are vague pronouns without antecedents."""
        first_word = sentence.strip().split()[0].lower() if sentence.strip() else ""
        if first_word in self.AMBIGUITY_PRONOUNS and not previous_context:
            return True
        return False

    def extract_entities(self, text: str) -> List[str]:
        """Extracts candidate entities: capitalized multi-word sequences, quoted terms, numbers, dates."""
        entities = []
        # Multi-word proper nouns: e.g., "World Health Organization", "Albert Einstein"
        prop_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        entities.extend(prop_nouns)
        
        # Quoted terms
        quotes = re.findall(r'["\']([^"\']+)["\']', text)
        entities.extend(quotes)
        
        # Years / Dates: e.g., "2024", "19th century"
        years = re.findall(r'\b(?:19|20)\d{2}\b', text)
        entities.extend(years)
        
        # Single capitalized words (excluding start of sentence if not already included)
        words = text.split()
        for w in words[1:]:
            clean = re.sub(r'[^\w]', '', w)
            if clean and clean[0].isupper() and clean not in entities:
                entities.append(clean)
                
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                deduped.append(e)
        return deduped

    def decompose(self, document_or_claim: str) -> List[AtomicClaim]:
        """
        Decomposes input text into atomic, independently verifiable claims.
        Guarantees: N_subclaims <= N_sentences unless compound conjunctions explicitly require splitting.
        """
        sentences = self.split_sentences(document_or_claim)
        atomic_claims: List[AtomicClaim] = []
        
        for idx, sentence in enumerate(sentences):
            verifiable = self.is_verifiable(sentence)
            entities = self.extract_entities(sentence)
            prev_context = sentences[idx - 1] if idx > 0 else None
            ambiguous = self.detect_ambiguity(sentence, prev_context)

            # Check for compound sentences joined by explicit contrasting or additive conjunctions:
            # e.g., "Company X reported record profits in 2023, but laid off 15% of its workforce in 2024."
            compound_split = re.split(r';|\b,\s*(?:but|however|whereas|while|although)\s+', sentence)
            
            if len(compound_split) > 1 and all(len(c.split()) >= 4 for c in compound_split):
                for part in compound_split:
                    part_clean = part.strip()
                    if not part_clean.endswith('.'):
                        part_clean += '.'
                    # Capitalize first letter
                    part_clean = part_clean[0].upper() + part_clean[1:]
                    part_entities = self.extract_entities(part_clean)
                    
                    atomic_claims.append(
                        AtomicClaim(
                            text=part_clean,
                            parent_sentence=sentence,
                            entities=part_entities if part_entities else entities,
                            is_verifiable=verifiable,
                            ambiguity_flag=ambiguous,
                            confidence=0.9 if not ambiguous else 0.6
                        )
                    )
            else:
                atomic_claims.append(
                    AtomicClaim(
                        text=sentence,
                        parent_sentence=sentence,
                        entities=entities,
                        is_verifiable=verifiable,
                        ambiguity_flag=ambiguous,
                        confidence=1.0 if not ambiguous else 0.7
                    )
                )

        return atomic_claims
