import re
from typing import List, Optional, Tuple
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

    COMMON_GIVEN_NAMES = {
        "peter", "william", "john", "david", "mary", "michael", "james", "robert",
        "sarah", "emily", "joseph", "thomas", "charles", "daniel", "matthew",
        "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
        "kenneth", "kevin", "brian", "george", "edward", "ronald", "timothy",
        "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
        "stephen", "jonathan", "larry", "justin", "scott", "brandon", "frank",
        "benjamin", "gregory", "samuel", "raymond", "patrick", "alexander", "jack",
        "dennis", "jerry", "tyler", "aaron", "henry", "douglas", "adam",
        "zachary", "nathan", "walter", "harold", "kyle", "carl", "arthur",
        "gerald", "roger", "keith", "jeremy", "terry", "lawrence", "sean",
        "christian", "albert", "joe", "ethan", "austin", "jesse", "willie",
        "billy", "bryan", "bruce", "jordan", "ralph", "roy", "noah", "dylan",
        "eugene", "wayne", "louis", "harry", "elizabeth", "jennifer", "maria",
        "susan", "margaret", "dorothy", "lisa", "nancy", "karen", "betty",
        "helen", "sandra", "donna", "carol", "ruth", "sharon", "michelle",
        "laura", "kimberly", "deborah", "jessica", "shirley", "cynthia", "angela",
        "melissa", "brenda", "amy", "anna", "rebecca", "virginia", "kathleen",
        "pamela", "martha", "debra", "amanda", "stephanie", "carolyn", "christine",
        "marie", "janet", "catherine", "frances", "ann", "joyce", "diane", "alice",
        "julie", "heather", "teresa", "doris", "gloria", "evelyn", "jean", "cheryl",
        "mildred", "katherine", "joan", "ashley", "judith", "rose", "janice",
        "kelly", "nicole", "judy", "christina", "kathy", "theresa", "beverly",
        "denise", "tammy", "irene", "jane", "lori", "rachel", "marilyn", "andrea",
        "kathryn", "louise", "sara", "anne", "jacqueline", "wanda", "bonnie",
        "julia", "ruby", "lois", "tina", "phyllis", "norma", "paula", "diana",
        "annie", "lillian", "robin", "peggy", "crystal", "gladys", "rita", "dawn",
        "connie", "florence", "tracy", "edna", "tiffany", "carmen", "rosa", "cindy",
        "grace", "wendy", "victoria", "edith", "kim", "sherry", "sylvia",
        "josephine", "thelma", "shannon", "sheila", "ethel", "ellen", "elaine",
        "marjorie", "carrie", "charlotte", "monica", "esther", "pauline", "emma",
        "juanita", "anita", "lucas", "oliver", "liam", "elijah", "ben"
    }

    TITLES_AND_HONORIFICS = {
        "dr", "mr", "mrs", "ms", "prof", "professor", "sir", "lord", "lady",
        "saint", "st", "king", "queen", "prince", "princess", "pope", "uncle",
        "aunt", "president", "senator", "governor", "general", "admiral", "prime minister"
    }

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

    def check_underspecified_mononym(
        self,
        sentence: str,
        entities: List[str],
        previous_context: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Detects referential ambiguity when key subjects are vague pronouns without antecedents,
        or isolated common given names (mononyms) lacking surname, title, or disambiguating context.
        """
        words = sentence.strip().split()
        if not words:
            return False, None

        first_word = words[0].lower().rstrip(".,;:!?")
        if first_word in self.AMBIGUITY_PRONOUNS and not previous_context:
            return True, f"Claim begins with unresolved pronoun '{first_word}' without prior context."

        multi_word_entities = [e.lower() for e in entities if len(e.split()) > 1]

        for ent in entities:
            ent_clean = ent.strip().strip(".,;:!?")
            ent_lower = ent_clean.lower()

            if len(ent_clean.split()) == 1 and ent_lower in self.COMMON_GIVEN_NAMES:
                # Check if this name is part of any multi-word entity in the claim
                if any(ent_lower in m.split() for m in multi_word_entities):
                    continue

                # Check if preceded by a title or honorific in the sentence
                pat_title = rf"\b({'|'.join(self.TITLES_AND_HONORIFICS)})\.?\s+{re.escape(ent_clean)}\b"
                if re.search(pat_title, sentence, re.IGNORECASE):
                    continue

                # Check if immediately followed by another capitalized word (surname)
                pat_surname = rf"\b{re.escape(ent_clean)}\s+([A-Z][a-z]+)\b"
                m_surname = re.search(pat_surname, sentence)
                if m_surname:
                    next_word = m_surname.group(1).lower()
                    if next_word not in self.COMMON_GRAMMAR and next_word not in self.COMMON_STARTERS:
                        continue

                return True, (
                    f"Subject '{ent_clean}' is an underspecified common given name lacking surname, "
                    f"title, or disambiguating context. Verification cannot proceed without identifying which {ent_clean} is referenced."
                )

        return False, None

    COMMON_STARTERS = {
        "the", "a", "an", "in", "on", "at", "if", "why", "when", "what", "where", "how", "who",
        "there", "it", "its", "this", "that", "these", "those", "more", "all", "any", "some",
        "video", "viral", "according", "while", "after", "before", "one", "two", "three", "several", "most"
    }

    COMMON_GRAMMAR = {
        "with", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are",
        "it", "its", "by", "as", "from", "be", "been", "has", "had", "have", "do", "did", "does",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must", "get", "got",
        "out", "about", "over", "after", "before", "all", "any", "some", "no", "not", "their", "his",
        "her", "my", "our", "your", "says", "said", "shows", "shown", "intended", "plastered", "thrown",
        "arriving", "claimed", "claiming", "inside", "outside", "between", "among"
    }

    def extract_entities(self, text: str) -> List[str]:
        """Extracts candidate entities: capitalized multi-word sequences, quoted terms, numbers, dates."""
        entities = []
        # Quoted terms
        quotes = re.findall(r'["\']([^"\']+)["\']', text)
        entities.extend(quotes)

        # Years / Dates: e.g., "2024", "19th century"
        years = re.findall(r'\b(?:19|20)\d{2}\b', text)
        entities.extend(years)

        # Words analysis for headline detection
        words = text.split()
        cap_ratio = sum(1 for w in words if w and w[0].isupper()) / max(1, len(words))
        is_headline = cap_ratio >= 0.50

        # Multi-word proper nouns: e.g., "World Health Organization", "Albert Einstein"
        prop_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        for pn in prop_nouns:
            pn_words = pn.split()
            if is_headline or any(w.lower() in self.COMMON_GRAMMAR for w in pn_words):
                chunk = []
                for w in pn_words:
                    if w.lower() in self.COMMON_GRAMMAR:
                        if chunk:
                            entities.append(" ".join(chunk))
                            chunk = []
                    else:
                        chunk.append(w)
                if chunk:
                    entities.append(" ".join(chunk))
            else:
                entities.append(pn)

        # Single capitalized words (including first word if not a generic sentence starter)
        for idx, w in enumerate(words):
            clean = re.sub(r'[^\w]', '', w)
            if not clean or not clean[0].isupper():
                continue
            if idx == 0 and clean.lower() in self.COMMON_STARTERS:
                continue
            if clean.lower() in self.COMMON_GRAMMAR or clean.lower() in self.COMMON_STARTERS:
                continue
            entities.append(clean)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for e in entities:
            clean_e = e.strip()
            if clean_e.lower() not in seen and len(clean_e) >= 2:
                seen.add(clean_e.lower())
                deduped.append(clean_e)
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

            is_under, under_reason = self.check_underspecified_mononym(sentence, entities, prev_context)

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
                    part_under, part_under_reason = self.check_underspecified_mononym(
                        part_clean, part_entities if part_entities else entities, prev_context
                    )
                    
                    atomic_claims.append(
                        AtomicClaim(
                            text=part_clean,
                            parent_sentence=sentence,
                            entities=part_entities if part_entities else entities,
                            is_verifiable=verifiable,
                            ambiguity_flag=ambiguous or part_under,
                            is_underspecified=part_under,
                            ambiguity_reason=part_under_reason,
                            confidence=0.9 if not (ambiguous or part_under) else 0.5
                        )
                    )
            else:
                atomic_claims.append(
                    AtomicClaim(
                        text=sentence,
                        parent_sentence=sentence,
                        entities=entities,
                        is_verifiable=verifiable,
                        ambiguity_flag=ambiguous or is_under,
                        is_underspecified=is_under,
                        ambiguity_reason=under_reason,
                        confidence=1.0 if not (ambiguous or is_under) else 0.5
                    )
                )

        return atomic_claims
