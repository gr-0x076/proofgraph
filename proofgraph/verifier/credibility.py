from typing import Dict

class SourceCredibilityScorer:
    """
    Implements source credibility augmentation based on CONFACT (2025).
    Maintains a domain authority database to prevent low-credibility rumors
    from outvoting authoritative fact-checks in conflict resolution.
    """
    DOMAIN_CREDIBILITY: Dict[str, float] = {
        # High Authority / Fact-Checking & Peer-Reviewed
        "reuters.com": 0.95,
        "apnews.com": 0.95,
        "bloomberg.com": 0.95,
        "bbc.com": 0.92,
        "nature.com": 0.98,
        "science.org": 0.98,
        "who.int": 0.95,
        "cdc.gov": 0.95,
        "wikipedia.org": 0.88,
        "factcheck.org": 0.95,
        "snopes.com": 0.90,
        "politifact.com": 0.90,
        "techcrunch.com": 0.88,
        "britannica.com": 0.92,
        
        # Medium Authority
        "medium.com": 0.50,
        "substack.com": 0.55,
        "reddit.com": 0.35,
        "twitter.com": 0.35,
        "x.com": 0.35,
        
        # Low Authority / Content Farms / Satire
        "theonion.com": 0.10,
        "babylonbee.com": 0.10,
        "infowars.com": 0.15,
        "naturalnews.com": 0.15,
    }

    def get_credibility(self, domain_or_url: str) -> float:
        """Returns normalized credibility score between 0.0 and 1.0."""
        if not domain_or_url:
            return 0.5
        d_lower = domain_or_url.lower()
        for domain, score in self.DOMAIN_CREDIBILITY.items():
            if domain in d_lower:
                return score
        # Educational / Government TLD boost
        if d_lower.endswith(".gov") or d_lower.endswith(".edu"):
            return 0.92
        return 0.50  # Default neutral credibility
