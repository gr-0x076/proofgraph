import urllib.parse
from typing import Dict, Optional

class SourceCredibilityScorer:
    """
    Source credibility prior inspired by CONFACT (2025).
    Maintains a domain authority database to prevent low-credibility rumors
    from outvoting authoritative fact-checks in conflict resolution.
    
    Hardened against domain spoofing (e.g., 'reuters.com.evil.org' or 'fake-reuters.com')
    via strict hostname extraction and label-bounded suffix matching.
    """
    DOMAIN_CREDIBILITY: Dict[str, float] = {
        # High Authority / Fact-Checking & Peer-Reviewed Institutions
        "reuters.com": 0.95,
        "apnews.com": 0.95,
        "bloomberg.com": 0.95,
        "bbc.com": 0.92,
        "bbc.co.uk": 0.92,
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
        "nasa.gov": 0.98,
        "nobelprize.org": 0.98,
        "loc.gov": 0.97,
        "nist.gov": 0.99,
        "noaa.gov": 0.98,
        "claymath.org": 0.98,
        
        # Medium / Open Publishing Platforms
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

    def _extract_canonical_hostname(self, domain_or_url: str) -> str:
        """Parses and normalizes hostnames, stripping ports and 'www.' prefixes."""
        if not domain_or_url:
            return ""
        s = domain_or_url.strip().lower()
        if not s.startswith("http://") and not s.startswith("https://"):
            s = "https://" + s
        try:
            parsed = urllib.parse.urlparse(s)
            host = parsed.hostname or ""
            if host.startswith("www."):
                host = host[4:]
            return host
        except Exception:
            return ""

    def get_credibility(self, domain_or_url: str) -> float:
        """
        Returns normalized credibility score between 0.0 and 1.0.
        Uses exact domain matching or label-bounded subdomain matching.
        """
        host = self._extract_canonical_hostname(domain_or_url)
        if not host:
            return 0.50

        # Exact match
        if host in self.DOMAIN_CREDIBILITY:
            return self.DOMAIN_CREDIBILITY[host]

        # Label-bounded subdomain match (e.g., 'news.bbc.co.uk' matches 'bbc.co.uk')
        # Prevents spoofing: 'reuters.com.evil.org' will NOT match 'reuters.com'
        for domain, score in self.DOMAIN_CREDIBILITY.items():
            if host.endswith("." + domain):
                return score

        # Institutional TLD prior
        if host.endswith(".gov") or host.endswith(".edu") or host.endswith(".mil"):
            return 0.92

        # Default unclassified domain credibility
        return 0.50

    # Backwards-compatible alias
    score_url = get_credibility
