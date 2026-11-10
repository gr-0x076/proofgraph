import os
import json
import urllib.request
import urllib.parse
from typing import List, Optional, Dict, Any

from proofgraph.schemas.evidence import EvidencePassage

class TinyFishWebAgent:
    """
    First-Class TinyFish Web Agent Integration:
    Connects to TinyFish Cloud Platform (agent.tinyfish.ai) to execute goal-based
    autonomous web research, page navigation, and evidence extraction for claim verification.
    """
    ENDPOINT = "https://agent.tinyfish.ai/v1/automation/run"

    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 25):
        self.api_key = api_key if api_key is not None else os.environ.get("TINYFISH_API_KEY")
        self.timeout = timeout_seconds


    @property
    def is_available(self) -> bool:
        """Returns True if a TinyFish API key is present."""
        return bool(self.api_key and self.api_key.startswith("sk-tinyfish"))

    def research_claim_on_web(
        self,
        claim_text: str,
        target_url: Optional[str] = None,
        max_evidence: int = 3
    ) -> List[EvidencePassage]:
        """
        Dispatches an autonomous web research task to the TinyFish Agent.
        If target_url is provided, navigates directly to that site.
        Otherwise, targets an authoritative encyclopedia or fact-checking source.
        """
        if not self.is_available:
            return []

        # Target URL selection
        url = target_url or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(claim_text[:50])}"
        goal = (
            f"Fact-check this assertion: '{claim_text}'. "
            f"Extract up to {max_evidence} direct factual excerpts, quotes, or refutations from the page "
            f"along with the exact source title and URL."
        )

        payload = {
            "url": url,
            "goal": goal,
            "agent_config": {
                "max_steps": 10
            }
        }

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "ProofGraph-FactChecker/1.0"
        }

        try:
            req = urllib.request.Request(
                self.ENDPOINT,
                headers=headers,
                data=json.dumps(payload).encode("utf-8")
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return self._parse_agent_response(data, claim_text, url)
        except Exception as e:
            # Graceful fallback: return empty list on timeout or network block
            return []

        return []

    def _parse_agent_response(
        self,
        response_data: Dict[str, Any],
        claim_text: str,
        fallback_url: str
    ) -> List[EvidencePassage]:
        """Parses the structured JSON output returned by TinyFish Agent into EvidencePassage models."""
        passages: List[EvidencePassage] = []
        result = response_data.get("result") or response_data.get("output") or {}

        # Handle list of items or single text output
        items = result.get("evidence", []) if isinstance(result, dict) else []
        if isinstance(result, str) and result.strip():
            items = [{"text": result.strip(), "source_title": "TinyFish Web Agent Finding"}]

        for idx, item in enumerate(items[:5]):
            text = item.get("text") if isinstance(item, dict) else str(item)
            if not text or len(text.strip()) < 15:
                continue

            src_url = item.get("url", fallback_url) if isinstance(item, dict) else fallback_url
            src_title = item.get("source_title", "TinyFish Autonomous Research") if isinstance(item, dict) else "TinyFish Autonomous Research"
            
            domain = ""
            try:
                domain = urllib.parse.urlparse(src_url).netloc
            except Exception:
                pass

            passages.append(
                EvidencePassage(
                    id=f"tinyfish_{idx+1}",
                    text=text.strip(),
                    source_url=src_url,
                    source_domain=domain,
                    source_title=src_title,
                    relevance_score=0.88
                )
            )

        return passages
