import os
import re
import json
import ipaddress
import urllib.request
import urllib.parse
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from proofgraph.schemas.evidence import EvidencePassage

class TinyFishWebAgent:
    """
    TinyFish Search and Fetch integration:
    - TinySearch API (api.search.tinyfish.ai): Real-time structured search results.
    - TinyFetch API (api.fetch.tinyfish.ai): Chromium page rendering and clean markdown extraction.

    Searches for candidate pages, fetches their rendered content, and preserves
    the source URL/title returned by TinyFish. The relevance value is a lexical
    retrieval proxy, not a calibrated probability or semantic relevance model.
    """
    SEARCH_ENDPOINT = "https://api.search.tinyfish.ai"
    FETCH_ENDPOINT = "https://api.fetch.tinyfish.ai"

    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 15):
        self.api_key = api_key if api_key is not None else os.environ.get("TINYFISH_API_KEY")
        self.timeout = timeout_seconds
        self.last_status: str = "initialized"
        self.last_error_code: Optional[int] = None
        self.last_error_message: Optional[str] = None
        self.total_queries_run: int = 0

    @property
    def is_available(self) -> bool:
        """Returns True if a valid TinyFish API key is present."""
        return bool(self.api_key and self.api_key.startswith("sk-tinyfish"))

    def _calculate_lexical_relevance(self, query: str, text: str, rank_pos: int = 0) -> float:
        """Return a transparent lexical-overlap/rank proxy in [0, 1], not confidence."""
        q_tokens = {t for t in re.findall(r'\w+', query.lower()) if len(t) > 2}
        t_tokens = {t for t in re.findall(r'\w+', text.lower()) if len(t) > 2}
        if not q_tokens or not t_tokens:
            return 0.0
        overlap = len(q_tokens.intersection(t_tokens)) / float(len(q_tokens))
        rank_signal = 1.0 / (1.0 + max(0, rank_pos))
        score = 0.8 * overlap + 0.2 * rank_signal
        return round(float(min(1.0, max(0.0, score))), 3)

    @staticmethod
    def _valid_public_http_url(url: str) -> bool:
        """Reject malformed, credential-bearing, localhost, and literal private URLs."""
        try:
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                return False
            host = parsed.hostname.rstrip(".").lower()
            if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
                return False
            try:
                addr = ipaddress.ip_address(host)
            except ValueError:
                return True
            return addr.is_global
        except (TypeError, ValueError):
            return False

    def _post_fetch(self, urls: List[str]) -> Dict[str, Any]:
        """Fetch a bounded URL batch, retaining per-URL API errors in status metadata."""
        urls = [u for u in urls[:10] if self._valid_public_http_url(u)]
        if not urls:
            self.last_status = "invalid_or_empty_urls"
            self.last_error_message = "No valid public HTTP(S) URLs to fetch"
            return {"results": [], "errors": []}
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "ProofGraph-FactChecker/1.0",
        }
        payload = {"urls": urls, "format": "markdown", "ttl": 0}
        try:
            req = urllib.request.Request(
                self.FETCH_ENDPOINT,
                headers=headers,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
            )
            with self._open_with_retry(req) as resp:
                self.last_error_code = resp.status
                data = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
                self.last_status = "invalid_response"
                self.last_error_message = "TinyFetch returned an unexpected response shape"
                return {"results": [], "errors": []}
            errors = data.get("errors", [])
            self.last_status = "partial_fetch_errors" if errors else "success"
            self.last_error_message = f"TinyFetch reported {len(errors)} per-URL error(s)" if errors else None
            return data
        except urllib.error.HTTPError as exc:
            self.last_status = f"http_error_{exc.code}"
            self.last_error_code = exc.code
            self.last_error_message = f"TinyFetch returned HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_status = "network_exception"
            self.last_error_message = str(exc)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self.last_status = "invalid_response"
            self.last_error_message = str(exc)
        except Exception as exc:
            self.last_status = "request_exception"
            self.last_error_message = str(exc)
        return {"results": [], "errors": []}

    def _open_with_retry(self, request):
        """Retry only transient 429/503 responses, with a short bounded backoff."""
        for attempt in range(3):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 503} or attempt == 2:
                    raise
                try:
                    delay = float((exc.headers or {}).get("Retry-After", 0))
                except (AttributeError, TypeError, ValueError):
                    delay = 0.25 * (2 ** attempt)
                time.sleep(min(max(delay, 0.0), 2.0))
        raise RuntimeError("TinyFish request retry loop exhausted")

    @staticmethod
    def _best_excerpt(text: str, query: str, max_chars: int = 1600) -> str:
        """Select a verbatim paragraph with the best query-term coverage."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if len(p.strip()) >= 15]
        if not paragraphs:
            return ""
        terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
        if terms:
            paragraphs.sort(
                key=lambda p: len(terms.intersection(re.findall(r"\w+", p.lower()))) / len(terms),
                reverse=True,
            )
        excerpt = paragraphs[0]
        return excerpt[:max_chars].strip()

    def search_web(self, query: str, max_results: int = 3) -> List[EvidencePassage]:
        """
        Executes a live web search using TinySearch API (api.search.tinyfish.ai).
        Returns structured EvidencePassage objects with exact URLs, titles, and snippets.
        """
        if not self.is_available:
            self.last_status = "unconfigured_key"
            return []

        self.total_queries_run += 1
        encoded_query = urllib.parse.quote(query)
        url = f"{self.SEARCH_ENDPOINT}?query={encoded_query}"
        headers = {
            "X-API-Key": self.api_key,
            "User-Agent": "ProofGraph-FactChecker/1.0"
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with self._open_with_retry(req) as resp:
                self.last_error_code = resp.status
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    self.last_status = "success"
                    self.last_error_message = None
                    if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
                        self.last_status = "invalid_response"
                        self.last_error_message = "TinySearch returned an unexpected response shape"
                        return []
                    return self._parse_search_results(data, query, max_results)
                else:
                    self.last_status = f"http_error_{resp.status}"
                    self.last_error_message = f"TinySearch returned HTTP {resp.status}"
                    return []
        except urllib.error.HTTPError as e:
            self.last_status = f"http_error_{e.code}"
            self.last_error_code = e.code
            self.last_error_message = str(e)
            return []
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.last_status = "network_exception"
            self.last_error_message = str(e)
            return []
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as e:
            self.last_status = "invalid_response"
            self.last_error_message = str(e)
            return []
        except Exception as e:
            self.last_status = "request_exception"
            self.last_error_message = str(e)
            return []

    def _parse_search_results(self, data: Dict[str, Any], query: str, max_results: int) -> List[EvidencePassage]:
        """Parses TinySearch JSON results into verified EvidencePassage objects."""
        passages: List[EvidencePassage] = []
        raw_results = data.get("results", [])

        for idx, item in enumerate(raw_results[:max_results]):
            snippet = item.get("snippet", "")
            title = item.get("title", "")
            src_url = item.get("url", "")

            if not snippet or len(snippet.strip()) < 15:
                continue

            # Validate real HTTP/HTTPS URL
            if not self._valid_public_http_url(src_url):
                continue

            # Official TinySearch API provides results[].domain
            domain = item.get("domain") or item.get("site_name") or urllib.parse.urlparse(src_url).netloc
            relevance = self._calculate_lexical_relevance(query, f"{title} {snippet}", rank_pos=idx)

            passages.append(
                EvidencePassage(
                    id=f"tinyfish_search_{idx+1}",
                    text=f"{title}: {snippet}".strip(),
                    source_url=src_url,
                    source_domain=domain,
                    source_title=title or f"TinySearch: {domain}",
                    relevance_score=relevance,
                    metadata={
                        "retrieval_method": "tinyfish_search",
                        "relevance_method": "lexical_overlap_and_search_rank_proxy",
                        "position": item.get("position", idx + 1),
                        "domain": domain,
                        "site_name": item.get("site_name", domain),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            )

        return passages

    def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetches and renders full clean text from a URL using TinyFetch API."""
        if not self.is_available:
            self.last_status = "unconfigured_key"
            return None
        data = self._post_fetch([url])
        results = data.get("results", [])
        return results[0].get("text") if results and isinstance(results[0], dict) else None

    def audit_citation_url(self, url: str, claim_text: str, max_evidence: int = 5) -> List[EvidencePassage]:
        """Fetches and extracts evidence passages from a cited source URL using TinyFetch or HTTP fallback."""
        return self.research_claim_on_web(claim_text, target_url=url, max_evidence=max_evidence)

    def _fetch_via_http_fallback(self, url: str, query: str, max_evidence: int = 5) -> List[EvidencePassage]:
        """Direct HTTP fetch fallback when TinyFish API key is unavailable or in offline/test environment."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ProofGraph-CitationAuditor/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw_html = resp.read().decode("utf-8", errors="replace")

            # Strip scripts, styles, and HTML tags
            clean_text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', raw_html, flags=re.IGNORECASE)
            clean_text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', clean_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            paragraphs = [p.strip() for p in re.split(r'(?<=[.!?])\s+', clean_text) if len(p.strip()) >= 30]
            if not paragraphs:
                return []

            q_terms = {t for t in re.findall(r'\w+', query.lower()) if len(t) > 2}
            scored = []
            for p in paragraphs:
                p_terms = {t for t in re.findall(r'\w+', p.lower()) if len(t) > 2}
                score = len(q_terms.intersection(p_terms)) / float(len(q_terms)) if q_terms else 0.0
                scored.append((score, p))

            scored.sort(key=lambda x: x[0], reverse=True)
            domain = urllib.parse.urlsplit(url).hostname or "unknown"

            passages = []
            for idx, (score, p_text) in enumerate(scored[:max_evidence]):
                passages.append(EvidencePassage(
                    id=f"http_fetch_{idx+1}",
                    text=p_text[:800].strip(),
                    source_url=url,
                    source_domain=domain,
                    source_title=f"Source: {domain}",
                    relevance_score=round(float(score), 3),
                    metadata={
                        "retrieval_method": "http_direct_fetch",
                        "audit_url": url,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                ))
            return passages
        except Exception:
            return []

    def research_claim_on_web(
        self,
        claim_text: str,
        target_url: Optional[str] = None,
        max_evidence: int = 3
    ) -> List[EvidencePassage]:
        """
        Primary entry point for claim research:
        1. If target_url is provided, fetches the page using TinyFetch (with HTTP fallback).
        2. Otherwise, executes a targeted web search using TinySearch to discover
           authoritative pages and extracts evidence passages with verified URLs.
        """
        if target_url:
            if self.is_available:
                data = self._post_fetch([target_url])
                fetched = self._passages_from_fetch(data, claim_text)
                if fetched:
                    return fetched[:max_evidence]
            return self._fetch_via_http_fallback(target_url, claim_text, max_evidence)

        if not self.is_available:
            self.last_status = "unconfigured_key"
            return []

        # Search, then fetch the discovered pages so the verifier receives page text,
        # not only search snippets. Keep snippets as an explicit fallback on fetch failure.
        search_hits = self.search_web(claim_text, max_results=min(max_evidence, 10))
        if not search_hits:
            return []
        fetched_data = self._post_fetch([p.source_url for p in search_hits if p.source_url])
        fetched = self._passages_from_fetch(fetched_data, claim_text)
        if fetched:
            return fetched[:max_evidence]
        if self.last_status in {"success", "partial_fetch_errors"}:
            self.last_status = "fetch_no_extractable_content"
            self.last_error_message = "TinyFetch returned no usable page text; returning TinySearch snippets"
        return search_hits[:max_evidence]

    def _passages_from_fetch(self, data: Dict[str, Any], query: str) -> List[EvidencePassage]:
        passages: List[EvidencePassage] = []
        for idx, item in enumerate(data.get("results", [])[:10]):
            if not isinstance(item, dict):
                continue
            text = self._best_excerpt(item.get("text", ""), query)
            source_url = item.get("final_url") or item.get("url") or ""
            if len(text) < 15 or not self._valid_public_http_url(source_url):
                continue
            domain = urllib.parse.urlsplit(source_url).hostname or "unknown"
            title = item.get("title") or domain
            passages.append(EvidencePassage(
                id=f"tinyfish_fetch_{idx + 1}",
                text=text,
                source_url=source_url,
                source_domain=domain,
                source_title=title,
                relevance_score=self._calculate_lexical_relevance(query, text, idx),
                metadata={
                    "retrieval_method": "tinyfish_fetch",
                    "requested_url": item.get("url"),
                    "final_url": item.get("final_url"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "relevance_method": "lexical_overlap_and_search_rank_proxy",
                },
            ))
        return passages
