"""
NeuroAgent — Web Search Tool
Supports DuckDuckGo (default/free), Brave, and SerpAPI backends.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("agent.tools.web_search")
settings = get_settings()


class WebSearchResult:
    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}

    def __repr__(self) -> str:
        return f"WebSearchResult(title={self.title!r}, url={self.url!r})"


class WebSearchTool:
    """
    Unified web search wrapper.
    Selects backend based on SEARCH_PROVIDER env var.
    Mock mode returns synthetic results for key-free testing.
    """

    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        if settings.mock_llm:
            return self._mock_results(query, max_results)

        provider = settings.search_provider.lower()
        if provider == "duckduckgo":
            return await self._duckduckgo(query, max_results)
        elif provider == "brave":
            return await self._brave(query, max_results)
        elif provider == "serpapi":
            return await self._serpapi(query, max_results)
        else:
            raise ValueError(f"Unknown search provider: {provider}")

    # ── DuckDuckGo (free, no key) ───────────────────────────────────────────

    async def _duckduckgo(self, query: str, max_results: int) -> List[WebSearchResult]:
        from duckduckgo_search import DDGS

        results: List[WebSearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    )
                )
        logger.info("DuckDuckGo search", query=query[:60], results=len(results))
        return results

    # ── Brave Search API ────────────────────────────────────────────────────

    async def _brave(self, query: str, max_results: int) -> List[WebSearchResult]:
        import httpx

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": settings.brave_api_key,
        }
        params = {"q": query, "count": max_results}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for r in data.get("web", {}).get("results", []):
            results.append(
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("description", ""),
                )
            )
        logger.info("Brave search", query=query[:60], results=len(results))
        return results

    # ── SerpAPI ─────────────────────────────────────────────────────────────

    async def _serpapi(self, query: str, max_results: int) -> List[WebSearchResult]:
        import httpx

        params = {
            "q": query,
            "api_key": settings.serpapi_key,
            "num": max_results,
            "engine": "google",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://serpapi.com/search", params=params, timeout=15.0
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for r in data.get("organic_results", []):
            results.append(
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                )
            )
        logger.info("SerpAPI search", query=query[:60], results=len(results))
        return results

    # ── Mock (deterministic test data) ──────────────────────────────────────

    @staticmethod
    def _mock_results(query: str, max_results: int) -> List[WebSearchResult]:
        templates = [
            (
                f"Introduction to {query}",
                f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                f"{query} is a rapidly evolving field with significant recent progress. "
                f"Researchers have demonstrated breakthrough results in multiple benchmarks.",
            ),
            (
                f"Recent advances in {query} — Nature (2024)",
                f"https://www.nature.com/articles/{query.replace(' ', '-')}-2024",
                f"A new study published in Nature reveals that {query} has achieved "
                f"state-of-the-art performance, surpassing previous methods by 23%.",
            ),
            (
                f"{query}: Challenges and Future Directions",
                f"https://arxiv.org/abs/2401.{hash(query) % 99999:05d}",
                f"Despite remarkable progress, {query} still faces fundamental challenges "
                f"including scalability, interpretability, and real-world deployment.",
            ),
            (
                f"Practical applications of {query}",
                f"https://techcrunch.com/{query.replace(' ', '-')}-applications",
                f"Industry leaders are rapidly adopting {query} for production use cases, "
                f"with reported efficiency gains of 40–60% over traditional methods.",
            ),
            (
                f"{query} — Benchmark Comparison 2024",
                f"https://paperswithcode.com/{query.replace(' ', '-')}",
                f"The latest benchmarks show {query} achieving top scores on MMLU, "
                f"HumanEval, and domain-specific evaluation sets.",
            ),
        ]
        return [
            WebSearchResult(title=t, url=u, snippet=s)
            for t, u, s in templates[:max_results]
        ]


# Module-level singleton
web_search_tool = WebSearchTool()
