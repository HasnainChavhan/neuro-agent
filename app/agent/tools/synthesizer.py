"""
NeuroAgent — Synthesizer Tool
Combines web search results + RAG documents into a coherent answer.
Uses GPT-4o (real) or template-based synthesis (mock mode).
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("agent.tools.synthesizer")
settings = get_settings()

_SYNTHESIS_SYSTEM = """\
You are an expert research synthesizer. Given a research question, web search results,
and retrieved knowledge base documents, produce a concise, accurate, well-structured
answer. Include key facts, cite sources where possible, and flag any contradictions.

Output format (Markdown):
## Answer
[2–4 paragraph synthesis]

## Key Findings
- Finding 1 (Source: ...)
- Finding 2 (Source: ...)

## Confidence
[High / Medium / Low] — reason
"""


class SynthesizerTool:
    """
    Synthesizes search results and RAG docs into a structured answer.
    """

    async def synthesize(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        rag_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Returns a structured synthesis dict."""
        if settings.mock_llm:
            return self._mock_synthesis(query, search_results, rag_docs)
        return await self._llm_synthesis(query, search_results, rag_docs)

    async def _llm_synthesis(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        rag_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        import openai

        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

        context = self._build_context(search_results, rag_docs)

        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Research question: {query}\n\n"
                        f"Context:\n{context}\n\n"
                        "Please synthesize a comprehensive answer."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        answer_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        logger.info(
            "Synthesis complete (LLM)",
            query=query[:60],
            tokens=tokens_used,
        )

        return {
            "answer": answer_text,
            "sources": [r.get("url") or r.get("source_url") for r in search_results + rag_docs if r.get("url") or r.get("source_url")],
            "tokens_used": tokens_used,
            "model": settings.openai_model,
        }

    def _mock_synthesis(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        rag_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        snippets = [r.get("snippet", r.get("content", "")) for r in search_results[:3]]
        rag_snippets = [d.get("content", "")[:200] for d in rag_docs[:2]]
        sources = [
            r.get("url") or r.get("source_url")
            for r in search_results + rag_docs
            if r.get("url") or r.get("source_url")
        ]

        answer = f"""## Answer

Based on comprehensive research synthesis across {len(search_results)} web sources and \
{len(rag_docs)} knowledge base documents, here is what is known about **{query}**:

{snippets[0] if snippets else f'{query} is a complex topic with active research ongoing.'}

{snippets[1] if len(snippets) > 1 else 'Multiple research groups have contributed foundational work in this domain.'}

From the knowledge base: {rag_snippets[0] if rag_snippets else 'No prior cached knowledge found for this query.'}

## Key Findings
- {query} shows significant promise in multiple application domains
- Recent benchmarks demonstrate 20–40% improvement over baselines
- Open challenges remain in scalability and real-world deployment
- Community adoption is growing, with major publications in 2024–2025

## Confidence
**Medium–High** — Synthesized from {len(search_results)} live sources and \
{len(rag_docs)} RAG documents. (Mock mode: not using live LLM)
"""

        return {
            "answer": answer,
            "sources": sources[:8],
            "tokens_used": 0,  # mock
            "model": "mock",
        }

    @staticmethod
    def _build_context(
        search_results: List[Dict[str, Any]],
        rag_docs: List[Dict[str, Any]],
    ) -> str:
        parts = []
        for i, r in enumerate(search_results[:5], 1):
            parts.append(
                f"[Web {i}] {r.get('title','')}\nURL: {r.get('url','')}\n{r.get('snippet','')}"
            )
        for i, d in enumerate(rag_docs[:3], 1):
            parts.append(
                f"[RAG {i}] {d.get('source_url','')}\n{d.get('content','')[:400]}"
            )
        return "\n\n---\n\n".join(parts)


# Module-level singleton
synthesizer_tool = SynthesizerTool()
