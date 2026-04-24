"""
NeuroAgent — Query Planner
Decomposes a complex research query into ordered sub-queries using GPT-4o.
Falls back to a simple split heuristic in mock mode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("agent.planner")
settings = get_settings()

_SYSTEM_PROMPT = """\
You are a research planning assistant. Given a complex research query, decompose it
into 2–5 focused, self-contained sub-queries that together will fully answer the
original question. Return a JSON object with:
  {
    "sub_queries": ["...", "..."],
    "reasoning": "brief explanation of decomposition strategy"
  }
Output ONLY the JSON object, no markdown fences.
"""


@dataclass
class ResearchPlan:
    original_query: str
    sub_queries: List[str]
    reasoning: str = ""
    estimated_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.estimated_steps = len(self.sub_queries) * 2  # search + synthesize per sub


class QueryPlanner:
    """
    Decomposes a complex query into sub-queries.

    Real mode: GPT-4o JSON mode
    Mock mode: rule-based heuristic splitter
    """

    async def plan(self, query: str) -> ResearchPlan:
        if settings.mock_llm:
            return self._mock_plan(query)
        return await self._llm_plan(query)

    async def _llm_plan(self, query: str) -> ResearchPlan:
        import openai

        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Research query: {query}"},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)

        sub_queries = data.get("sub_queries", [query])
        reasoning = data.get("reasoning", "")

        logger.info(
            "Query planned (LLM)",
            original=query[:80],
            sub_count=len(sub_queries),
        )
        return ResearchPlan(
            original_query=query,
            sub_queries=sub_queries,
            reasoning=reasoning,
        )

    @staticmethod
    def _mock_plan(query: str) -> ResearchPlan:
        """
        Heuristic decomposition for mock mode:
        - Splits on 'and', 'vs', '?', ','
        - Generates contextual sub-queries from the main query
        """
        import re

        # Remove trailing punctuation
        clean = query.strip().rstrip("?.")

        # Strategy: generate 3 archetypal research angles
        sub_queries = [
            f"Overview and definition: {clean}",
            f"Recent developments and breakthroughs in {clean}",
            f"Key challenges and future directions for {clean}",
        ]

        # If query contains connectors, split on them too
        splits = re.split(r"\band\b|\bvs\b|\bversus\b|,", clean, flags=re.IGNORECASE)
        if len(splits) > 1:
            sub_queries = [s.strip() for s in splits if s.strip()]

        logger.info(
            "Query planned (mock)",
            original=query[:80],
            sub_count=len(sub_queries),
        )
        return ResearchPlan(
            original_query=query,
            sub_queries=sub_queries,
            reasoning="Mock heuristic decomposition (LLM disabled)",
        )


# Module-level singleton
query_planner = QueryPlanner()
