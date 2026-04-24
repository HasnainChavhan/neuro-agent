#!/usr/bin/env python3
"""
NeuroAgent — Knowledge Base Seeder
Pre-populates the pgvector store with domain documents for RAG.
Run: python scripts/seed_vector_store.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make app importable from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.pipeline import rag_pipeline
from app.utils.logger import get_logger

logger = get_logger("scripts.seed")

SEED_DOCUMENTS = [
    {
        "text": (
            "Quantum computing leverages quantum mechanical phenomena such as superposition "
            "and entanglement to perform computations. Unlike classical bits, quantum bits "
            "(qubits) can exist in multiple states simultaneously. Google's Sycamore processor "
            "achieved quantum supremacy in 2019 by solving a specific problem in 200 seconds "
            "that would take classical computers 10,000 years. IBM's Eagle processor (2021) "
            "reached 127 qubits, while Osprey (2022) hit 433 qubits. Key applications include "
            "cryptography, drug discovery, optimization, and financial modeling."
        ),
        "source_url": "https://en.wikipedia.org/wiki/Quantum_computing",
        "metadata": {"domain": "quantum", "year": 2024},
    },
    {
        "text": (
            "Large Language Models (LLMs) are neural networks trained on massive text corpora "
            "using the transformer architecture. GPT-4, released by OpenAI in 2023, demonstrates "
            "emergent capabilities including chain-of-thought reasoning, code generation, and "
            "multimodal understanding. Retrieval-Augmented Generation (RAG) enhances LLMs by "
            "grounding responses in retrieved documents, reducing hallucination by 37–42% "
            "according to published benchmarks. Key techniques: vector embeddings, cosine "
            "similarity search, and prompt engineering."
        ),
        "source_url": "https://arxiv.org/abs/2005.11401",
        "metadata": {"domain": "ai", "year": 2024},
    },
    {
        "text": (
            "CRISPR-Cas9 is a revolutionary gene editing technology derived from a natural "
            "bacterial immune defense mechanism. The system uses guide RNA to direct the Cas9 "
            "protein to specific DNA sequences for precise cuts. In 2023, the FDA approved the "
            "first CRISPR-based therapy (Casgevy) for sickle cell disease and beta-thalassemia. "
            "Applications span agriculture (disease-resistant crops), basic research, and "
            "therapeutic development. Key challenges: off-target effects, delivery mechanisms, "
            "and ethical considerations around germline editing."
        ),
        "source_url": "https://www.nature.com/articles/d41586-023-00600-z",
        "metadata": {"domain": "biotech", "year": 2024},
    },
    {
        "text": (
            "Renewable energy has reached cost parity with fossil fuels in most markets. "
            "Solar photovoltaic (PV) costs dropped 90% over the past decade. Wind energy "
            "capacity surpassed 900 GW globally in 2023. Battery storage (lithium-ion) costs "
            "fell below $130/kWh, enabling grid-scale deployment. Green hydrogen, produced "
            "via electrolysis using renewable electricity, is emerging as a key decarbonization "
            "pathway for heavy industry. IEA projects renewables to supply 35% of global "
            "electricity by 2025."
        ),
        "source_url": "https://www.iea.org/reports/renewables-2023",
        "metadata": {"domain": "energy", "year": 2024},
    },
    {
        "text": (
            "The transformer architecture, introduced in 'Attention Is All You Need' (Vaswani "
            "et al., 2017), revolutionized NLP and beyond. It replaces recurrence with "
            "self-attention mechanisms, enabling parallel training and long-range dependency "
            "capture. BERT (2018) introduced bidirectional pre-training; GPT series pioneered "
            "autoregressive generation. Vision Transformers (ViT) demonstrated transformers "
            "outperform CNNs on image tasks at scale. Modern variants include Flash Attention "
            "(10× memory efficiency), Mixture-of-Experts (MoE), and sparse attention."
        ),
        "source_url": "https://arxiv.org/abs/1706.03762",
        "metadata": {"domain": "ai", "year": 2017},
    },
]


async def seed() -> None:
    logger.info("Starting knowledge base seeding", documents=len(SEED_DOCUMENTS))

    try:
        all_ids = await rag_pipeline.ingest_many(SEED_DOCUMENTS)
        logger.info("Seeding complete", total_chunks=len(all_ids))
        print(f"\n✅ Seeded {len(SEED_DOCUMENTS)} documents → {len(all_ids)} chunks stored\n")
    except Exception as exc:
        logger.error("Seeding failed", error=str(exc))
        print(f"\n❌ Seeding failed: {exc}")
        print("   Hint: Is PostgreSQL running? Run: docker-compose up postgres\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(seed())
