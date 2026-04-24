# ADR 003 — Agent Framework: LangChain

**Status:** Accepted  
**Date:** 2025-01-01  
**Deciders:** NeuroAgent Engineering

---

## Context

NeuroAgent needs a framework for:
1. Tool-calling orchestration (web search, RAG, synthesizer)
2. Multi-step ReAct-style reasoning loops
3. Structured output parsing (Pydantic integration)
4. Retry/fallback logic composability

## Options Considered

| Framework | Tool Calling | Structured Output | Async | Maturity | Notes |
|---|---|---|---|---|---|
| **LangChain** | ✅ Excellent | ✅ Pydantic native | ✅ Full | High | Largest ecosystem |
| LlamaIndex | ✅ Good | ✅ Good | ✅ Good | Medium | Better for pure RAG |
| AutoGen (Microsoft) | ✅ Good | ⚠️ Limited | ⚠️ Partial | Medium | Multi-agent focus |
| Custom (no framework) | ✅ Full control | ✅ Full control | ✅ Full | N/A | High dev cost |
| LangGraph | ✅ Excellent | ✅ Native | ✅ Full | Medium | LangChain's graph layer |

## Decision

**LangChain + LangGraph** is chosen.

### Rationale

- **Widest tool ecosystem**: 100+ pre-built tool integrations (DuckDuckGo, SerpAPI, etc.)
- **Native Pydantic v2 support**: structured output parsing via `.with_structured_output()`
- **AgentExecutor**: battle-tested ReAct loop with configurable step limits
- **LangGraph**: enables stateful, cyclic agent graphs for complex multi-step workflows
- **Community**: largest Python AI framework, extensive documentation and examples
- **Async-first**: `astream`, `ainvoke`, `abatch` on all primitives

### LlamaIndex Comparison

LlamaIndex excels at document indexing pipelines but has narrower tool-calling support. For NeuroAgent's primary use case (multi-tool autonomous agent), LangChain's `AgentExecutor` is a better fit out-of-the-box.

### Custom Framework Consideration

A lightweight custom orchestrator (`app/agent/orchestrator.py`) was implemented to avoid LangChain overhead for simple sequential pipelines. LangChain is used for text splitting (`RecursiveCharacterTextSplitter`) and will be used for complex multi-hop reasoning when `MOCK_LLM=false`.

## Consequences

- LangChain API surface changes frequently (pin exact versions)
- Adds ~50MB to Docker image (acceptable)
- LangGraph enables future upgrade to stateful multi-agent workflows with shared memory
