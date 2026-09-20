# 🧠 NeuroAgent — RAG Document Chatbot

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-purple)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-grade RAG chatbot** that lets you upload documents (PDF/TXT) and ask questions in natural language — powered by LangChain, ChromaDB vector database, and open-source HuggingFace models. **No API key required.**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     NeuroAgent RAG Pipeline                       │
│                                                                  │
│  PDF/TXT Document                                                │
│       │                                                          │
│       ▼                                                          │
│  Document Loader ──► Text Splitter ──► Chunk Documents           │
│  (PyPDFLoader)   (RecursiveChar)    (1000 tokens)               │
│                              │                                   │
│                              ▼                                   │
│                  HuggingFace Embeddings                          │
│               (all-MiniLM-L6-v2, 384-dim)                       │
│                              │                                   │
│                              ▼                                   │
│                    ChromaDB Vector Store ◄── Similarity Search   │
│                              │                   ▲              │
│                              │                   │              │
│                     User Question ────────────────┘             │
│                              │                                   │
│                              ▼                                   │
│                    Prompt + Context ──► FLAN-T5 LLM              │
│                                            │                    │
│                                            ▼                    │
│                                       Final Answer               │
│                                      + Source Docs               │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 📄 **Upload any PDF or TXT document** — instantly indexed
- 🔍 **Semantic search** via ChromaDB + sentence-transformers embeddings
- 🤖 **Free LLM inference** — google/flan-t5-base, no API key needed
- 💬 **Chat history** maintained across sessions
- 📚 **Source attribution** — see which document sections were used
- 🚀 **FastAPI backend** + **Streamlit UI** 
- 🐳 **Docker Compose** — one-command startup

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google FLAN-T5 (HuggingFace, free) |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| PDF Parsing | PyPDF |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Containerization | Docker |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/HasnainChavhan/neuro-agent
cd neuro-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start with Docker (recommended)
docker-compose up --build

# OR run locally:
# Terminal 1 - API
uvicorn src.api.main:app --reload --port 8000
# Terminal 2 - UI
streamlit run app/streamlit_app.py
```

- **Streamlit UI**: http://localhost:8501
- **FastAPI docs**: http://localhost:8000/docs

---

## 📡 API Reference

### POST `/upload` — Upload Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```
Response: `{"message": "Document indexed", "chunks": 42}`

### POST `/chat` — Ask Question
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```
Response:
```json
{
  "answer": "The document discusses...",
  "sources": ["chunk 1 excerpt...", "chunk 2 excerpt..."],
  "processing_time": 1.23
}
```

### GET `/stats` — Collection Statistics  
### DELETE `/reset` — Clear vector store  
### GET `/health` — Health check  

---

## 📁 Project Structure

```
neuro-agent/
├── src/
│   ├── config.py               # Configuration settings
│   ├── document_processor.py   # PDF/TXT loading & chunking
│   ├── vector_store.py         # ChromaDB management
│   ├── rag_chain.py            # LangChain RAG pipeline
│   └── api/
│       ├── main.py             # FastAPI endpoints
│       └── schemas.py          # Pydantic models
├── app/
│   └── streamlit_app.py        # Chat UI
├── tests/                      # pytest suite
├── sample_docs/sample.txt      # Demo document
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---
*Built by [Hasnain Chavhan](https://github.com/HasnainChavhan) — Open to NLP / ML Engineer roles*
