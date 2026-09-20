"""Configuration for the NeuroAgent RAG system."""
from dataclasses import dataclass


@dataclass
class Config:
    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector database
    VECTOR_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "documents"
    TOP_K_RESULTS: int = 5

    # LLM
    LLM_MODEL: str = "google/flan-t5-base"
    LLM_MAX_LENGTH: int = 512
    LLM_TEMPERATURE: float = 0.7

    # API
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: tuple = (".pdf", ".txt")

    # Paths
    UPLOADED_DOCS_PATH: str = "./uploaded_docs"


config = Config()
