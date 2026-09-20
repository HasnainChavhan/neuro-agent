from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
from langchain_core.documents import Document
from .config import config
import shutil
import os

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.persist_directory = config.VECTOR_DB_PATH
        self._init_vector_store()

    def _init_vector_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

    def add_documents(self, chunks: List[Document]) -> None:
        self.vector_store.add_documents(documents=chunks)
        self.vector_store.persist()

    def similarity_search(self, query: str, k: int = config.TOP_K_RESULTS) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config.TOP_K_RESULTS})

    def delete_collection(self) -> None:
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self._init_vector_store()

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.vector_store._collection.count()
            return {"document_count": count}
        except Exception:
            return {"document_count": 0}
