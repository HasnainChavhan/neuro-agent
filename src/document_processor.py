from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from .config import config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_text(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        return loader.load()

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(docs)
