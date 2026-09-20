import pytest
from src.document_processor import DocumentProcessor
from langchain_core.documents import Document

def test_split_documents():
    processor = DocumentProcessor()
    doc = Document(page_content="This is a very long document. " * 100, metadata={"source": "test.txt"})
    chunks = processor.split_documents([doc])
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= processor.text_splitter._chunk_size
