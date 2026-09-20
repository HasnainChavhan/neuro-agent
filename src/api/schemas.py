from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    processing_time: float

class UploadResponse(BaseModel):
    filename: str
    message: str
    chunks_added: int
    processing_time: float

class StatsResponse(BaseModel):
    document_count: int
