from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGChain
from src.api.schemas import ChatRequest, ChatResponse, UploadResponse, StatsResponse
import os
import time

app = FastAPI(title="Neuro-Agent RAG API", description="LLM-powered document chatbot backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploaded_docs", exist_ok=True)

# Initialize components
doc_processor = DocumentProcessor()
vector_store = VectorStoreManager()
rag_chain = RAGChain(vector_store)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()
    file_path = f"uploaded_docs/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        if file.filename.endswith(".pdf"):
            docs = doc_processor.load_pdf(file_path)
        elif file.filename.endswith(".txt"):
            docs = doc_processor.load_text(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        chunks = doc_processor.split_documents(docs)
        vector_store.add_documents(chunks)
        
        return {
            "filename": file.filename,
            "message": "File processed and indexed successfully",
            "chunks_added": len(chunks),
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    try:
        result = rag_chain.answer(request.question)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    stats = vector_store.get_collection_stats()
    return {"document_count": stats["document_count"]}

@app.delete("/reset")
def reset_db():
    vector_store.delete_collection()
    return {"message": "Vector store reset successfully"}
