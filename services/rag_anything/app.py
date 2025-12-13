import asyncio
from http.client import HTTPException
import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor
from init import init_rag, get_api_key
from fastapi import FastAPI
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DATA_DIR = os.getenv("RAG_DATA_DIR", "./data")
WORKING_DIR = os.getenv("RAG_WORKING_DIR", "./app/rag_storage")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    get_api_key()
    app.state.rag = await init_rag(True)    

@app.post("/query")
async def answer_query(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    response = await app.state.rag.aquery(query, mode="hybrid")
    return {"response": response}

