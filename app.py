import os
import logging
import uvicorn
import asyncio 
import contextlib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq 

# --- IMPORT CORE LOGIC ---
from wiked import wik_ans 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GLOBAL RESOURCES & CONSTANTS ---
EMBEDDING_MODEL = None
LLM_CLIENT = None
LLM_MODEL_NAME = "llama-3.1-8b-instant" 

# --- FastAPI Models for API ---

class RAGQueryInput(BaseModel):
    query: str = Field(description="The user's question for factual QA.")

class RAGAnswerOutput(BaseModel):
    query: str
    answer: str
    retrieved_sources: List[str] 
    retrieval_method: str

# ----------------------------------------------------
## Lifespan Event Handler (Replaces on_event)
# ----------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """
    Initializes global resources (models, clients) during startup 
    and performs cleanup during shutdown.
    """
    global EMBEDDING_MODEL, LLM_CLIENT
    
    # --- STARTUP LOGIC ---
    logger.info("Application starting up: Loading models...")
    
    # 1. Embedding Model (Singleton)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=model_name)
    logger.info("Embedding Model initialized.")
    
    # 2. LLM Client (Singleton)
    logger.info(f"Initializing LLM Client: {LLM_MODEL_NAME}...")
    try:
        LLM_CLIENT = ChatGroq(temperature=0.0, model_name=LLM_MODEL_NAME)
        logger.info("LLM Client initialized.")
    except Exception as e:
        logger.error(f"LLM Client initialization failed. Check API Key: {e}")
        LLM_CLIENT = None 

    # Yield control back to FastAPI to handle requests
    yield
    
    # --- SHUTDOWN LOGIC (Cleanup is usually minimal for these resources) ---
    logger.info("Application shutting down.")
    # Add any necessary resource cleanup here (e.g., closing database connections)
    
# ----------------------------------------------------
# FastAPI Setup
# ----------------------------------------------------

# Pass the lifespan context manager to the FastAPI instance
app = FastAPI(
    title="Wiked: Advanced RAG Microservice",
    description="Low-latency API for Retrieval-Augmented Generation.",
    lifespan=lifespan_handler 
)
    
# --- RAG PIPELINE ENDPOINT ---

@app.post("/wiked_answer", response_model=RAGAnswerOutput)
async def wiked_answer_endpoint(data: RAGQueryInput):
    """
    Executes the full LLM-enhanced RAG pipeline by orchestrating 
    topic generation, retrieval, and final answer generation.
    """
    
    if not LLM_CLIENT or not EMBEDDING_MODEL:
        raise HTTPException(status_code=503, 
                            detail="Service not ready: Core models failed to initialize.")

    try:
        # Call the main logic function imported from wiked.py
        # This function must return the final answer, a list of sources, and the method string.
        answer, sources, method= wik_ans(
            user_query=data.query, 
            model=LLM_CLIENT, 
            embedding=EMBEDDING_MODEL
        )
        
        return RAGAnswerOutput(
            query=data.query,
            answer=answer,
            retrieved_sources=sources,
            retrieval_method=method
        )
    
    except Exception as e:
        logger.error(f"Error during RAG pipeline execution: {e}", exc_info=True)
        # Detailed internal error message for debugging
        raise HTTPException(status_code=500, detail=f"RAG execution failed due to an internal error: {e}")