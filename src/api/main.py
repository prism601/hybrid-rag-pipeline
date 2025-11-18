"""
FastAPI application for hybrid RAG pipeline
"""
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
from pathlib import Path
import tempfile

from src.core.rag_pipeline import RAGPipeline, RAGResponse
from src.core.config import settings
from src.utils.cache import cache_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid RAG Pipeline API",
    description="Enterprise-grade Retrieval Augmented Generation system",
    version="1.0.0",
)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize RAG pipeline
rag_pipeline: Optional[RAGPipeline] = None


# Request/Response models
class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    use_reranking: bool = Field(default=True, description="Enable reranking")
    generate_answer: bool = Field(default=False, description="Generate LLM answer")


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    contexts: List[str]
    scores: List[float]
    citations: List[Dict[str, Any]]
    hallucination_detected: Optional[bool]
    hallucination_score: Optional[float]
    ragas_score: Optional[Dict[str, float]]
    metadata: Dict[str, Any]


class IndexRequest(BaseModel):
    """Index request model"""
    file_paths: List[str] = Field(..., description="List of file paths to index")
    optimize_chunks: bool = Field(default=False, description="Optimize chunk size")


class IndexResponse(BaseModel):
    """Index response model"""
    num_documents: int
    num_chunks: int
    avg_chunk_size: float
    status: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    pipeline_initialized: bool
    num_indexed_chunks: int


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global rag_pipeline
    try:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(
            use_vector_store=True,
            use_bm25=True,
            use_reranking=True,
            use_hallucination_detection=settings.enable_faithfulness_check,
            use_citations=True,
        )
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Hybrid RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if rag_pipeline else "unhealthy",
        version="1.0.0",
        pipeline_initialized=rag_pipeline is not None,
        num_indexed_chunks=len(rag_pipeline.chunks) if rag_pipeline else 0,
    )


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index documents into the pipeline"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        stats = rag_pipeline.index_documents(
            file_paths=request.file_paths,
            optimize_chunks=request.optimize_chunks,
        )

        return IndexResponse(
            num_documents=stats['num_documents'],
            num_chunks=stats['num_chunks'],
            avg_chunk_size=stats['avg_chunk_size'],
            status="success",
        )
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    # Save uploaded file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Index the file
        stats = rag_pipeline.index_documents([tmp_path])

        # Clean up
        Path(tmp_path).unlink()

        return {
            "filename": file.filename,
            "status": "indexed",
            "num_chunks": stats['num_chunks'],
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG pipeline"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    if not rag_pipeline.chunks:
        raise HTTPException(status_code=400, detail="No documents indexed")

    try:
        response = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            generate_answer=request.generate_answer,
        )

        # Convert citations to dict
        citations = [c.to_dict() for c in response.citations]

        # Extract RAGAS scores
        ragas_scores = None
        if response.ragas_score:
            ragas_scores = {
                "faithfulness": response.ragas_score.faithfulness,
                "answer_relevance": response.ragas_score.answer_relevance,
                "context_precision": response.ragas_score.context_precision,
                "context_recall": response.ragas_score.context_recall,
                "overall_score": response.ragas_score.overall_score,
            }

        return QueryResponse(
            answer=response.answer,
            contexts=response.contexts,
            scores=response.scores,
            citations=citations,
            hallucination_detected=response.hallucination_score.is_hallucination if response.hallucination_score else None,
            hallucination_score=response.hallucination_score.faithfulness_score if response.hallucination_score else None,
            ragas_score=ragas_scores,
            metadata=response.metadata,
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/batch")
async def batch_query(questions: List[str], top_k: int = 5):
    """Process multiple queries in batch"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        responses = rag_pipeline.batch_query(questions, top_k=top_k)

        return {
            "results": [
                {
                    "question": q,
                    "answer": r.answer,
                    "num_contexts": len(r.contexts),
                    "avg_score": sum(r.scores) / len(r.scores) if r.scores else 0,
                }
                for q, r in zip(questions, responses)
            ]
        }
    except Exception as e:
        logger.error(f"Batch query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """Clear the cache"""
    try:
        cache_manager.clear()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    return {
        "num_chunks": len(rag_pipeline.chunks),
        "retrieval_method": rag_pipeline._get_retrieval_method(),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "vector_db_type": settings.vector_db_type,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )
