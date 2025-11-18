"""
Unit tests for RAG pipeline
"""
import pytest
from pathlib import Path
import tempfile

from src.core.rag_pipeline import RAGPipeline
from src.core.document_loader import Document
from src.core.chunking import ChunkerFactory, ChunkingStrategy


@pytest.fixture
def sample_document():
    """Create sample document"""
    return Document(
        content="This is a test document about machine learning and AI.",
        metadata={"source": "test"},
        source="test.txt",
        doc_type="text"
    )


@pytest.fixture
def rag_pipeline():
    """Create RAG pipeline instance"""
    return RAGPipeline(
        use_vector_store=True,
        use_bm25=True,
        use_reranking=False,
        use_hallucination_detection=False,
    )


def test_pipeline_initialization(rag_pipeline):
    """Test pipeline initializes correctly"""
    assert rag_pipeline is not None
    assert rag_pipeline.vector_store is not None
    assert rag_pipeline.bm25_retriever is not None


def test_document_indexing(rag_pipeline):
    """Test document indexing"""
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for indexing")
        temp_path = f.name

    try:
        stats = rag_pipeline.index_documents([temp_path])

        assert stats['num_documents'] > 0
        assert stats['num_chunks'] > 0
        assert len(rag_pipeline.chunks) > 0
    finally:
        Path(temp_path).unlink()


def test_query(rag_pipeline):
    """Test querying"""
    # Create and index test document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Machine learning is a subset of artificial intelligence.")
        temp_path = f.name

    try:
        rag_pipeline.index_documents([temp_path])

        response = rag_pipeline.query(
            question="What is machine learning?",
            top_k=3
        )

        assert response is not None
        assert len(response.contexts) > 0
        assert len(response.chunks) > 0
        assert len(response.scores) > 0
    finally:
        Path(temp_path).unlink()


def test_chunking_strategies():
    """Test different chunking strategies"""
    document = Document(
        content="A" * 1000,  # Long text
        metadata={},
        source="test",
        doc_type="text"
    )

    for strategy in [ChunkingStrategy.FIXED_SIZE, ChunkingStrategy.RECURSIVE]:
        chunker = ChunkerFactory.create_chunker(
            strategy=strategy,
            chunk_size=100,
            overlap=20
        )
        chunks = chunker.chunk([document])
        assert len(chunks) > 0


def test_batch_query(rag_pipeline):
    """Test batch querying"""
    # Index test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("AI and machine learning are related fields.")
        temp_path = f.name

    try:
        rag_pipeline.index_documents([temp_path])

        questions = ["What is AI?", "What is ML?"]
        responses = rag_pipeline.batch_query(questions, top_k=2)

        assert len(responses) == len(questions)
        assert all(r.answer for r in responses)
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
