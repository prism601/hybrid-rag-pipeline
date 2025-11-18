"""
Main RAG Pipeline orchestrating all components
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import asyncio

from src.core.document_loader import DocumentLoaderFactory, Document
from src.core.chunking import ChunkerFactory, ChunkingStrategy, Chunk, ChunkOptimizer
from src.core.embeddings import EmbeddingFactory, EmbeddingModel
from src.retrieval.vector_store import VectorStoreFactory, VectorStore
from src.retrieval.sparse_retrieval import BM25Retriever, HybridRetriever, MMRRetriever
from src.retrieval.knowledge_graph import KnowledgeGraphExtractor, KnowledgeGraph, KnowledgeGraphRetriever
from src.reranking.cross_encoder import CrossEncoderReranker, HybridReranker
from src.evaluation.hallucination_detection import HallucinationDetector, HallucinationScore
from src.evaluation.ragas_metrics import RAGASEvaluator, RAGASScore
from src.utils.citation import CitationGenerator, Citation
from src.utils.cache import cached
from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG pipeline"""
    answer: str
    contexts: List[str]
    chunks: List[Chunk]
    scores: List[float]
    citations: List[Citation]
    hallucination_score: Optional[HallucinationScore]
    ragas_score: Optional[RAGASScore]
    metadata: Dict[str, Any]


class RAGPipeline:
    """Enterprise RAG pipeline with all features"""

    def __init__(
        self,
        use_vector_store: bool = True,
        use_bm25: bool = True,
        use_knowledge_graph: bool = False,
        use_reranking: bool = True,
        use_hallucination_detection: bool = True,
        use_citations: bool = True,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize RAG pipeline

        Args:
            use_vector_store: Enable dense retrieval
            use_bm25: Enable sparse retrieval
            use_knowledge_graph: Enable knowledge graph
            use_reranking: Enable cross-encoder reranking
            use_hallucination_detection: Enable hallucination detection
            use_citations: Enable citation generation
            embedding_model: Custom embedding model
        """
        logger.info("Initializing RAG pipeline...")

        # Document loading
        self.document_loader = DocumentLoaderFactory()

        # Chunking
        self.chunker = ChunkerFactory.create_chunker(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=settings.default_chunk_size,
            overlap=settings.default_chunk_overlap,
        )
        self.chunk_optimizer = ChunkOptimizer() if settings.auto_optimize_chunks else None

        # Embedding
        self.embedding_model = embedding_model or EmbeddingFactory.create_embedding_model()

        # Retrievers
        self.use_vector_store = use_vector_store
        self.use_bm25 = use_bm25
        self.use_knowledge_graph = use_knowledge_graph

        if use_vector_store:
            self.vector_store = VectorStoreFactory.create_vector_store(
                embedding_model=self.embedding_model
            )
        else:
            self.vector_store = None

        if use_bm25:
            self.bm25_retriever = BM25Retriever()
        else:
            self.bm25_retriever = None

        # Hybrid retrieval
        if use_vector_store and use_bm25:
            self.hybrid_retriever = HybridRetriever(
                dense_retriever=self.vector_store,
                sparse_retriever=self.bm25_retriever,
                alpha=settings.hybrid_alpha,
            )
        else:
            self.hybrid_retriever = None

        # MMR for diversity
        if settings.use_mmr and use_vector_store:
            self.mmr_retriever = MMRRetriever(
                base_retriever=self.vector_store,
                lambda_mult=1 - settings.mmr_diversity_score,
            )
        else:
            self.mmr_retriever = None

        # Knowledge graph
        if use_knowledge_graph:
            self.kg_extractor = KnowledgeGraphExtractor()
            self.knowledge_graph: Optional[KnowledgeGraph] = None
            self.kg_retriever: Optional[KnowledgeGraphRetriever] = None
        else:
            self.kg_extractor = None
            self.knowledge_graph = None
            self.kg_retriever = None

        # Reranking
        if use_reranking:
            self.reranker = HybridReranker()
        else:
            self.reranker = None

        # Hallucination detection
        if use_hallucination_detection:
            self.hallucination_detector = HallucinationDetector()
        else:
            self.hallucination_detector = None

        # Evaluation
        self.ragas_evaluator = RAGASEvaluator() if settings.enable_ragas_metrics else None

        # Citation generation
        if use_citations:
            self.citation_generator = CitationGenerator()
        else:
            self.citation_generator = None

        # Storage
        self.chunks: List[Chunk] = []
        self.chunk_store: Dict[str, Chunk] = {}

        logger.info("RAG pipeline initialized successfully")

    def index_documents(
        self,
        file_paths: List[str],
        optimize_chunks: bool = False,
    ) -> Dict[str, Any]:
        """
        Index documents into the pipeline

        Args:
            file_paths: List of document paths
            optimize_chunks: Whether to optimize chunk size

        Returns:
            Indexing statistics
        """
        logger.info(f"Indexing {len(file_paths)} documents...")

        # Load documents
        documents = self.document_loader.load_documents(file_paths)
        logger.info(f"Loaded {len(documents)} document pages")

        # Optimize chunk size if enabled
        if optimize_chunks and self.chunk_optimizer:
            optimal_size, _ = self.chunk_optimizer.find_optimal_chunk_size(documents)
            logger.info(f"Optimal chunk size: {optimal_size}")
            self.chunker.chunk_size = optimal_size

        # Chunk documents
        chunks = self.chunker.chunk(documents)
        self.chunks.extend(chunks)

        # Build chunk store
        for chunk in chunks:
            self.chunk_store[chunk.chunk_id] = chunk

        logger.info(f"Created {len(chunks)} chunks")

        # Index into vector store
        if self.vector_store:
            self.vector_store.add_chunks(chunks)

        # Index into BM25
        if self.bm25_retriever:
            self.bm25_retriever.add_chunks(chunks)

        # Build knowledge graph
        if self.kg_extractor:
            self.knowledge_graph = self.kg_extractor.extract_from_chunks(chunks)
            self.kg_retriever = KnowledgeGraphRetriever(
                self.knowledge_graph,
                self.chunk_store
            )

        stats = {
            'num_documents': len(documents),
            'num_chunks': len(chunks),
            'avg_chunk_size': sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
            'indexed_in_vector_store': self.use_vector_store,
            'indexed_in_bm25': self.use_bm25,
            'has_knowledge_graph': self.use_knowledge_graph,
        }

        logger.info(f"Indexing complete: {stats}")
        return stats

    @cached(ttl=3600)
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        generate_answer: bool = False,
        llm_client: Optional[Any] = None,
    ) -> RAGResponse:
        """
        Query the RAG pipeline

        Args:
            question: User question
            top_k: Number of results to return
            use_reranking: Whether to rerank results
            generate_answer: Whether to generate answer with LLM
            llm_client: LLM client for answer generation

        Returns:
            RAG response with results
        """
        logger.info(f"Query: {question}")

        # Retrieve relevant chunks
        chunks_with_scores = self._retrieve(question, top_k * 2)

        # Rerank if enabled
        if use_reranking and self.reranker:
            chunks_with_scores = self.reranker.rerank(
                question,
                chunks_with_scores,
                top_k=top_k
            )
        else:
            chunks_with_scores = chunks_with_scores[:top_k]

        # Extract chunks and contexts
        chunks = [chunk for chunk, _ in chunks_with_scores]
        scores = [score for _, score in chunks_with_scores]
        contexts = [chunk.text for chunk in chunks]

        # Generate answer if requested
        answer = ""
        if generate_answer and llm_client:
            answer = self._generate_answer(question, contexts, llm_client)
        else:
            # Return concatenated contexts
            answer = "\n\n".join(contexts[:3])

        # Generate citations
        citations = []
        if self.citation_generator and answer:
            answer, citations = self.citation_generator.generate_citations(
                answer,
                chunks_with_scores
            )

        # Detect hallucinations
        hallucination_score = None
        if self.hallucination_detector and answer:
            hallucination_score = self.hallucination_detector.detect(answer, contexts)

        # RAGAS evaluation
        ragas_score = None
        if self.ragas_evaluator and answer:
            ragas_score = self.ragas_evaluator.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
            )

        response = RAGResponse(
            answer=answer,
            contexts=contexts,
            chunks=chunks,
            scores=scores,
            citations=citations,
            hallucination_score=hallucination_score,
            ragas_score=ragas_score,
            metadata={
                'num_chunks': len(chunks),
                'retrieval_method': self._get_retrieval_method(),
                'reranked': use_reranking,
            }
        )

        logger.info(f"Query complete: {len(chunks)} chunks retrieved")
        return response

    def _retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """Internal retrieval method"""
        if self.hybrid_retriever:
            # Use hybrid retrieval
            results = self.hybrid_retriever.search(query, top_k=top_k)
        elif self.vector_store:
            # Use dense retrieval
            results = self.vector_store.search(query, top_k=top_k)
        elif self.bm25_retriever:
            # Use sparse retrieval
            results = self.bm25_retriever.search(query, top_k=top_k)
        else:
            logger.warning("No retriever available")
            return []

        # Apply MMR for diversity if enabled
        if self.mmr_retriever and settings.use_mmr:
            results = self.mmr_retriever.search(
                query,
                top_k=top_k,
                embedding_model=self.embedding_model,
            )

        return results

    def _generate_answer(
        self,
        question: str,
        contexts: List[str],
        llm_client: Any,
    ) -> str:
        """Generate answer using LLM"""
        context_str = "\n\n".join(f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts))

        prompt = f"""Answer the following question based on the provided contexts. Be concise and accurate.

Question: {question}

{context_str}

Answer:"""

        try:
            response = llm_client.complete(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer"

    def _get_retrieval_method(self) -> str:
        """Get description of retrieval method"""
        methods = []
        if self.hybrid_retriever:
            methods.append("hybrid")
        elif self.vector_store:
            methods.append("dense")
        elif self.bm25_retriever:
            methods.append("sparse")

        if self.use_knowledge_graph:
            methods.append("knowledge_graph")

        return "+".join(methods) if methods else "none"

    async def query_async(self, *args, **kwargs) -> RAGResponse:
        """Async version of query"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(*args, **kwargs))

    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[RAGResponse]:
        """Process multiple queries in batch"""
        return [self.query(q, **kwargs) for q in questions]

    async def batch_query_async(
        self,
        questions: List[str],
        **kwargs
    ) -> List[RAGResponse]:
        """Async batch query"""
        tasks = [self.query_async(q, **kwargs) for q in questions]
        return await asyncio.gather(*tasks)
