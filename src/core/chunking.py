"""
Advanced chunking strategies with automatic optimization
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.document_loader import Document

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any]
    start_char: int
    end_char: int

    def __len__(self) -> int:
        return len(self.text)

    def __str__(self) -> str:
        return f"Chunk(id={self.chunk_id}, length={len(self.text)}, index={self.chunk_index})"


class BaseChunker:
    """Base class for text chunking"""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents into smaller pieces"""
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunking with overlap"""

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into fixed-size chunks"""
        all_chunks = []

        for doc in documents:
            text = doc.content
            doc_id = f"{doc.source}_{doc.page_number or 0}"

            chunks = self._split_text(text)

            for idx, chunk_text in enumerate(chunks):
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    chunk_index=idx,
                    metadata={
                        **doc.metadata,
                        "chunking_strategy": "fixed_size",
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                    },
                    start_char=idx * (self.chunk_size - self.overlap),
                    end_char=idx * (self.chunk_size - self.overlap) + len(chunk_text),
                )
                all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} fixed-size chunks from {len(documents)} documents")
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk:
                chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive chunking that respects document structure"""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        super().__init__(chunk_size, overlap)
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Recursively split documents at natural boundaries"""
        all_chunks = []

        for doc in documents:
            text = doc.content
            doc_id = f"{doc.source}_{doc.page_number or 0}"

            chunk_texts = self._recursive_split(text, self.chunk_size)

            for idx, chunk_text in enumerate(chunk_texts):
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    chunk_index=idx,
                    metadata={
                        **doc.metadata,
                        "chunking_strategy": "recursive",
                    },
                    start_char=0,  # Would need tracking in real implementation
                    end_char=len(chunk_text),
                )
                all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} recursive chunks from {len(documents)} documents")
        return all_chunks

    def _recursive_split(self, text: str, chunk_size: int) -> List[str]:
        """Recursively split text at natural boundaries"""
        if len(text) <= chunk_size:
            return [text] if text else []

        # Try each separator
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                chunks = []
                current_chunk = []
                current_size = 0

                for split in splits:
                    split_size = len(split) + len(separator)

                    if current_size + split_size <= chunk_size:
                        current_chunk.append(split)
                        current_size += split_size
                    else:
                        if current_chunk:
                            chunks.append(separator.join(current_chunk))
                        current_chunk = [split]
                        current_size = split_size

                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                return chunks

        # Fallback to character-level split
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


class SemanticChunker(BaseChunker):
    """Semantic chunking using embeddings similarity"""

    def __init__(
        self,
        chunk_size: int = 512,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
    ):
        super().__init__(chunk_size, 0)
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Create semantically coherent chunks"""
        all_chunks = []

        for doc in documents:
            text = doc.content
            doc_id = f"{doc.source}_{doc.page_number or 0}"

            # Split into sentences first
            sentences = self._split_into_sentences(text)

            if not sentences:
                continue

            # Group sentences by semantic similarity
            chunk_texts = self._group_by_similarity(sentences)

            for idx, chunk_text in enumerate(chunk_texts):
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    chunk_index=idx,
                    metadata={
                        **doc.metadata,
                        "chunking_strategy": "semantic",
                        "similarity_threshold": self.similarity_threshold,
                    },
                    start_char=0,
                    end_char=len(chunk_text),
                )
                all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} semantic chunks from {len(documents)} documents")
        return all_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_by_similarity(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks based on semantic similarity"""
        if not sentences:
            return []

        # Encode all sentences
        embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = np.dot(current_embedding, embeddings[i]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
            )

            # Check if we should add to current chunk or start new one
            current_text = ' '.join(current_chunk)
            if (
                similarity >= self.similarity_threshold
                and len(current_text) + len(sentences[i]) <= self.chunk_size
            ):
                current_chunk.append(sentences[i])
                # Update embedding to average
                current_embedding = (current_embedding + embeddings[i]) / 2
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class SentenceChunker(BaseChunker):
    """Chunk by complete sentences"""

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split into sentence-based chunks"""
        all_chunks = []

        for doc in documents:
            text = doc.content
            doc_id = f"{doc.source}_{doc.page_number or 0}"

            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunk_texts = self._group_sentences(sentences)

            for idx, chunk_text in enumerate(chunk_texts):
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    chunk_index=idx,
                    metadata={
                        **doc.metadata,
                        "chunking_strategy": "sentence",
                    },
                    start_char=0,
                    end_char=len(chunk_text),
                )
                all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} sentence chunks from {len(documents)} documents")
        return all_chunks

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks"""
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            if current_size + sentence_size <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


class ChunkOptimizer:
    """Automatically optimize chunk size for best retrieval performance"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def find_optimal_chunk_size(
        self,
        documents: List[Document],
        candidate_sizes: List[int] = [256, 512, 1024],
        test_queries: Optional[List[str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Find optimal chunk size through experimentation"""
        if not test_queries:
            # Generate synthetic queries from documents
            test_queries = self._generate_test_queries(documents)

        results = {}

        for size in candidate_sizes:
            chunker = FixedSizeChunker(chunk_size=size, overlap=size // 10)
            chunks = chunker.chunk(documents)

            # Evaluate retrieval quality
            score = self._evaluate_chunks(chunks, test_queries)
            results[size] = score

            logger.info(f"Chunk size {size}: score={score:.4f}")

        # Select best size
        optimal_size = max(results, key=results.get)

        return optimal_size, results

    def _generate_test_queries(self, documents: List[Document], n: int = 10) -> List[str]:
        """Generate test queries from document content"""
        # Extract key sentences as queries
        queries = []
        for doc in documents[:5]:  # Use first 5 docs
            sentences = re.split(r'(?<=[.!?])\s+', doc.content)
            # Pick middle sentences (usually more informative)
            if len(sentences) > 3:
                queries.extend(sentences[1:3])

        return queries[:n]

    def _evaluate_chunks(self, chunks: List[Chunk], queries: List[str]) -> float:
        """Evaluate chunk quality for retrieval"""
        if not chunks or not queries:
            return 0.0

        # Encode chunks and queries
        chunk_texts = [c.text for c in chunks[:100]]  # Limit for performance
        chunk_embeddings = self.model.encode(chunk_texts)
        query_embeddings = self.model.encode(queries)

        # Calculate average retrieval score
        scores = []
        for query_emb in query_embeddings:
            similarities = np.dot(chunk_embeddings, query_emb) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb)
            )
            # Get max similarity (best match)
            scores.append(np.max(similarities))

        return float(np.mean(scores))


class ChunkerFactory:
    """Factory for creating chunkers"""

    @staticmethod
    def create_chunker(
        strategy: ChunkingStrategy,
        chunk_size: int = 512,
        overlap: int = 50,
        **kwargs,
    ) -> BaseChunker:
        """Create chunker based on strategy"""
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(chunk_size, overlap)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(chunk_size, overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(chunk_size, **kwargs)
        elif strategy == ChunkingStrategy.SENTENCE:
            return SentenceChunker(chunk_size, overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
