"""
Vector store implementations for dense retrieval
"""
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.chunking import Chunk
from src.core.embeddings import EmbeddingModel, EmbeddingFactory
from src.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Base vector store interface"""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to vector store"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks"""
        pass

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all vectors"""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "chroma_db",
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding model
        self.embedding_model = embedding_model or EmbeddingFactory.create_embedding_model()

        logger.info(
            f"Initialized ChromaDB vector store: {collection_name} "
            f"(dim={self.embedding_model.get_dimension()})"
        )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to ChromaDB"""
        if not chunks:
            return

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, batch_size=settings.batch_size)

        # Prepare data
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter,
        )

        # Convert to chunks
        chunks_with_scores = []
        if results['ids'] and results['ids'][0]:
            for idx, chunk_id in enumerate(results['ids'][0]):
                chunk = Chunk(
                    text=results['documents'][0][idx],
                    chunk_id=chunk_id,
                    doc_id=results['metadatas'][0][idx].get('doc_id', ''),
                    chunk_index=results['metadatas'][0][idx].get('chunk_index', idx),
                    metadata=results['metadatas'][0][idx],
                    start_char=0,
                    end_char=len(results['documents'][0][idx]),
                )
                score = 1 - results['distances'][0][idx]  # Convert distance to similarity
                chunks_with_scores.append((chunk, score))

        return chunks_with_scores

    def delete_all(self) -> None:
        """Delete all vectors"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Deleted all vectors from {self.collection_name}")


class FAISSVectorStore(VectorStore):
    """FAISS vector store for efficient similarity search"""

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        import faiss

        self.index_path = index_path
        self.embedding_model = embedding_model or EmbeddingFactory.create_embedding_model()
        self.dimension = self.embedding_model.get_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store chunks separately
        self.chunks: List[Chunk] = []

        # Load index if exists
        if index_path and Path(index_path).exists():
            self._load_index()

        logger.info(f"Initialized FAISS vector store (dim={self.dimension})")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to FAISS index"""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, batch_size=settings.batch_size)

        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to FAISS index")

        # Save if path provided
        if self.index_path:
            self._save_index()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks"""
        query_embedding = self.embedding_model.encode([query])[0]

        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32').reshape(1, -1),
            top_k
        )

        # Convert to results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]

                # Apply filter if provided
                if filter:
                    if not self._matches_filter(chunk, filter):
                        continue

                # Convert L2 distance to similarity score
                score = 1 / (1 + distance)
                results.append((chunk, score))

        return results

    def delete_all(self) -> None:
        """Delete all vectors"""
        import faiss
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        logger.info("Deleted all vectors from FAISS index")

    def _save_index(self) -> None:
        """Save FAISS index to disk"""
        import faiss
        import pickle

        if not self.index_path:
            return

        index_path = Path(self.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))

        # Save chunks
        chunks_path = index_path.with_suffix('.chunks')
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Saved FAISS index to {index_path}")

    def _load_index(self) -> None:
        """Load FAISS index from disk"""
        import faiss
        import pickle

        if not self.index_path:
            return

        index_path = Path(self.index_path)

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load chunks
        chunks_path = index_path.with_suffix('.chunks')
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)

        logger.info(f"Loaded FAISS index from {index_path}")

    @staticmethod
    def _matches_filter(chunk: Chunk, filter: Dict[str, Any]) -> bool:
        """Check if chunk matches filter"""
        for key, value in filter.items():
            if chunk.metadata.get(key) != value:
                return False
        return True


class VectorStoreFactory:
    """Factory for creating vector stores"""

    @staticmethod
    def create_vector_store(
        store_type: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        **kwargs,
    ) -> VectorStore:
        """Create vector store based on type"""
        store_type = store_type or settings.vector_db_type

        if store_type == "chromadb":
            return ChromaVectorStore(embedding_model=embedding_model, **kwargs)
        elif store_type == "faiss":
            return FAISSVectorStore(embedding_model=embedding_model, **kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
