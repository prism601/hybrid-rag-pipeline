"""
Embedding generation with caching and batch processing
"""
from typing import List, Optional, Dict, Any
import hashlib
import pickle
from pathlib import Path
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
import openai

from src.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Base embedding model interface"""

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        raise NotImplementedError

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence Transformers embedding model"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded SentenceTransformer model: {model_name}")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model"""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self._dimension = 1536 if "3-small" in model_name else 3072
        logger.info(f"Initialized OpenAI embedding model: {model_name}")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension


class CachedEmbedding(EmbeddingModel):
    """Embedding model with disk caching"""

    def __init__(self, base_model: EmbeddingModel, cache_dir: str = "data/embeddings"):
        self.base_model = base_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, np.ndarray] = {}
        logger.info(f"Initialized cached embeddings with cache_dir={cache_dir}")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode with caching"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)

            # Check memory cache
            if cache_key in self._memory_cache:
                embeddings.append((i, self._memory_cache[cache_key]))
            else:
                # Check disk cache
                cached_emb = self._load_from_disk(cache_key)
                if cached_emb is not None:
                    embeddings.append((i, cached_emb))
                    self._memory_cache[cache_key] = cached_emb
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

        # Compute uncached embeddings
        if uncached_texts:
            new_embeddings = self.base_model.encode(uncached_texts, batch_size)

            # Cache new embeddings
            for text, emb in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text)
                self._save_to_disk(cache_key, emb)
                self._memory_cache[cache_key] = emb

            # Add to results
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, emb))

        # Sort by original index
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.base_model.get_dimension()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use hash of model name + text
        key_str = f"{self.base_model.model_name}:{text}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _load_from_disk(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from disk"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_to_disk(self, cache_key: str, embedding: np.ndarray) -> None:
        """Save embedding to disk"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")


class EmbeddingFactory:
    """Factory for creating embedding models"""

    @staticmethod
    def create_embedding_model(
        model_name: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: str = "data/embeddings",
    ) -> EmbeddingModel:
        """Create embedding model"""
        model_name = model_name or settings.embedding_model

        # Determine model type
        if model_name.startswith("text-embedding"):
            base_model = OpenAIEmbedding(model_name)
        else:
            base_model = SentenceTransformerEmbedding(model_name)

        # Add caching
        if use_cache:
            return CachedEmbedding(base_model, cache_dir)

        return base_model
