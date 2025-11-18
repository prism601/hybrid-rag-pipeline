"""
BM25 and sparse retrieval implementations
"""
from typing import List, Optional, Dict, Any, Tuple
import logging
from collections import defaultdict
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
import re

from src.core.chunking import Chunk

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 sparse retrieval with persistence"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 retriever

        Args:
            k1: Controls term frequency saturation (typical: 1.2-2.0)
            b: Controls document length normalization (typical: 0.75)
            epsilon: Floor value for IDF
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Index chunks for BM25 retrieval"""
        if not chunks:
            return

        self.chunks.extend(chunks)

        # Tokenize all texts
        texts = [chunk.text for chunk in chunks]
        tokenized = [self._tokenize(text) for text in texts]
        self.tokenized_corpus.extend(tokenized)

        # Build/rebuild BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )

        logger.info(f"Indexed {len(chunks)} chunks with BM25")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search using BM25"""
        if self.bm25 is None or not self.chunks:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k * 2]  # Get extra for filtering

        # Create results
        results = []
        for idx in top_indices:
            if idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            score = float(scores[idx])

            # Apply filter
            if filter and not self._matches_filter(chunk, filter):
                continue

            results.append((chunk, score))

            if len(results) >= top_k:
                break

        return results

    def save(self, path: str) -> None:
        """Save BM25 index to disk"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'k1': self.k1,
            'b': self.b,
            'epsilon': self.epsilon,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved BM25 index to {path}")

    def load(self, path: str) -> None:
        """Load BM25 index from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.bm25 = data['bm25']
        self.chunks = data['chunks']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data.get('k1', 1.5)
        self.b = data.get('b', 0.75)
        self.epsilon = data.get('epsilon', 0.25)

        logger.info(f"Loaded BM25 index from {path}")

    def delete_all(self) -> None:
        """Clear all indexed data"""
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
        logger.info("Cleared BM25 index")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    @staticmethod
    def _matches_filter(chunk: Chunk, filter: Dict[str, Any]) -> bool:
        """Check if chunk matches filter"""
        for key, value in filter.items():
            if chunk.metadata.get(key) != value:
                return False
        return True


class TFIDFRetriever:
    """TF-IDF based retrieval (alternative to BM25)"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = None
        self.chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Index chunks with TF-IDF"""
        if not chunks:
            return

        self.chunks.extend(chunks)

        # Build TF-IDF matrix
        texts = [chunk.text for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        logger.info(f"Indexed {len(chunks)} chunks with TF-IDF")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search using TF-IDF cosine similarity"""
        if self.tfidf_matrix is None or not self.chunks:
            return []

        # Transform query
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])

            if filter and not self._matches_filter(chunk, filter):
                continue

            results.append((chunk, score))

            if len(results) >= top_k:
                break

        return results

    def delete_all(self) -> None:
        """Clear all indexed data"""
        self.tfidf_matrix = None
        self.chunks = []
        logger.info("Cleared TF-IDF index")

    @staticmethod
    def _matches_filter(chunk: Chunk, filter: Dict[str, Any]) -> bool:
        """Check if chunk matches filter"""
        for key, value in filter.items():
            if chunk.metadata.get(key) != value:
                return False
        return True


class HybridRetriever:
    """Combines dense and sparse retrieval with weighted fusion"""

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        alpha: float = 0.5,
    ):
        """
        Initialize hybrid retriever

        Args:
            dense_retriever: Dense retrieval (vector store)
            sparse_retriever: Sparse retrieval (BM25/TF-IDF)
            alpha: Weight for dense retrieval (1-alpha for sparse)
                  alpha=1: Only dense, alpha=0: Only sparse
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Hybrid search with score fusion"""
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query, top_k=top_k * 2, filter=filter)
        sparse_results = self.sparse_retriever.search(query, top_k=top_k * 2, filter=filter)

        # Normalize scores to [0, 1]
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)

        # Combine scores
        combined_scores: Dict[str, Tuple[Chunk, float]] = {}

        for chunk, score in dense_normalized:
            combined_scores[chunk.chunk_id] = (chunk, self.alpha * score)

        for chunk, score in sparse_normalized:
            if chunk.chunk_id in combined_scores:
                existing_chunk, existing_score = combined_scores[chunk.chunk_id]
                combined_scores[chunk.chunk_id] = (
                    existing_chunk,
                    existing_score + (1 - self.alpha) * score
                )
            else:
                combined_scores[chunk.chunk_id] = (chunk, (1 - self.alpha) * score)

        # Sort by combined score
        results = list(combined_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    @staticmethod
    def _normalize_scores(results: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """Normalize scores to [0, 1] range"""
        if not results:
            return []

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [(chunk, 1.0) for chunk, _ in results]

        normalized = [
            (chunk, (score - min_score) / (max_score - min_score))
            for chunk, score in results
        ]

        return normalized


class MMRRetriever:
    """Maximal Marginal Relevance retriever for diversity"""

    def __init__(
        self,
        base_retriever,
        lambda_mult: float = 0.5,
    ):
        """
        Initialize MMR retriever

        Args:
            base_retriever: Base retriever to use
            lambda_mult: Balance between relevance and diversity
                        1 = only relevance, 0 = only diversity
        """
        self.base_retriever = base_retriever
        self.lambda_mult = lambda_mult

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        embedding_model=None,
    ) -> List[Tuple[Chunk, float]]:
        """Search with MMR for diverse results"""
        # Get initial candidates
        candidates = self.base_retriever.search(
            query,
            top_k=top_k * 3,
            filter=filter
        )

        if not candidates or not embedding_model:
            return candidates[:top_k]

        # Get embeddings
        query_emb = embedding_model.encode([query])[0]
        candidate_texts = [chunk.text for chunk, _ in candidates]
        candidate_embs = embedding_model.encode(candidate_texts)

        # MMR selection
        selected_indices = []
        selected_embs = []

        while len(selected_indices) < top_k and len(selected_indices) < len(candidates):
            best_idx = None
            best_score = float('-inf')

            for i, (chunk, relevance_score) in enumerate(candidates):
                if i in selected_indices:
                    continue

                # Relevance to query
                relevance = self._cosine_similarity(query_emb, candidate_embs[i])

                # Diversity (max similarity to already selected)
                if selected_embs:
                    similarities = [
                        self._cosine_similarity(candidate_embs[i], sel_emb)
                        for sel_emb in selected_embs
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0

                # MMR score
                mmr_score = self.lambda_mult * relevance - (1 - self.lambda_mult) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embs.append(candidate_embs[best_idx])

        # Return selected results
        return [candidates[i] for i in selected_indices]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
