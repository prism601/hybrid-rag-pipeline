"""
Cross-encoder re-ranking for improved retrieval quality
"""
from typing import List, Tuple, Optional
import logging

import numpy as np
from sentence_transformers import CrossEncoder

from src.core.chunking import Chunk
from src.core.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-rank retrieved chunks using cross-encoder"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ):
        """
        Initialize cross-encoder reranker

        Args:
            model_name: HuggingFace cross-encoder model
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded cross-encoder model: {model_name}")

    def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Re-rank chunks using cross-encoder

        Args:
            query: Search query
            chunks_with_scores: List of (chunk, score) tuples
            top_k: Number of top results to return

        Returns:
            Re-ranked list of (chunk, score) tuples
        """
        if not chunks_with_scores:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.text) for chunk, _ in chunks_with_scores]

        # Get cross-encoder scores
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Combine with chunks
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(chunks_with_scores, scores)
        ]

        # Sort by new scores
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        if top_k:
            reranked = reranked[:top_k]

        logger.info(f"Re-ranked {len(chunks_with_scores)} chunks to top {len(reranked)}")
        return reranked


class HybridReranker:
    """Combine multiple re-ranking signals"""

    def __init__(
        self,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        use_diversity: bool = True,
        diversity_weight: float = 0.2,
    ):
        """
        Initialize hybrid reranker

        Args:
            cross_encoder: Cross-encoder reranker
            use_diversity: Whether to apply diversity penalty
            diversity_weight: Weight for diversity signal
        """
        self.cross_encoder = cross_encoder or CrossEncoderReranker()
        self.use_diversity = use_diversity
        self.diversity_weight = diversity_weight

    def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Re-rank with multiple signals"""
        if not chunks_with_scores:
            return []

        # Get cross-encoder scores
        reranked = self.cross_encoder.rerank(query, chunks_with_scores, top_k=None)

        # Apply diversity if enabled
        if self.use_diversity:
            reranked = self._apply_diversity_penalty(reranked)

        # Sort and return top-k
        reranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def _apply_diversity_penalty(
        self,
        chunks_with_scores: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """Apply penalty to similar chunks"""
        if len(chunks_with_scores) <= 1:
            return chunks_with_scores

        # Calculate text similarity between chunks
        texts = [chunk.text for chunk, _ in chunks_with_scores]
        similarity_matrix = self._calculate_similarity_matrix(texts)

        # Apply diversity penalty
        adjusted_scores = []
        for i, (chunk, score) in enumerate(chunks_with_scores):
            # Calculate average similarity to higher-ranked chunks
            if i > 0:
                similarities = similarity_matrix[i, :i]
                avg_similarity = np.mean(similarities)
                diversity_penalty = avg_similarity * self.diversity_weight
                adjusted_score = score * (1 - diversity_penalty)
            else:
                adjusted_score = score

            adjusted_scores.append((chunk, adjusted_score))

        return adjusted_scores

    @staticmethod
    def _calculate_similarity_matrix(texts: List[str]) -> np.ndarray:
        """Calculate pairwise text similarity using Jaccard"""
        n = len(texts)
        similarity_matrix = np.zeros((n, n))

        # Tokenize texts
        tokenized = [set(text.lower().split()) for text in texts]

        for i in range(n):
            for j in range(i + 1, n):
                # Jaccard similarity
                intersection = len(tokenized[i] & tokenized[j])
                union = len(tokenized[i] | tokenized[j])
                similarity = intersection / union if union > 0 else 0
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix


class LostInMiddleReranker:
    """Reorder chunks to avoid 'lost in the middle' effect"""

    def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Reorder to avoid lost-in-middle effect
        Places most relevant chunks at beginning and end
        """
        if not chunks_with_scores:
            return []

        # Already sorted by score
        sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)

        if top_k:
            sorted_chunks = sorted_chunks[:top_k]

        # Reorder: best first, second-best last, third-best second, etc.
        reordered = []
        left = 0
        right = len(sorted_chunks) - 1
        at_start = True

        while left <= right:
            if at_start:
                reordered.append(sorted_chunks[left])
                left += 1
            else:
                reordered.append(sorted_chunks[right])
                right -= 1
            at_start = not at_start

        logger.info(f"Applied lost-in-middle reordering to {len(reordered)} chunks")
        return reordered


class ContextualReranker:
    """Re-rank based on contextual relevance"""

    def __init__(self, llm_client=None):
        """
        Initialize contextual reranker

        Args:
            llm_client: LLM client for relevance scoring
        """
        self.llm_client = llm_client

    def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Re-rank using LLM-based relevance assessment"""
        if not self.llm_client or not chunks_with_scores:
            return chunks_with_scores[:top_k] if top_k else chunks_with_scores

        # For each chunk, ask LLM to score relevance
        rescored = []

        for chunk, original_score in chunks_with_scores:
            relevance_score = self._assess_relevance(query, chunk.text)
            # Combine original and LLM score
            combined_score = 0.5 * original_score + 0.5 * relevance_score
            rescored.append((chunk, combined_score))

        # Sort by combined score
        rescored.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            rescored = rescored[:top_k]

        return rescored

    def _assess_relevance(self, query: str, text: str) -> float:
        """Use LLM to assess relevance on 0-1 scale"""
        prompt = f"""On a scale of 0 to 1, how relevant is the following text to the query?
Query: {query}

Text: {text[:500]}

Return only a number between 0 and 1."""

        try:
            # Simple relevance scoring
            # In production, would use actual LLM API
            response = self.llm_client.complete(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Failed to assess relevance: {e}")
            return 0.5  # Neutral score on error
