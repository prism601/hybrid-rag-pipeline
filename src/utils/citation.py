"""
Citation generation with confidence scores
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
import logging

from src.core.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a citation with source information"""
    text: str
    source: str
    page_number: Optional[int]
    chunk_id: str
    confidence: float
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'source': self.source,
            'page_number': self.page_number,
            'chunk_id': self.chunk_id,
            'confidence': self.confidence,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'metadata': self.metadata,
        }

    def format_citation(self, style: str = "apa") -> str:
        """Format citation in specified style"""
        if style == "apa":
            return self._format_apa()
        elif style == "mla":
            return self._format_mla()
        elif style == "chicago":
            return self._format_chicago()
        else:
            return self._format_simple()

    def _format_apa(self) -> str:
        """Format in APA style"""
        page = f", p. {self.page_number}" if self.page_number else ""
        return f"{self.source}{page}"

    def _format_mla(self) -> str:
        """Format in MLA style"""
        page = f" {self.page_number}" if self.page_number else ""
        return f"({self.source}{page})"

    def _format_chicago(self) -> str:
        """Format in Chicago style"""
        page = f", {self.page_number}" if self.page_number else ""
        return f"{self.source}{page}"

    def _format_simple(self) -> str:
        """Simple format"""
        page = f":p{self.page_number}" if self.page_number else ""
        conf = f" ({self.confidence:.2f})" if self.confidence < 1.0 else ""
        return f"[{self.source}{page}{conf}]"


class CitationGenerator:
    """Generate citations for RAG responses"""

    def __init__(
        self,
        min_confidence: float = 0.5,
        max_citations_per_sentence: int = 3,
    ):
        """
        Initialize citation generator

        Args:
            min_confidence: Minimum confidence for citation
            max_citations_per_sentence: Max citations per sentence
        """
        self.min_confidence = min_confidence
        self.max_citations_per_sentence = max_citations_per_sentence

    def generate_citations(
        self,
        answer: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
    ) -> Tuple[str, List[Citation]]:
        """
        Generate citations for answer

        Args:
            answer: Generated answer
            chunks_with_scores: Source chunks with relevance scores

        Returns:
            Tuple of (cited_answer, citations_list)
        """
        if not chunks_with_scores:
            return answer, []

        # Split answer into sentences
        sentences = self._split_sentences(answer)

        # Generate citations for each sentence
        all_citations = []
        cited_sentences = []

        current_pos = 0
        for sentence in sentences:
            # Find best citations for this sentence
            citations = self._find_citations_for_sentence(
                sentence,
                chunks_with_scores
            )

            # Format sentence with citations
            if citations:
                citation_markers = self._format_citation_markers(citations)
                cited_sentence = f"{sentence}{citation_markers}"
                all_citations.extend(citations)
            else:
                cited_sentence = sentence

            cited_sentences.append(cited_sentence)
            current_pos += len(sentence)

        # Combine into cited answer
        cited_answer = ' '.join(cited_sentences)

        # Deduplicate citations
        unique_citations = self._deduplicate_citations(all_citations)

        logger.info(f"Generated {len(unique_citations)} citations for answer")

        return cited_answer, unique_citations

    def _find_citations_for_sentence(
        self,
        sentence: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
    ) -> List[Citation]:
        """Find best citations for a sentence"""
        from sentence_transformers import SentenceTransformer

        # Load model for similarity
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Encode sentence
        sentence_emb = model.encode([sentence])[0]

        # Score each chunk
        scored_chunks = []
        for chunk, retrieval_score in chunks_with_scores:
            # Check if chunk text appears in sentence (exact match)
            overlap_score = self._calculate_overlap(sentence, chunk.text)

            # Semantic similarity
            chunk_emb = model.encode([chunk.text])[0]
            import numpy as np
            semantic_score = float(np.dot(sentence_emb, chunk_emb) / (
                np.linalg.norm(sentence_emb) * np.linalg.norm(chunk_emb)
            ))

            # Combined score
            combined_score = 0.3 * overlap_score + 0.4 * semantic_score + 0.3 * retrieval_score

            if combined_score >= self.min_confidence:
                scored_chunks.append((chunk, combined_score))

        # Sort by score and take top citations
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored_chunks[:self.max_citations_per_sentence]

        # Create citations
        citations = []
        for chunk, score in top_chunks:
            citation = Citation(
                text=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                source=chunk.metadata.get('source', chunk.doc_id),
                page_number=chunk.metadata.get('page_number'),
                chunk_id=chunk.chunk_id,
                confidence=score,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=chunk.metadata,
            )
            citations.append(citation)

        return citations

    @staticmethod
    def _calculate_overlap(text1: str, text2: str) -> float:
        """Calculate word overlap between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _format_citation_markers(self, citations: List[Citation]) -> str:
        """Format citation markers"""
        if not citations:
            return ""

        # Create numbered citations
        markers = []
        for i, citation in enumerate(citations, 1):
            markers.append(f"[{i}]")

        return " " + ",".join(markers)

    @staticmethod
    def _deduplicate_citations(citations: List[Citation]) -> List[Citation]:
        """Remove duplicate citations"""
        seen = set()
        unique = []

        for citation in citations:
            key = (citation.source, citation.chunk_id)
            if key not in seen:
                seen.add(key)
                unique.append(citation)

        return unique

    def format_bibliography(
        self,
        citations: List[Citation],
        style: str = "simple"
    ) -> str:
        """Format bibliography from citations"""
        if not citations:
            return ""

        lines = ["References:", ""]

        for i, citation in enumerate(citations, 1):
            formatted = citation.format_citation(style)
            lines.append(f"{i}. {formatted}")

        return "\n".join(lines)
