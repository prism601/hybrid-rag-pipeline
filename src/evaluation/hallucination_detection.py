"""
Hallucination detection and faithfulness scoring
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HallucinationScore:
    """Results of hallucination detection"""
    is_hallucination: bool
    confidence: float
    faithfulness_score: float
    details: Dict[str, Any]


class FaithfulnessChecker:
    """Check if generated answer is faithful to source context"""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.7,
    ):
        """
        Initialize faithfulness checker

        Args:
            model_name: NLI model for entailment checking
            threshold: Threshold for faithfulness (0-1)
        """
        self.model = CrossEncoder(model_name)
        self.threshold = threshold
        logger.info(f"Initialized faithfulness checker with model: {model_name}")

    def check_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> HallucinationScore:
        """
        Check if answer is faithful to contexts

        Args:
            answer: Generated answer
            contexts: Source contexts

        Returns:
            HallucinationScore with faithfulness assessment
        """
        if not contexts:
            return HallucinationScore(
                is_hallucination=True,
                confidence=1.0,
                faithfulness_score=0.0,
                details={"reason": "No context provided"}
            )

        # Split answer into claims
        claims = self._extract_claims(answer)

        if not claims:
            return HallucinationScore(
                is_hallucination=False,
                confidence=0.5,
                faithfulness_score=0.5,
                details={"reason": "No claims extracted"}
            )

        # Check each claim against contexts
        claim_scores = []
        unsupported_claims = []

        for claim in claims:
            max_entailment = 0.0
            best_context = None

            for context in contexts:
                # Check entailment
                score = self._check_entailment(context, claim)

                if score > max_entailment:
                    max_entailment = score
                    best_context = context

            claim_scores.append(max_entailment)

            if max_entailment < self.threshold:
                unsupported_claims.append({
                    "claim": claim,
                    "max_support": max_entailment,
                    "best_context": best_context[:100] if best_context else None
                })

        # Calculate overall faithfulness
        faithfulness_score = float(np.mean(claim_scores))
        is_hallucination = faithfulness_score < self.threshold

        return HallucinationScore(
            is_hallucination=is_hallucination,
            confidence=abs(faithfulness_score - self.threshold),
            faithfulness_score=faithfulness_score,
            details={
                "num_claims": len(claims),
                "num_unsupported": len(unsupported_claims),
                "unsupported_claims": unsupported_claims,
                "claim_scores": claim_scores,
            }
        )

    def _check_entailment(self, context: str, claim: str) -> float:
        """Check if context entails claim"""
        pair = (context, claim)
        score = self.model.predict([pair])[0]

        # Convert to 0-1 probability
        # For NLI models: typically outputs logits or probabilities
        # We assume higher score = more entailment
        return float(1 / (1 + np.exp(-score)))  # Sigmoid

    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from text"""
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims


class SemanticConsistencyChecker:
    """Check semantic consistency between answer and context"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.5,
    ):
        """
        Initialize semantic consistency checker

        Args:
            model_name: Sentence transformer model
            threshold: Similarity threshold
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def check_consistency(
        self,
        answer: str,
        contexts: List[str],
    ) -> HallucinationScore:
        """Check semantic consistency between answer and contexts"""
        if not contexts:
            return HallucinationScore(
                is_hallucination=True,
                confidence=1.0,
                faithfulness_score=0.0,
                details={"reason": "No context provided"}
            )

        # Encode answer and contexts
        answer_emb = self.model.encode([answer])[0]
        context_embs = self.model.encode(contexts)

        # Calculate similarities
        similarities = []
        for ctx_emb in context_embs:
            sim = self._cosine_similarity(answer_emb, ctx_emb)
            similarities.append(sim)

        # Get max similarity
        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))

        is_hallucination = max_similarity < self.threshold

        return HallucinationScore(
            is_hallucination=is_hallucination,
            confidence=abs(max_similarity - self.threshold),
            faithfulness_score=max_similarity,
            details={
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "similarities": [float(s) for s in similarities],
            }
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class FactChecker:
    """Check factual consistency using token-level alignment"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def check_facts(
        self,
        answer: str,
        contexts: List[str],
    ) -> HallucinationScore:
        """Check factual consistency"""
        if not contexts:
            return HallucinationScore(
                is_hallucination=True,
                confidence=1.0,
                faithfulness_score=0.0,
                details={"reason": "No context provided"}
            )

        # Extract key facts (numbers, dates, names)
        answer_facts = self._extract_facts(answer)
        context_facts = set()

        for context in contexts:
            context_facts.update(self._extract_facts(context))

        # Check how many answer facts are in context
        if not answer_facts:
            return HallucinationScore(
                is_hallucination=False,
                confidence=0.5,
                faithfulness_score=0.5,
                details={"reason": "No facts extracted"}
            )

        supported_facts = answer_facts & context_facts
        unsupported_facts = answer_facts - context_facts

        support_ratio = len(supported_facts) / len(answer_facts)
        is_hallucination = support_ratio < self.threshold

        return HallucinationScore(
            is_hallucination=is_hallucination,
            confidence=abs(support_ratio - self.threshold),
            faithfulness_score=support_ratio,
            details={
                "answer_facts": list(answer_facts),
                "supported_facts": list(supported_facts),
                "unsupported_facts": list(unsupported_facts),
                "support_ratio": support_ratio,
            }
        )

    @staticmethod
    def _extract_facts(text: str) -> set:
        """Extract factual elements (numbers, dates, proper nouns)"""
        facts = set()

        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        facts.update(numbers)

        # Extract dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        facts.update(dates)

        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        facts.update(years)

        # Extract capitalized words (proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        facts.update(proper_nouns)

        return facts


class HallucinationDetector:
    """Ensemble hallucination detector"""

    def __init__(
        self,
        use_faithfulness: bool = True,
        use_semantic: bool = True,
        use_facts: bool = True,
        threshold: float = 0.7,
    ):
        """
        Initialize ensemble detector

        Args:
            use_faithfulness: Use NLI-based faithfulness checker
            use_semantic: Use semantic consistency checker
            use_facts: Use fact checker
            threshold: Overall threshold for hallucination
        """
        self.use_faithfulness = use_faithfulness
        self.use_semantic = use_semantic
        self.use_facts = use_facts
        self.threshold = threshold

        self.checkers = []

        if use_faithfulness:
            self.checkers.append(("faithfulness", FaithfulnessChecker()))

        if use_semantic:
            self.checkers.append(("semantic", SemanticConsistencyChecker()))

        if use_facts:
            self.checkers.append(("facts", FactChecker()))

        logger.info(f"Initialized hallucination detector with {len(self.checkers)} checkers")

    def detect(
        self,
        answer: str,
        contexts: List[str],
    ) -> HallucinationScore:
        """Detect hallucinations using ensemble of methods"""
        if not self.checkers:
            return HallucinationScore(
                is_hallucination=False,
                confidence=0.0,
                faithfulness_score=0.5,
                details={"reason": "No checkers enabled"}
            )

        # Run all checkers
        results = {}
        scores = []

        for name, checker in self.checkers:
            if name == "faithfulness":
                result = checker.check_faithfulness(answer, contexts)
            elif name == "semantic":
                result = checker.check_consistency(answer, contexts)
            elif name == "facts":
                result = checker.check_facts(answer, contexts)
            else:
                continue

            results[name] = result
            scores.append(result.faithfulness_score)

        # Aggregate scores
        avg_score = float(np.mean(scores))
        is_hallucination = avg_score < self.threshold

        return HallucinationScore(
            is_hallucination=is_hallucination,
            confidence=abs(avg_score - self.threshold),
            faithfulness_score=avg_score,
            details={
                "individual_results": {
                    name: {
                        "is_hallucination": result.is_hallucination,
                        "score": result.faithfulness_score,
                        "details": result.details
                    }
                    for name, result in results.items()
                },
                "num_checkers": len(self.checkers),
                "scores": scores,
            }
        )
