"""
RAGAS (Retrieval Augmented Generation Assessment) metrics
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RAGASScore:
    """RAGAS evaluation scores"""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    details: Dict[str, Any]


class FaithfulnessMetric:
    """Measures factual consistency of answer with context"""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.model = CrossEncoder(model_name)

    def compute(self, answer: str, contexts: List[str]) -> float:
        """
        Compute faithfulness score

        Args:
            answer: Generated answer
            contexts: Retrieved contexts

        Returns:
            Faithfulness score (0-1)
        """
        if not contexts or not answer:
            return 0.0

        # Extract statements from answer
        statements = self._extract_statements(answer)

        if not statements:
            return 0.5

        # Check each statement against contexts
        scores = []
        for statement in statements:
            max_score = 0.0
            for context in contexts:
                score = self._check_entailment(context, statement)
                max_score = max(max_score, score)
            scores.append(max_score)

        return float(np.mean(scores))

    def _check_entailment(self, premise: str, hypothesis: str) -> float:
        """Check if premise entails hypothesis"""
        score = self.model.predict([(premise, hypothesis)])[0]
        return float(1 / (1 + np.exp(-score)))

    @staticmethod
    def _extract_statements(text: str) -> List[str]:
        """Extract statements from text"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]


class AnswerRelevanceMetric:
    """Measures how well answer addresses the question"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compute(self, question: str, answer: str) -> float:
        """
        Compute answer relevance score

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        if not question or not answer:
            return 0.0

        # Encode question and answer
        question_emb = self.model.encode([question])[0]
        answer_emb = self.model.encode([answer])[0]

        # Calculate cosine similarity
        similarity = np.dot(question_emb, answer_emb) / (
            np.linalg.norm(question_emb) * np.linalg.norm(answer_emb)
        )

        return float(max(0.0, similarity))


class ContextPrecisionMetric:
    """Measures precision of retrieved contexts"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def compute(
        self,
        question: str,
        contexts: List[str],
        answer: Optional[str] = None
    ) -> float:
        """
        Compute context precision

        Args:
            question: User question
            contexts: Retrieved contexts
            answer: Generated answer (optional)

        Returns:
            Precision score (0-1)
        """
        if not contexts or not question:
            return 0.0

        # Score each context's relevance to question
        scores = []
        for context in contexts:
            score = self.model.predict([(question, context)])[0]
            scores.append(score)

        # Calculate precision at k
        # Higher weight for higher-ranked contexts
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_scores = [s * w for s, w in zip(scores, weights)]

        if not weighted_scores:
            return 0.0

        return float(sum(weighted_scores) / sum(weights))


class ContextRecallMetric:
    """Measures recall of retrieved contexts"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def compute(
        self,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Compute context recall

        Args:
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Ground truth answer (if available)

        Returns:
            Recall score (0-1)
        """
        if not contexts or not answer:
            return 0.0

        # Use ground truth if available, otherwise use answer
        reference = ground_truth if ground_truth else answer

        # Extract statements from reference
        statements = self._extract_statements(reference)

        if not statements:
            return 0.5

        # Check how many statements are covered by contexts
        covered = 0
        for statement in statements:
            if self._is_covered(statement, contexts):
                covered += 1

        return float(covered / len(statements))

    def _is_covered(self, statement: str, contexts: List[str]) -> bool:
        """Check if statement is covered by any context"""
        statement_emb = self.model.encode([statement])[0]

        for context in contexts:
            # Split context into sentences
            sentences = self._extract_statements(context)

            for sent in sentences:
                sent_emb = self.model.encode([sent])[0]
                similarity = np.dot(statement_emb, sent_emb) / (
                    np.linalg.norm(statement_emb) * np.linalg.norm(sent_emb)
                )

                if similarity > 0.7:  # Threshold for coverage
                    return True

        return False

    @staticmethod
    def _extract_statements(text: str) -> List[str]:
        """Extract statements from text"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]


class RAGASEvaluator:
    """RAGAS evaluation framework"""

    def __init__(self):
        """Initialize RAGAS evaluator with all metrics"""
        self.faithfulness_metric = FaithfulnessMetric()
        self.answer_relevance_metric = AnswerRelevanceMetric()
        self.context_precision_metric = ContextPrecisionMetric()
        self.context_recall_metric = ContextRecallMetric()

        logger.info("Initialized RAGAS evaluator")

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> RAGASScore:
        """
        Evaluate RAG pipeline using RAGAS metrics

        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Ground truth answer (optional)

        Returns:
            RAGASScore with all metrics
        """
        # Compute individual metrics
        faithfulness = self.faithfulness_metric.compute(answer, contexts)
        answer_relevance = self.answer_relevance_metric.compute(question, answer)
        context_precision = self.context_precision_metric.compute(question, contexts, answer)
        context_recall = self.context_recall_metric.compute(answer, contexts, ground_truth)

        # Compute overall score (harmonic mean)
        scores = [faithfulness, answer_relevance, context_precision, context_recall]
        overall_score = self._harmonic_mean(scores)

        return RAGASScore(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            overall_score=overall_score,
            details={
                "question": question,
                "answer_length": len(answer),
                "num_contexts": len(contexts),
                "has_ground_truth": ground_truth is not None,
            }
        )

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> List[RAGASScore]:
        """Evaluate multiple examples"""
        if ground_truths is None:
            ground_truths = [None] * len(questions)

        results = []
        for q, a, c, gt in zip(questions, answers, contexts_list, ground_truths):
            score = self.evaluate(q, a, c, gt)
            results.append(score)

        return results

    def aggregate_scores(self, scores: List[RAGASScore]) -> Dict[str, float]:
        """Aggregate scores across multiple examples"""
        if not scores:
            return {}

        return {
            "mean_faithfulness": float(np.mean([s.faithfulness for s in scores])),
            "mean_answer_relevance": float(np.mean([s.answer_relevance for s in scores])),
            "mean_context_precision": float(np.mean([s.context_precision for s in scores])),
            "mean_context_recall": float(np.mean([s.context_recall for s in scores])),
            "mean_overall_score": float(np.mean([s.overall_score for s in scores])),
            "std_overall_score": float(np.std([s.overall_score for s in scores])),
        }

    @staticmethod
    def _harmonic_mean(values: List[float]) -> float:
        """Calculate harmonic mean"""
        values = [v for v in values if v > 0]
        if not values:
            return 0.0
        return float(len(values) / sum(1.0 / v for v in values))
