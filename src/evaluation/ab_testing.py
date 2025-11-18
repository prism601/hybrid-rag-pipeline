"""
A/B testing framework for comparing retrieval strategies
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import time
from collections import defaultdict
import random

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class Variant(Enum):
    """A/B test variants"""
    A = "A"
    B = "B"


@dataclass
class ExperimentResult:
    """Result from a single query"""
    query: str
    variant: Variant
    latency_ms: float
    num_results: int
    relevance_score: Optional[float]
    user_feedback: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class ExperimentStats:
    """Statistics for an A/B test"""
    variant: Variant
    num_samples: int
    mean_latency: float
    mean_relevance: float
    mean_user_feedback: float
    p_value: float
    is_significant: bool
    improvement: float
    details: Dict[str, Any]


class ABTest:
    """A/B test for comparing retrieval strategies"""

    def __init__(
        self,
        name: str,
        variant_a_fn: Callable,
        variant_b_fn: Callable,
        traffic_split: float = 0.5,
        significance_level: float = 0.05,
    ):
        """
        Initialize A/B test

        Args:
            name: Test name
            variant_a_fn: Function for variant A
            variant_b_fn: Function for variant B
            traffic_split: Fraction of traffic for variant A (0-1)
            significance_level: Statistical significance threshold
        """
        self.name = name
        self.variant_a_fn = variant_a_fn
        self.variant_b_fn = variant_b_fn
        self.traffic_split = traffic_split
        self.significance_level = significance_level

        self.results: List[ExperimentResult] = []
        self.variant_results = {
            Variant.A: [],
            Variant.B: [],
        }

        logger.info(f"Initialized A/B test: {name}")

    def run_query(
        self,
        query: str,
        force_variant: Optional[Variant] = None,
        **kwargs
    ) -> tuple[Any, ExperimentResult]:
        """
        Run query with randomly assigned variant

        Args:
            query: Search query
            force_variant: Force specific variant (for testing)
            **kwargs: Additional arguments for retrieval function

        Returns:
            Tuple of (retrieval_result, experiment_result)
        """
        # Assign variant
        if force_variant:
            variant = force_variant
        else:
            variant = Variant.A if random.random() < self.traffic_split else Variant.B

        # Select function
        fn = self.variant_a_fn if variant == Variant.A else self.variant_b_fn

        # Execute and measure
        start_time = time.time()
        result = fn(query, **kwargs)
        latency_ms = (time.time() - start_time) * 1000

        # Create experiment result
        experiment_result = ExperimentResult(
            query=query,
            variant=variant,
            latency_ms=latency_ms,
            num_results=len(result) if isinstance(result, list) else 1,
            relevance_score=None,  # To be filled by evaluation
            user_feedback=None,  # To be filled by user
            metadata={}
        )

        # Store result
        self.results.append(experiment_result)
        self.variant_results[variant].append(experiment_result)

        return result, experiment_result

    def add_feedback(
        self,
        result_index: int,
        relevance_score: Optional[float] = None,
        user_feedback: Optional[float] = None,
    ) -> None:
        """Add evaluation feedback to result"""
        if 0 <= result_index < len(self.results):
            if relevance_score is not None:
                self.results[result_index].relevance_score = relevance_score
            if user_feedback is not None:
                self.results[result_index].user_feedback = user_feedback

    def analyze(self) -> Dict[Variant, ExperimentStats]:
        """Analyze A/B test results"""
        stats_dict = {}

        for variant in [Variant.A, Variant.B]:
            results = self.variant_results[variant]

            if not results:
                continue

            # Calculate metrics
            latencies = [r.latency_ms for r in results]
            relevances = [r.relevance_score for r in results if r.relevance_score is not None]
            feedbacks = [r.user_feedback for r in results if r.user_feedback is not None]

            mean_latency = np.mean(latencies) if latencies else 0.0
            mean_relevance = np.mean(relevances) if relevances else 0.0
            mean_feedback = np.mean(feedbacks) if feedbacks else 0.0

            stats_dict[variant] = {
                'num_samples': len(results),
                'mean_latency': mean_latency,
                'mean_relevance': mean_relevance,
                'mean_feedback': mean_feedback,
                'latencies': latencies,
                'relevances': relevances,
                'feedbacks': feedbacks,
            }

        # Statistical comparison
        if Variant.A in stats_dict and Variant.B in stats_dict:
            # Compare relevance scores (primary metric)
            a_rel = stats_dict[Variant.A]['relevances']
            b_rel = stats_dict[Variant.B]['relevances']

            if a_rel and b_rel:
                t_stat, p_value = stats.ttest_ind(a_rel, b_rel)
                is_significant = p_value < self.significance_level
                improvement = (np.mean(b_rel) - np.mean(a_rel)) / np.mean(a_rel) * 100
            else:
                p_value = 1.0
                is_significant = False
                improvement = 0.0

            # Create stats objects
            results_dict = {}
            for variant in [Variant.A, Variant.B]:
                results_dict[variant] = ExperimentStats(
                    variant=variant,
                    num_samples=stats_dict[variant]['num_samples'],
                    mean_latency=stats_dict[variant]['mean_latency'],
                    mean_relevance=stats_dict[variant]['mean_relevance'],
                    mean_user_feedback=stats_dict[variant]['mean_feedback'],
                    p_value=p_value,
                    is_significant=is_significant,
                    improvement=improvement if variant == Variant.B else 0.0,
                    details=stats_dict[variant]
                )

            return results_dict

        return {}

    def get_winner(self) -> Optional[Variant]:
        """Determine winning variant"""
        stats = self.analyze()

        if Variant.A not in stats or Variant.B not in stats:
            return None

        if not stats[Variant.A].is_significant:
            return None  # No significant difference

        # Winner is the one with higher relevance
        if stats[Variant.B].mean_relevance > stats[Variant.A].mean_relevance:
            return Variant.B
        else:
            return Variant.A

    def print_report(self) -> str:
        """Generate text report of results"""
        stats = self.analyze()

        if not stats:
            return "Insufficient data for analysis"

        report_lines = [
            f"A/B Test Report: {self.name}",
            "=" * 50,
            ""
        ]

        for variant in [Variant.A, Variant.B]:
            if variant not in stats:
                continue

            s = stats[variant]
            report_lines.extend([
                f"Variant {variant.value}:",
                f"  Samples: {s.num_samples}",
                f"  Mean Latency: {s.mean_latency:.2f} ms",
                f"  Mean Relevance: {s.mean_relevance:.4f}",
                f"  Mean User Feedback: {s.mean_user_feedback:.4f}",
                ""
            ])

        if Variant.A in stats and Variant.B in stats:
            report_lines.extend([
                f"Statistical Comparison:",
                f"  P-value: {stats[Variant.B].p_value:.4f}",
                f"  Significant: {stats[Variant.B].is_significant}",
                f"  Improvement: {stats[Variant.B].improvement:.2f}%",
                ""
            ])

            winner = self.get_winner()
            if winner:
                report_lines.append(f"Winner: Variant {winner.value}")
            else:
                report_lines.append("Winner: No significant difference")

        return "\n".join(report_lines)


class ExperimentManager:
    """Manage multiple A/B tests"""

    def __init__(self):
        self.experiments: Dict[str, ABTest] = {}

    def create_experiment(
        self,
        name: str,
        variant_a_fn: Callable,
        variant_b_fn: Callable,
        **kwargs
    ) -> ABTest:
        """Create new A/B test"""
        experiment = ABTest(name, variant_a_fn, variant_b_fn, **kwargs)
        self.experiments[name] = experiment
        return experiment

    def get_experiment(self, name: str) -> Optional[ABTest]:
        """Get experiment by name"""
        return self.experiments.get(name)

    def list_experiments(self) -> List[str]:
        """List all experiment names"""
        return list(self.experiments.keys())

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        summary = {}

        for name, exp in self.experiments.items():
            stats = exp.analyze()
            winner = exp.get_winner()

            summary[name] = {
                'num_queries': len(exp.results),
                'winner': winner.value if winner else None,
                'stats': {
                    v.value: {
                        'num_samples': s.num_samples,
                        'mean_relevance': s.mean_relevance,
                    }
                    for v, s in stats.items()
                }
            }

        return summary
