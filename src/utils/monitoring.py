"""
Monitoring and metrics collection
"""
from typing import Dict, Any
import time
import logging
from functools import wraps
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and export metrics"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PROMETHEUS_AVAILABLE

        if self.enabled:
            # Define metrics
            self.query_counter = Counter(
                'rag_queries_total',
                'Total number of queries',
                ['status']
            )

            self.query_latency = Histogram(
                'rag_query_latency_seconds',
                'Query latency in seconds',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            )

            self.retrieval_counter = Counter(
                'rag_retrievals_total',
                'Total number of retrievals',
                ['method']
            )

            self.chunks_retrieved = Histogram(
                'rag_chunks_retrieved',
                'Number of chunks retrieved per query',
                buckets=[1, 5, 10, 20, 50, 100]
            )

            self.hallucination_detected = Counter(
                'rag_hallucinations_detected_total',
                'Total number of detected hallucinations'
            )

            self.indexed_chunks = Gauge(
                'rag_indexed_chunks_total',
                'Total number of indexed chunks'
            )

            logger.info("Prometheus metrics initialized")
        else:
            logger.warning("Prometheus metrics disabled")

    def record_query(self, status: str = "success"):
        """Record a query"""
        if self.enabled:
            self.query_counter.labels(status=status).inc()

    def record_query_latency(self, latency_seconds: float):
        """Record query latency"""
        if self.enabled:
            self.query_latency.observe(latency_seconds)

    def record_retrieval(self, method: str):
        """Record a retrieval"""
        if self.enabled:
            self.retrieval_counter.labels(method=method).inc()

    def record_chunks_retrieved(self, count: int):
        """Record number of chunks retrieved"""
        if self.enabled:
            self.chunks_retrieved.observe(count)

    def record_hallucination(self):
        """Record hallucination detection"""
        if self.enabled:
            self.hallucination_detected.inc()

    def set_indexed_chunks(self, count: int):
        """Set total indexed chunks"""
        if self.enabled:
            self.indexed_chunks.set(count)

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        if self.enabled:
            return generate_latest(REGISTRY)
        return b""


# Global metrics collector
metrics_collector = MetricsCollector()


def track_latency(metric_name: str = "operation"):
    """Decorator to track operation latency"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                metrics_collector.record_query("success")
                return result
            except Exception as e:
                metrics_collector.record_query("error")
                raise
            finally:
                latency = time.time() - start_time
                metrics_collector.record_query_latency(latency)
                logger.info(f"{metric_name} latency: {latency:.3f}s")

        return wrapper
    return decorator


@contextmanager
def measure_time(operation_name: str):
    """Context manager to measure operation time"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{operation_name} took {elapsed:.3f}s")
