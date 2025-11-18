#!/usr/bin/env python3
"""
Main entry point for the Hybrid RAG Pipeline

Examples:
    # Run API server
    python main.py serve

    # Index documents
    python main.py index data/documents/*.pdf

    # Query
    python main.py query "What is machine learning?"

    # Run example
    python main.py example
"""

import sys
import argparse
import logging
from pathlib import Path

from src.core.rag_pipeline import RAGPipeline
from src.core.config import settings
from src.utils.logger import setup_logging

# Setup logging
setup_logging(log_level=settings.log_level)
logger = logging.getLogger(__name__)


def serve():
    """Start API server"""
    import uvicorn
    from src.api.main import app

    logger.info(f"Starting API server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )


def index_documents(file_paths: list[str], optimize: bool = False):
    """Index documents"""
    logger.info(f"Indexing {len(file_paths)} documents...")

    pipeline = RAGPipeline()
    stats = pipeline.index_documents(file_paths, optimize_chunks=optimize)

    logger.info(f"✓ Indexed {stats['num_documents']} documents")
    logger.info(f"✓ Created {stats['num_chunks']} chunks")
    logger.info(f"✓ Average chunk size: {stats['avg_chunk_size']:.0f} chars")


def query(question: str, top_k: int = 5):
    """Query the pipeline"""
    logger.info(f"Query: {question}")

    pipeline = RAGPipeline()

    # Check if we have indexed documents
    if not pipeline.chunks:
        logger.error("No documents indexed! Run 'python main.py index <files>' first")
        return

    response = pipeline.query(question, top_k=top_k)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{response.answer}\n")
    print(f"Sources: {len(response.citations)} citations")

    if response.hallucination_score:
        print(f"Faithfulness: {response.hallucination_score.faithfulness_score:.3f}")


def run_example():
    """Run example workflow"""
    logger.info("Running example workflow...")

    # Create sample document
    sample_content = """
    # Machine Learning

    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.

    ## Deep Learning
    Deep learning is a type of machine learning based on neural networks.

    ## Applications
    - Natural Language Processing
    - Computer Vision
    - Robotics
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_content)
        temp_path = f.name

    try:
        # Initialize pipeline
        pipeline = RAGPipeline(
            use_vector_store=True,
            use_bm25=True,
            use_reranking=True,
        )

        # Index
        logger.info("Indexing sample document...")
        stats = pipeline.index_documents([temp_path])
        logger.info(f"Created {stats['num_chunks']} chunks")

        # Query
        logger.info("Querying...")
        response = pipeline.query("What is machine learning?", top_k=3)

        print("\n" + "=" * 60)
        print("EXAMPLE RESULTS")
        print("=" * 60)
        print(f"\nQuestion: What is machine learning?")
        print(f"\nAnswer:\n{response.answer}\n")
        print(f"Retrieved {len(response.contexts)} contexts")
        print(f"Citations: {len(response.citations)}")

        if response.ragas_score:
            print(f"\nRAGAS Overall Score: {response.ragas_score.overall_score:.3f}")

        print("\n" + "=" * 60)

    finally:
        Path(temp_path).unlink()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hybrid RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Serve command
    subparsers.add_parser('serve', help='Start API server')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('files', nargs='+', help='Files to index')
    index_parser.add_argument('--optimize', action='store_true', help='Optimize chunk size')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the pipeline')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of results')

    # Example command
    subparsers.add_parser('example', help='Run example workflow')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'serve':
            serve()
        elif args.command == 'index':
            index_documents(args.files, args.optimize)
        elif args.command == 'query':
            query(args.question, args.top_k)
        elif args.command == 'example':
            run_example()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
