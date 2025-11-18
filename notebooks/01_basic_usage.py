"""
Example: Basic RAG Pipeline Usage

This notebook demonstrates basic usage of the hybrid RAG pipeline.
"""

# %% Setup
from src.core.rag_pipeline import RAGPipeline
from src.core.document_loader import DocumentLoaderFactory
import logging

logging.basicConfig(level=logging.INFO)

# %% Initialize Pipeline
print("Initializing RAG Pipeline...")
pipeline = RAGPipeline(
    use_vector_store=True,
    use_bm25=True,
    use_reranking=True,
    use_hallucination_detection=True,
    use_citations=True,
)

# %% Load Sample Documents
# Create sample markdown document
sample_doc = """
# Artificial Intelligence

Artificial Intelligence (AI) is the simulation of human intelligence by machines.

## Machine Learning
Machine learning is a subset of AI that enables systems to learn from data.

### Deep Learning
Deep learning uses neural networks with multiple layers.

## Applications
- Natural Language Processing
- Computer Vision
- Robotics
- Autonomous Vehicles
"""

import tempfile
from pathlib import Path

with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
    f.write(sample_doc)
    doc_path = f.name

# %% Index Documents
print(f"\nIndexing document: {doc_path}")
stats = pipeline.index_documents([doc_path])

print(f"✓ Indexed {stats['num_documents']} documents")
print(f"✓ Created {stats['num_chunks']} chunks")
print(f"✓ Average chunk size: {stats['avg_chunk_size']:.0f} characters")

# %% Query the Pipeline
print("\nQuerying the pipeline...")
response = pipeline.query(
    question="What is machine learning?",
    top_k=3,
    use_reranking=True,
)

print(f"\nAnswer:\n{response.answer}\n")
print(f"Retrieved {len(response.contexts)} contexts")
print(f"Relevance scores: {[f'{s:.3f}' for s in response.scores]}")

# %% Check Hallucination Score
if response.hallucination_score:
    print(f"\nHallucination Detection:")
    print(f"  Faithfulness Score: {response.hallucination_score.faithfulness_score:.3f}")
    print(f"  Is Hallucination: {response.hallucination_score.is_hallucination}")

# %% View Citations
if response.citations:
    print(f"\nCitations ({len(response.citations)}):")
    for i, citation in enumerate(response.citations, 1):
        print(f"  [{i}] {citation.source} (confidence: {citation.confidence:.2f})")

# %% RAGAS Evaluation
if response.ragas_score:
    print(f"\nRAGAS Metrics:")
    print(f"  Faithfulness: {response.ragas_score.faithfulness:.3f}")
    print(f"  Answer Relevance: {response.ragas_score.answer_relevance:.3f}")
    print(f"  Context Precision: {response.ragas_score.context_precision:.3f}")
    print(f"  Context Recall: {response.ragas_score.context_recall:.3f}")
    print(f"  Overall Score: {response.ragas_score.overall_score:.3f}")

# %% Batch Queries
print("\nBatch querying...")
questions = [
    "What is deep learning?",
    "What are AI applications?",
    "Define artificial intelligence",
]

responses = pipeline.batch_query(questions, top_k=2)

for q, r in zip(questions, responses):
    print(f"\nQ: {q}")
    print(f"A: {r.answer[:100]}...")

# %% Cleanup
Path(doc_path).unlink()
print("\nDone!")
