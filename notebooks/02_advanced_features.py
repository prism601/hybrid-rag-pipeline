"""
Example: Advanced RAG Features

Demonstrates chunking strategies, knowledge graphs, and A/B testing.
"""

# %% Chunking Strategies Comparison
from src.core.chunking import ChunkerFactory, ChunkingStrategy
from src.core.document_loader import Document

sample_text = """
Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns and make decisions with minimal human intervention.

Deep learning is part of a broader family of machine learning methods based on artificial
neural networks with representation learning. Learning can be supervised, semi-supervised
or unsupervised.
"""

document = Document(
    content=sample_text,
    metadata={"source": "example"},
    source="example.txt",
    doc_type="text"
)

# Compare different chunking strategies
strategies = [
    ChunkingStrategy.FIXED_SIZE,
    ChunkingStrategy.RECURSIVE,
    ChunkingStrategy.SENTENCE,
]

for strategy in strategies:
    chunker = ChunkerFactory.create_chunker(
        strategy=strategy,
        chunk_size=100,
        overlap=20
    )
    chunks = chunker.chunk([document])
    print(f"\n{strategy.value}:")
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2], 1):
        print(f"  Chunk {i}: {chunk.text[:50]}...")

# %% Knowledge Graph Extraction
from src.retrieval.knowledge_graph import KnowledgeGraphExtractor

extractor = KnowledgeGraphExtractor(use_spacy=False)  # Fallback to patterns
kg = extractor.extract_from_chunks(chunks)

print(f"\nKnowledge Graph:")
print(f"  Entities: {len(kg.entities)}")
print(f"  Relations: {len(kg.relations)}")

# Show sample entities
for entity_id, entity in list(kg.entities.items())[:5]:
    print(f"  - {entity.text} ({entity.entity_type})")

# %% A/B Testing Framework
from src.evaluation.ab_testing import ABTest
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.sparse_retrieval import BM25Retriever

# Create test retrievers
vector_store = ChromaVectorStore(collection_name="ab_test")
bm25 = BM25Retriever()

# Index chunks
vector_store.add_chunks(chunks)
bm25.add_chunks(chunks)

# Define variants
def variant_dense(query, **kwargs):
    return vector_store.search(query, **kwargs)

def variant_sparse(query, **kwargs):
    return bm25.search(query, **kwargs)

# Create A/B test
ab_test = ABTest(
    name="dense_vs_sparse",
    variant_a_fn=variant_dense,
    variant_b_fn=variant_sparse,
    traffic_split=0.5
)

# Run test queries
test_queries = [
    "What is machine learning?",
    "Explain deep learning",
    "How does AI work?",
]

for query in test_queries:
    result, exp_result = ab_test.run_query(query, top_k=3)
    print(f"Query: {query}")
    print(f"  Variant: {exp_result.variant.value}")
    print(f"  Results: {exp_result.num_results}")
    print(f"  Latency: {exp_result.latency_ms:.2f}ms")

# Analyze results
print("\n" + ab_test.print_report())

# %% Hallucination Detection
from src.evaluation.hallucination_detection import HallucinationDetector

detector = HallucinationDetector(
    use_faithfulness=True,
    use_semantic=True,
    use_facts=True,
)

# Test cases
contexts = [
    "Machine learning is a subset of AI that enables computers to learn from data."
]

# Faithful answer
faithful_answer = "Machine learning is a type of AI that allows computers to learn from data."
result = detector.detect(faithful_answer, contexts)
print(f"\nFaithful Answer:")
print(f"  Hallucination: {result.is_hallucination}")
print(f"  Score: {result.faithfulness_score:.3f}")

# Hallucinated answer
hallucinated_answer = "Machine learning was invented in 1950 by Alan Turing in London."
result = detector.detect(hallucinated_answer, contexts)
print(f"\nHallucinated Answer:")
print(f"  Hallucination: {result.is_hallucination}")
print(f"  Score: {result.faithfulness_score:.3f}")
print(f"  Details: {result.details}")

print("\nDone!")
