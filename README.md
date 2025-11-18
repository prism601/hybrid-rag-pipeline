# Hybrid RAG Pipeline

Enterprise-grade Retrieval Augmented Generation system that combines multiple retrieval methods (dense, sparse, knowledge graphs) with intelligent re-ranking and hallucination detection.

## Features

### Core Capabilities
- **Multi-Modal Document Support**: PDF, HTML, Markdown, DOCX
- **Hybrid Retrieval**: Combines dense (vector), sparse (BM25), and knowledge graph retrieval
- **Intelligent Re-ranking**: Cross-encoder models for improved relevance
- **Hallucination Detection**: Multi-faceted faithfulness scoring
- **Citation Generation**: Automatic citation with confidence scores
- **RAGAS Evaluation**: Comprehensive metrics (faithfulness, answer relevance, context precision/recall)
- **A/B Testing Framework**: Compare retrieval strategies

### Production Features
- **Caching**: Redis-based caching for performance
- **Async Processing**: Batch operations and concurrent processing
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **API**: RESTful FastAPI with OpenAPI documentation
- **Dockerized**: Complete containerized deployment
- **Scalable**: Optimized for enterprise workloads

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Documents → Loader → Chunker → Embeddings                  │
│                           ↓                                   │
│              ┌────────────┼────────────┐                     │
│              ↓            ↓            ↓                     │
│         Vector DB      BM25      Knowledge Graph             │
│              ↓            ↓            ↓                     │
│              └────────────┼────────────┘                     │
│                    Hybrid Retrieval                          │
│                           ↓                                   │
│                  Cross-Encoder Reranking                     │
│                           ↓                                   │
│         ┌─────────────────┼─────────────────┐               │
│         ↓                 ↓                 ↓               │
│   Hallucination    Citation Gen      RAGAS Eval             │
│    Detection                                                  │
│         └─────────────────┼─────────────────┘               │
│                      Response                                 │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-rag-pipeline.git
cd hybrid-rag-pipeline

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for NER
python -m spacy download en_core_web_sm

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Docker Deployment (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Access services:
# - RAG API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
# - Neo4j Browser: http://localhost:7474 (neo4j/password)
```

### 3. Basic Usage

#### Python SDK

```python
from src.core.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    use_vector_store=True,
    use_bm25=True,
    use_reranking=True,
    use_hallucination_detection=True,
)

# Index documents
stats = pipeline.index_documents([
    "data/documents/doc1.pdf",
    "data/documents/doc2.md",
])
print(f"Indexed {stats['num_chunks']} chunks")

# Query
response = pipeline.query(
    question="What are the key findings?",
    top_k=5,
    use_reranking=True,
)

print(f"Answer: {response.answer}")
print(f"Sources: {len(response.citations)} citations")
print(f"Hallucination Score: {response.hallucination_score.faithfulness_score:.2f}")
```

#### REST API

```bash
# Index documents
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["data/documents/example.pdf"],
    "optimize_chunks": false
  }'

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "top_k": 5,
    "use_reranking": true
  }'

# Upload and index
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# LLM API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Vector Database
VECTOR_DB_TYPE=chromadb  # Options: chromadb, pinecone, qdrant

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval
TOP_K_RETRIEVAL=10
HYBRID_ALPHA=0.5  # 0=sparse only, 1=dense only
USE_MMR=true

# Hallucination Detection
HALLUCINATION_THRESHOLD=0.7
ENABLE_FAITHFULNESS_CHECK=true

# Performance
CACHE_TTL=3600
BATCH_SIZE=32
```

## Advanced Features

### Chunking Strategies

```python
from src.core.chunking import ChunkerFactory, ChunkingStrategy

# Recursive chunking (respects structure)
chunker = ChunkerFactory.create_chunker(
    strategy=ChunkingStrategy.RECURSIVE,
    chunk_size=512,
    overlap=50
)

# Semantic chunking (coherent meaning)
semantic_chunker = ChunkerFactory.create_chunker(
    strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=512,
    similarity_threshold=0.5
)

# Auto-optimize chunk size
from src.core.chunking import ChunkOptimizer
optimizer = ChunkOptimizer()
optimal_size, scores = optimizer.find_optimal_chunk_size(documents)
```

### Knowledge Graph Retrieval

```python
from src.retrieval.knowledge_graph import KnowledgeGraphExtractor

# Extract knowledge graph
extractor = KnowledgeGraphExtractor(use_spacy=True)
kg = extractor.extract_from_chunks(chunks)

# Query knowledge graph
from src.retrieval.knowledge_graph import KnowledgeGraphRetriever
kg_retriever = KnowledgeGraphRetriever(kg, chunk_store)
results = kg_retriever.search("AI applications", max_hops=2)
```

### A/B Testing

```python
from src.evaluation.ab_testing import ABTest

# Define variants
def variant_a(query, **kwargs):
    return vector_store.search(query, **kwargs)

def variant_b(query, **kwargs):
    return hybrid_retriever.search(query, **kwargs)

# Run A/B test
ab_test = ABTest("dense_vs_hybrid", variant_a, variant_b)

# Run queries
result, exp_result = ab_test.run_query("What is AI?")

# Analyze
stats = ab_test.analyze()
winner = ab_test.get_winner()
print(ab_test.print_report())
```

### RAGAS Evaluation

```python
from src.evaluation.ragas_metrics import RAGASEvaluator

evaluator = RAGASEvaluator()

# Evaluate single query
score = evaluator.evaluate(
    question="What is machine learning?",
    answer=response.answer,
    contexts=response.contexts,
    ground_truth="ML is a subset of AI..."  # Optional
)

print(f"Faithfulness: {score.faithfulness:.3f}")
print(f"Answer Relevance: {score.answer_relevance:.3f}")
print(f"Context Precision: {score.context_precision:.3f}")
print(f"Overall Score: {score.overall_score:.3f}")

# Batch evaluation
scores = evaluator.evaluate_batch(questions, answers, contexts_list)
aggregated = evaluator.aggregate_scores(scores)
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /index` - Index documents
- `POST /upload` - Upload and index file
- `POST /query` - Query the pipeline
- `POST /query/batch` - Batch queries
- `GET /stats` - Pipeline statistics
- `DELETE /cache` - Clear cache

Full API documentation: http://localhost:8000/docs

## Performance Optimization

### Caching

```python
from src.utils.cache import cached

@cached(ttl=3600)
def expensive_operation():
    # Results cached for 1 hour
    pass
```

### Async Processing

```python
# Async queries
response = await pipeline.query_async(question)

# Batch async
responses = await pipeline.batch_query_async(questions)
```

### Monitoring

Access Prometheus metrics at `/metrics`:

- `rag_queries_total` - Total queries
- `rag_query_latency_seconds` - Query latency histogram
- `rag_chunks_retrieved` - Chunks per query
- `rag_hallucinations_detected_total` - Hallucination count

View dashboards in Grafana: http://localhost:3000

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_retrieval.py -v
```

## Project Structure

```
hybrid-rag-pipeline/
├── src/
│   ├── core/           # Core RAG components
│   │   ├── config.py
│   │   ├── document_loader.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   └── rag_pipeline.py
│   ├── retrieval/      # Retrieval methods
│   │   ├── vector_store.py
│   │   ├── sparse_retrieval.py
│   │   └── knowledge_graph.py
│   ├── reranking/      # Re-ranking
│   │   └── cross_encoder.py
│   ├── evaluation/     # Evaluation & testing
│   │   ├── hallucination_detection.py
│   │   ├── ragas_metrics.py
│   │   └── ab_testing.py
│   ├── api/            # FastAPI application
│   │   └── main.py
│   └── utils/          # Utilities
│       ├── cache.py
│       ├── citation.py
│       ├── monitoring.py
│       └── logger.py
├── tests/              # Test suite
├── notebooks/          # Example notebooks
├── data/               # Data directory
├── config/             # Configuration files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Troubleshooting

### Common Issues

1. **ChromaDB errors**: Ensure directory permissions are correct
   ```bash
   chmod -R 755 chroma_db/
   ```

2. **Memory issues**: Reduce batch size in `.env`
   ```bash
   BATCH_SIZE=16
   ```

3. **Redis connection failed**: Check Redis is running
   ```bash
   docker-compose ps redis
   ```

4. **Slow queries**: Enable caching and reduce `TOP_K_RETRIEVAL`

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file

## Citation

If you use this system in research, please cite:

```bibtex
@software{hybrid_rag_pipeline,
  title={Hybrid RAG Pipeline: Enterprise Retrieval Augmented Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hybrid-rag-pipeline}
}
```

## Support

- Documentation: See `/docs` folder
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## Roadmap

- [ ] Multi-modal embeddings (images, tables)
- [ ] Graph RAG with LLM-generated graphs
- [ ] Fine-tuning reranker on domain data
- [ ] Streaming responses
- [ ] Multi-language support
- [ ] Advanced caching strategies

## Acknowledgments

Built with:
- ChromaDB, FAISS - Vector databases
- Sentence Transformers - Embeddings
- LangChain - RAG patterns
- FastAPI - API framework
- RAGAS - Evaluation framework
