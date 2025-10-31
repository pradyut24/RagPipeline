# The Retrieval-First RAG Hackathon

## Problem Statement

Large Banks manage massive knowledge repositories policy documents, compliance manuals, product FAQs, operational procedures, risk guidelines, and regulatory updates. Employees across departments (customer service, compliance, operations, IT) often struggle to find precise, compliant answers quickly, especially under strict regulatory constraints.

Traditional search systems fail because:

- They rely on keyword matching, not semantic understanding.
- They cannot handle domain-specific jargon or multi-step reasoning.
- They do not ensure answers are grounded in official sources, risking compliance breaches.

### Learning Objectives

By the end of this hackathon, participants will:

1. **Understand RAG fundamentals:** Learn how Retrieval-Augmented Generation works by building the critical retrieval component from scratch
2. **Master retrieval engineering:** Gain hands-on experience with chunking strategies, embedding models, vector databases, and hybrid retrieval approaches
3. **Apply engineering best practices:** Develop skills in modular design, testing, reproducibility, and performance optimization
4. **Make informed trade-offs:** Learn to justify technical decisions based on accuracy, performance, and resource constraints (including API costs)
5. **Build production-ready systems:** Create maintainable, configurable, and well-documented pipelines that could be deployed in real-world scenarios

### Repository Structure

- **`code/src/`** — Source Code
- **`code/tests/`** — Comprehensive test suite:
  - Unit tests for each major component (chunking, embedding, retrieval)
  - Integration tests for the full pipeline
  - Edge cases (empty queries, no matches, malformed input)
- **`artifacts/`** — These artifacts will be evaluated by the automated LLM reviewer

## Submission Artifacts

### Code

- **`code/src/`** — Main code organized into clear modules

### Required files under `artifacts/`

- **`results.json`** — The submission mapping questions to ranked article IDs
- **`setup-instructions.md`** — Environment and reproducible run steps
- **`design-overview.json`** — Design summary including:
  - Architecture diagram/description (preprocessing → indexing → retrieval flow)
  - Key design decisions and rationale for
    - chunking strategy
    - Vector Store Selection
    - Embedding model choice
    - Retrieval approach
    - Batched embedding generation
    - Cost optimization and resource management
  - API usage and cost analysis (which APIs used, estimated costs, optimization strategies)
  - Trade-offs considered (accuracy vs speed vs cost)
  - Known limitations and failure modes
  - Ideas for future improvements
- **`video.mp4`** — Short demo video (visual aids like highlighting matched text in retrieval traces are encouraged)
- **`answer-generation-prompt.md`** — Prompt used for answer generation

### Structure of results.json

```json
[
  {
    "question_id": 1,
    "question": "Am I allowed to accept payments on my site while my Wix Payments account is still being verified?",
    "answer": "You can start accepting payments on your site using [Wix Payments](https://support.wix.com/en/article/about-wix-payments) almost immediately.....",
    "article_ids": [
      "49d988fadbf11fa4685c847590078ff9394c2fe7566094f504f53ca4aca465"
    ]
  },
  {
    "question_id": 2,
    "question": "I am looking to purchase the yearly premium plan for $17.00 which comes with a free domain for one year. However the voucher isn't appearing at checkout...",
    "answer": "When you purchase a yearly premium....",
    "article_ids": [
      "49d988fadbf11fa4685c847590078ff9394c2fe7566094f504f53ca4aca465",
      "AsD988fadbf11fa4685c847590078ff9394c2fe7566094f504f53ca4aca465"
    ]
  }
]
```

## Scoring

**Total: 100 points**

The suggested weights below reflect the hackathon's pedagogical emphasis on solid engineering, modular design, and reproducibility rather than purely optimizing retrieval metrics.

### 1. Retrieval Accuracy — 20 points

### 2. Pipeline & Code Quality — 30 points _(Primary Learning Objective)_

- **Clear separation of concerns**:
  - Distinct modules for preprocessing, indexing, and retrieval
  - Each component has a single, well-defined responsibility
  - Clean interfaces between components
- **Modularity and configurability**:
  - All hyperparameters externalized to config files
  - Easy to swap embedding models, similarity metrics, or retrieval strategies
  - Components can be tested and developed independently
- **Error handling, reproducibility, and test coverage**:
  - Proper exception handling and validation
  - Deterministic behavior (seeded random operations)
  - Tests cover core functionality and edge cases
  - Clear logging for debugging
- **Clean repo layout, maintainability, and code clarity**:
  - Consistent code style and documentation
  - Meaningful variable/function names
  - Comments explaining non-obvious design choices
  - README makes it easy for others to understand and run the code

### 3. Scalability & Performance — 20 points _(including cost optimization and resource management)_

- **Batched embedding generation, caching, and efficient index usage**:
  - Batch API calls to reduce latency and cost
  - Cache embeddings to avoid redundant API calls
  - Efficient vector index usage
- **Cost optimization and resource management**:
  - Track and minimize API costs
  - Implement rate limiting and retry logic
  - Use appropriate model tiers (e.g., text-embedding-3-small vs large)

### 4. Documentation & Explainability — 30 points

- README runbook and design notes (10 pts)
- Short writeup justifying chunking, embedding/storage choices, and failure modes (10 pts)

## High-Level Expectations and Areas to Explore _(Not Prescriptive)_

The goal of these notes is to point teams toward meaningful design choices to justify — not to prescribe exact implementation steps. Judges will reward clear reasoning and trade-offs.

### Key RAG Concepts to Explore

#### 1. **Chunking & Context Preservation**

**Why it matters:** Chunking determines what information the retriever can find. Too small = lost context; too large = noisy matches.

**What to explore:**

- Fixed-size vs semantic chunking (paragraphs, sentences)
- Chunk size (200–600 tokens is typical, but experiment!)
- Overlap strategies (0%, 10%, 20%) to preserve context across boundaries
- Metadata preservation (document title, section headers)

**What to document:** Your chosen strategy, experiments that led to it, and observed impact on retrieval quality.

---

#### 2. **Embedding Model Selection**

**Why it matters:** Different embedding models capture different semantic relationships. This is the "brain" of your semantic search.

**What to explore:**

- Commercial embedding providers:
  - **OpenAI** (`text-embedding-3-small`, `text-embedding-3-large`) — widely used, good general performance
  - **Cohere** (`embed-english-v3.0`, `embed-multilingual-v3.0`) — excellent for search/retrieval tasks
  - **Voyage AI** (`voyage-2`, `voyage-large-2`) — optimized for retrieval
  - **Google Vertex AI** embeddings — integrated with Gemini ecosystem
- Embedding dimensionality trade-offs (512 vs 1024 vs 1536 vs 3072 dimensions)
- Cost per token vs retrieval quality
- Caching strategies to minimize API costs

**What to document:** Model choice, why it fits the KB domain, performance/cost trade-offs, and total embedding cost for the dataset.

---

#### 3. **Vector Storage & Indexing**

**Why it matters:** Efficient storage and search are crucial for scalability beyond this small dataset.

**What to explore:**

- Simple approaches: NumPy arrays with cosine similarity, FAISS for local indexing (teams may start simple and explain trade-offs before moving to vector DBs)
- Vector databases: Chroma, Qdrant, Pinecone, Weaviate
- Hybrid storage: vectors + metadata + original text
- Index types (flat, HNSW, IVF) and their speed/accuracy trade-offs

**What to document:** Your storage solution, why it's appropriate for this scale, and how it would scale to 100K+ documents.

---

#### 4. **Retrieval Strategy & Ranking**

**Why it matters:** Combining multiple signals often beats pure vector search. This is where RAG systems shine.

**What to explore:**

- Pure vector similarity (cosine, dot product)
- Hybrid retrieval: vector + BM25/TF-IDF (lexical matching)
- Fusion strategies: reciprocal rank fusion, weighted combinations
- Re-ranking using:
  - Cross-encoder models (via APIs)
  - LLM-based re-ranking (GPT-4/5, Claude, Gemini as judges)
  - Cohere Rerank API or similar commercial services
- Query expansion or reformulation using LLMs
- Multi-stage retrieval (broad recall → precision re-ranking)

**What to document:** Your retrieval pipeline, how signals are combined, API costs for re-ranking, and experiments comparing approaches.

---

#### 5. **Confidence Thresholding & No-Answer Detection**

**Why it matters:** In production RAG, saying "I don't know" is better than hallucinating from irrelevant context.

**What to explore:**

- Similarity score thresholds
- Relative score gaps (best vs 2nd-best match)
- Cluster-based confidence (do top-K results agree?)
- Calibration experiments on known no-answer questions

**What to document:** Your threshold strategy, how you tuned it, and precision/recall trade-offs.

---

#### 6. **Reproducibility & Engineering Discipline**

**Why it matters:** Production ML systems must be debuggable, testable, and maintainable.

**What to explore:**

- Dependency management (requirements.txt, poetry, conda)
- API key management (environment variables, `.env` files, secrets management)
- Deterministic behavior (random seeds, sorted results)
- Configuration management (YAML, environment variables)
- Logging and observability (what gets logged at each pipeline stage?)
- Error handling (API failures, rate limits, malformed queries, missing data)
- Cost tracking and optimization (monitoring API usage)
- Caching strategies (embeddings, API responses) to reduce costs and improve speed

**What to document:** How others can reproduce your results exactly (including API setup), and how you'd debug production issues and manage costs.

---

#### 7. **Explainability & Debugging**

**Why it matters:** Understanding why a document was retrieved helps debug failures and build trust.

**What to explore:**

- Logging similarity scores for top-K results
- Highlighting matched chunks or key terms
- Providing retrieval reasoning (e.g., "matched on BM25 for term X, vector similarity 0.87")
- Failure mode analysis (common query types that fail)

**What to document:** Sample retrieval traces, common failure patterns, and debugging insights.

---

**Note:** Teams are encouraged to explore these areas experimentally and include brief notes in `design-overview.json` explaining what they tried and learned. Judges will look for clear rationale and evidence of experimentation more than matching any single "best" approach.

## Recommended Resources & Learning Materials

To help you build a strong foundation, consider these resources:

### RAG & Retrieval Fundamentals

- **LangChain RAG Tutorial** — Practical introduction to RAG pipelines
- **Pinecone Learning Center** — Excellent guides on vector search and embeddings
- **"Building RAG Systems" by LlamaIndex** — Comprehensive RAG architecture patterns

### Commercial LLM & Embedding APIs

- **OpenAI Embeddings Documentation** — API reference and best practices
- **Cohere Embed API** — Optimized for retrieval tasks
- **Voyage AI Documentation** — Retrieval-focused embeddings
- **Google Vertex AI** — Gemini ecosystem integration
- **Anthropic Claude API** — For LLM-based re-ranking and query expansion
- **API Cost Calculators** — Plan your usage and optimize costs

### Vector Databases & Search

- **FAISS Documentation** — Facebook's efficient similarity search library
- **Chroma Quickstart** — Simple vector database for prototyping
- **Weaviate/Qdrant Tutorials** — Production-grade vector database options
- **Pinecone** — Managed vector database service

### Engineering Best Practices

- **"The Twelve-Factor App"** — Configuration and deployment principles
- **Pytest Documentation** — Writing effective tests in Python
- **"Clean Code" principles** — Writing maintainable, readable code
- **API Best Practices** — Rate limiting, retries, error handling

### Cost Optimization

- **Caching strategies** — Reduce redundant API calls
- **Batch processing** — Optimize API usage
- **Model selection guides** — Choosing the right tier for your needs

---

## Wrap-Up: What We Expect to See

A focused retrieval pipeline that is **readable, reproducible, and justified**. The best teams will:

1. **Demonstrate RAG understanding** by explaining how their retrieval choices impact downstream answer quality
2. **Show experimental rigor** with evidence of testing different approaches and measuring impact
3. **Apply engineering discipline** through clean code, comprehensive tests, and proper documentation
4. **Make informed trade-offs** between accuracy, speed, cost, and complexity
5. **Build for the real world** with error handling, logging, maintainable architecture, and cost-conscious design
6. **Optimize API usage** through caching, batching, and smart model selection

**This is a learning opportunity.** We value thoughtful experimentation, clear documentation of failures and successes, and engineering practices over perfect metrics. Show us you understand RAG fundamentals and can build cost-effective, production-ready systems others would want to use.

**Important:** Include a cost analysis in your submission showing estimated API costs for processing the full dataset and handling typical query loads.

**Good luck — and remember: understanding why your system works (or fails) is more valuable than any single metric!**

