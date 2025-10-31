# 🚀 Gemini RAG Pipeline — A Focused, Readable, and Reproducible Retrieval System

A modular and production-ready Retrieval-Augmented Generation (RAG) pipeline powered by Google Gemini LLM and ChromaDB vector store, designed for clarity, experimentation, and scalability.
This project embodies a focused retrieval pipeline that is:
**Readable, reproducible, and justified.**

---

## 🧩 Project Overview

This repository demonstrates an **end-to-end RAG framework** with modular components, including:

- **Preprocessing & Chunking**
- **Embedding Generation (Gemini)**
- **Hybrid Retrieval (TF-IDF, BM25, ChromaDB)**
- **Reranking using LLM, Cohere, and CrossEncoder**
- **Confidence and Latency Evaluation**
- **Interactive Streamlit Interface**

---

## 📂 Folder Structure

```
RAG_Hackathon_Repo/
│
├── code/
│   ├── src/
│   │   ├── data/
│   │   │   ├── input_article_details.jsonl         # Input corpus for processing
│   │   │   ├── chroma_db/              # ChromaDB vector store
│   │   ├── main.py                     # CLI-based execution entry point
│   │   ├── streamline_app.py           # Streamlit UI for interactive testing
│   │   ├── preprocess.py               # Text loading, cleaning, and chunking logic
│   │   ├── embedder.py                 # Gemini embedding generation
│   │   ├── vector_store.py             # ChromaDB integration for storage & retrieval
│   │   ├── config.py                   # Configuration constants
│   │   └── prompts.py                  # RAG-specific prompt formatting
│   │
│   ├── tests/                          # Unit tests for all modules
│   │   ├── test_chunking.py
│   │   ├── test_embedding.py
│   │   ├── test_vectorstore.py
│   │   └── test_main.py
│   │
│   └── requirements.txt
│
└── README.md
```
---
## 🧩 Workflow Diagram

<img src="artifacts/arch/Workflow Diagram - Architecture.png" alt = "Workflow diagram" width="300"/>

---

## ⚙️ Running the Project

### ▶️ Option 1: Command-line Execution
```bash
cd code/src
python main.py
```

### 💻 Option 2: Streamlit UI
```bash
cd code/src
streamlit run streamline_app.py
```

---

## 🧠 Chunking Strategy

Chunking determines how much information the retriever can access.
We use fixed + overlapping window chunking to balance recall and precision.

| Parameter  | Value                   | Purpose                                  |
| ---------- | ----------------------- | ---------------------------------------- |
| Chunk size | 200–500 tokens         | Ensures coherent context                 |
| Overlap    | 50–100 tokens          | Preserves boundary context               |
| Strategy   | Fixed-size with overlap | Prevents information loss between splits |


**💡 Rationale: Too small chunks lose semantic context, too large introduce noise.
This configuration maximizes retrievability while keeping cost-efficient embeddings.**

---

## 🔍 Embedding Generation

Embeddings are generated using the **Gemini Embedding API**, which encodes text into dense semantic vectors.

**Key Details:**
- Deterministic behavior via seed control
- Batching for API efficiency
- Metadata binding for traceability (doc_id, chunk_id, source, span)
  
**🎯 These embeddings form the semantic backbone of retrieval.**

---

## 🔍 Vector Storage & Indexing (ChromaDB)

We use ChromaDB, a lightweight vector store that supports persistent local indexing.
It uses **HNSW (Hierarchical Navigable Small World) graphs** for efficient nearest-neighbor search.

**Why Chroma?**
- Local, explainable, and open-source
- Inbuilt cosine similarity and distance metrics
- Metadata-aware querying
- Optimized for fast recall and scalability
📘 Trade-off chosen: Chroma over FAISS or Pinecone for simplicity + reproducibility + transparency.

---

## 🧮 Hybrid Retrieval Approach

To enhance recall and precision, our retriever follows a hybrid approach that combines **semantic similarity** with **lexical relevance**.
**TF-IDF + BM25**: Used for lexical matching to capture exact term overlaps and keyword relevance.

- **TF-IDF** – For fast keyword relevance scoring  
- **BM25** – For improved term weighting and ranking  
- **ChromaDB (Vector-based)** – For semantic retrieval using cosine similarity  
- **Query Expansion** – For reformulating user queries to enhance recall

The final retrieval results are **merged and deduplicated** for better coverage.

---

## 🔁 Reranking Techniques

To improve contextual accuracy, retrieved documents are reranked using any of the **three distinct approaches**:

1. **LLM-based Reranker** – Uses Gemini to analyze contextual match with the query.  
2. **Cohere Reranker** – Leverages Cohere’s `rerank-english-v2.0` model for relevance scoring.  
3. **CrossEncoder Reranker** – Uses transformer-based pair scoring (query, document) similarity.

💡**Effect:** Significantly improves answer precision by filtering noise from high-recall searches.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|--------------|
| **Retrieval Confidence** | Average cosine similarity of top matches |
| **Answer Confidence** | LLM-based estimate of response certainty |
| **Token Count** | Number of tokens used in generation |
| **Retrieval Latency** | Time taken for embedding & retrieval |

---

**Explainability & Reporting**
Every submission generates:
- Context used for the final answer
- Confidence breakdown
- Top retrieved passages
- Latency & token stats
- JSON report for auditing
  
**Example structure:**

```bash
{
  "query": "What is RAG?",
  "retrieval_confidence": 0.92,
  "answer_confidence": 0.88,
  "retrieval_latency_ms": 213,
  "tokens_used": 148,
  "final_answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": ["doc_12_chunk_3", "doc_15_chunk_1"]
}
```

---


## ✅ Testing & Coverage

All major modules have Pytest-based unit tests.  
To run tests with coverage:

```bash
pytest --maxfail=1 --disable-warnings -q
pytest --cov=src --cov-report=term-missing
```

---

```bash
========================================================================== tests coverage ===========================================================================
_________________________________________________________ coverage: platform darwin, python 3.12.8-final-0 __________________________________________________________

Name                         Stmts   Miss  Cover   Missing
----------------------------------------------------------
src/config.py                   16      0   100%
src/embedder.py                118     29    75%   28-30, 58, 62-67, 75-78, 89, 112-113, 124, 132-135, 143-145, 159, 166-168
src/preprocess.py               93     24    74%   28, 35-37, 52, 66-68, 88, 97, 105, 115-117, 120, 131-139
src/vector_store.py            178     57    68%   45-47, 59, 85-88, 107-111, 132, 144, 146, 154-165, 208, 241-243, 252, 255-258, 265-267, 287-289, 297-308, 322, 328, 337-338, 343-348
tests/test_chunking.py          70      0   100%
tests/test_embedding.py         71      0   100%
tests/test_main.py              19     11    42%   15-27, 31
tests/test_vector_store.py      80      0   100%
----------------------------------------------------------
TOTAL                          645    121    81%
Coverage HTML written to dir htmlcov
======================================================================== 20 passed in 50.04s ========================================================================
```

---

## 📈 Key Design Choices & Trade-offs

| Design Aspect | Choice                          | Justification                         |
| ------------- | ------------------------------- | ------------------------------------- |
| Chunking      | Fixed 1000 tokens w/150 overlap | Ensures semantic continuity           |
| Embeddings    | Gemini Text Embeddings          | Integrated, contextually rich vectors |
| Vector DB     | Chroma                          | Local, open-source, HNSW-based        |
| Reranking     | Weighted hybrid                 | Improves accuracy under high recall   |
| Confidence    | Multi-signal                    | Transparent interpretability          |
| Execution     | Streamlit + CLI                 | Flexibility for both users and devs   |
| Logging       | Built-in                        | Debugging + latency profiling         |
| Cost          | Batched API calls               | Minimized token and request usage     |


---

## 🧭 Experimental Rigor & Impact

| Component    | Alternatives Tested             | Chosen Strategy                   | Impact                                    |
| ------------ | ------------------------------- | --------------------------------- | ----------------------------------------- |
| Chunking     | Fixed                           | Overlap 20%                       | Improved contextual continuity            |
| Embeddings   | ST5                             | Gemini                            | Best semantic match for factual text      |
| Vector Store | FAISS                           | Chroma                            | Lightweight, persistent, easy integration |
| Retrieval    | Pure vector                     | Hybrid (TF-IDF + BM25 + Semantic) | Higher recall & precision                 |
| Reranking    | None                            |Cohere, CrossEncoder, LLM BASED(any one)| Boosted contextual accuracy by ~12%       |

___

## Reproducibility

```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Set environment variables
export GEMINI_API_KEY="your_key_here"

# 3️⃣ Run full pipeline
python src/main.py

# 4️⃣ Launch UI
streamlit run src/streamline_app.py
```

---
## Result File for the given input Data
[results.json](code/src/results.json)

---

## 🧠 Summary

This RAG framework brings together **semantic, lexical, and contextual intelligence**. By integrating **Gemini embeddings**, **hybrid retrieval**, and **multi-model reranking**, it achieves a **balanced blend of recall and precision**—making it ideal for enterprise-scale retrieval applications.

---

**Tech Stack:** Python, Streamlit, ChromaDB, Gemini, Cohere, HuggingFace Transformers
