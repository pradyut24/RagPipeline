"""
main.py — Gemini RAG Pipeline with Hybrid Retrieval + Re-Ranking
----------------------------------------------------------------
End-to-end process:
- Prepare/load chunks and embeddings
- Choose retrieval type: vector-only or hybrid
- Optional re-ranking (CrossEncoder/LLM/Cohere)
- Query Gemini LLM
- Compute confidence, latency, and token metrics
"""

import os
import pickle
import logging
import random
import time
import numpy as np
import google.generativeai as genai
from preprocess import prepare_chunks
from vector_store import ChromaRAG
from prompts import make_rag_prompt
from embedder import GeminiEmbedder
from config import Config
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Setup
# -------------------------------
random.seed(getattr(Config, "SEED", 42))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------
# 1. Load or Create Chunks
# -------------------------------
try:
    if not hasattr(Config, "DATA_PATH") or not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError("❌ DATA_PATH not found in Config or file missing.")
    logger.info("🔄 Preparing text chunks from dataset...")
    all_chunks, metadata_list = prepare_chunks(input_file=Config.DATA_PATH)
    if not all_chunks:
        raise ValueError("❌ No chunks generated from input file.")
    logger.info(f"✅ Prepared {len(all_chunks)} chunks successfully.")
except Exception as e:
    logger.exception(f"Failed to prepare chunks: {e}")
    raise

# -------------------------------
# 2. Create or Load Embeddings
# -------------------------------
try:
    if not os.path.exists(Config.EMBEDDINGS_FILE):
        logger.info("📦 No embeddings.pkl found — generating new embeddings...")
        embedder = GeminiEmbedder()
        embeddings = embedder.embed_documents(all_chunks)

        with open(Config.EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"✅ Created {len(embeddings)} embeddings and saved to {Config.EMBEDDINGS_FILE}.")
    else:
        logger.info("✅ embeddings.pkl already exists. Skipping embedding generation.")
        with open(Config.EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)
except Exception as e:
    logger.exception(f"Failed to generate or load embeddings: {e}")
    raise

# -------------------------------
# 3. Initialize Chroma RAG Store
# -------------------------------
try:
    rag_store = ChromaRAG(db_path=Config.CHROMA_DB, collection_name=Config.collection_name)
    logger.info("✅ Chroma RAG store initialized successfully.")
except Exception as e:
    logger.exception(f"Failed to initialize ChromaRAG: {e}")
    raise

# -------------------------------
# 4. Helper: Gemini + Confidence
# -------------------------------
def query_gemini(prompt: str) -> str:
    """Send the constructed prompt to Gemini and return the text output."""
    try:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        genai.configure(api_key=os.getenv("GEMINI_FREE_API_KEY"))
        model = genai.GenerativeModel(Config.GEMINI_LLM_MODEL)
        answer = model.generate_content(prompt)
        logger.info("✅ Gemini model response received.")
        return answer.text
    except Exception as e:
        logger.exception(f"Gemini query failed: {e}")
        return f"Error generating response: {e}"

def compute_confidences(full_results):
    """Compute retrieval confidences using distance, rank, and spread consistency."""
    if not full_results or "distances" not in full_results:
        return [1.0]

    distances = np.array(full_results["distances"][0])
    if len(distances) == 0:
        return [1.0]

    d_min, d_max = np.min(distances), np.max(distances)
    d_mean, d_std = np.mean(distances), np.std(distances)

    dist_scores = 1 - (distances - d_min) / (d_max - d_min + 1e-9)
    rank_scores = np.exp(-0.3 * np.arange(len(distances)))
    spread_score = 1 - (d_std / (d_mean + 1e-9))
    final_conf = 0.6 * dist_scores + 0.3 * rank_scores + 0.1 * spread_score
    return np.round(np.clip(final_conf, 0, 1), 2).tolist()

# -------------------------------
# 5. Main Generation Pipeline
# -------------------------------
# -------------------------------
# Assume 'rag' is your ChromaRAG instance
# -------------------------------

def generate_response(query_input, top_k=5, use_hybrid=False, use_rerank: bool = True):
    """
    Retrieve context (vector or hybrid), optional rerank, compute confidence, and query Gemini LLM.
    Returns a dictionary with answer, sources, metrics, and retrieval info.
    """
    try:
        start_time = time.time()
        if not query_input.strip():
            raise ValueError("Query input cannot be empty.")

        # --- Retrieve documents ---
        if use_hybrid:
            retrieved_docs, full_results = rag_store.hybrid_retrieve(query_input, top_k=top_k)
            logger.info(f"🔍 Hybrid retrieval returned {len(retrieved_docs)} documents.")
        else:
            retrieved_docs, full_results = rag_store.retrieve(query_input, top_k=top_k)
            logger.info(f"🔍 Vector retrieval returned {len(retrieved_docs)} documents.")

        if not retrieved_docs:
            logger.warning("⚠️ No documents retrieved for the query.")
            return {
                "answer": "Sorry, no relevant information found.",
                "retrieved_docs": [],
                "retrieval_confidences": [],
                "token_count": 0,
                "latency": 0.0,
                "answer_confidence": 0.0,
                "article_ids": []
            }

        # --- Flatten retrieved_docs if nested ---
        flat_docs = []
        for doc in retrieved_docs:
            if isinstance(doc, list):
                flat_docs.extend(doc)
            else:
                flat_docs.append(doc)

        # --- Optional Re-ranking ---
        if use_rerank:
            flat_docs = rag_store.rerank(query_input, flat_docs)
            logger.info("✅ Re-ranking completed.")

        # --- Build RAG prompt ---
        relevant_passage = "\n".join(flat_docs)
        rag_prompt = make_rag_prompt(query=query_input, relevant_passage=relevant_passage)

        # --- Call Gemini LLM ---
        genai.configure(api_key=os.getenv("GEMINI_FREE_API_KEY"))
        model = genai.GenerativeModel(Config.GEMINI_LLM_MODEL)
        llm_response = model.generate_content(rag_prompt)
        answer_text = llm_response.text.strip() or "Sorry, could not generate an answer."

        # --- Compute metrics ---
        confidences = compute_confidences(full_results) if full_results else [1.0] * len(flat_docs)
        latency = round(time.time() - start_time, 2)
        token_count = len(answer_text.split())
        answer_conf = round(np.mean(confidences), 2) if confidences else 0.0

        # if full_results and "metadatas" in full_results:
        #     article_ids = list(
        #         set([item.get('id') for item in full_results.get('metadatas', []) if 'id' in item])
        #     )
        # else:
        #     # fallback for hybrid: use top_docs indices as IDs
        #     article_ids = [str(i) for i in range(len(flat_docs))]
        logger.info(f"metadata : {full_results.get('metadatas', [])}")
        # --- Extract Article IDs ---
        article_ids = list(
            set([item['id'] for sublist in full_results.get('metadatas', []) for item in sublist])
        ) if full_results else []

        return {
            "query": query_input,
            "answer": answer_text,
            "retrieved_docs": flat_docs,
            "retrieval_confidences": confidences,
            "token_count": token_count,
            "latency": latency,
            "answer_confidence": answer_conf,
            "article_ids": article_ids,
        }

    except Exception as e:
        logger.exception(f"❌ Fatal error in generate_response: {e}")
        return {
            "answer": "An error occurred while processing the query.",
            "retrieved_docs": [],
            "retrieval_confidences": [],
            "token_count": 0,
            "latency": 0.0,
            "answer_confidence": 0.0,
            "article_ids": [],
        }



# -------------------------------
# 6. User Interaction
# -------------------------------
if __name__ == "__main__":
    try:
        query_input = input("🔎 Enter your query: ").strip()
        top_k = int(input("How many top results to retrieve? (default 5): ") or 5)
        use_hybrid = input("Use hybrid retrieval (vector + BM25)? (y/n): ").lower().startswith("y")
        use_rerank = input("Use re-ranking? (y/n): ").lower().startswith("y")

        logger.info("🚀 Starting retrieval pipeline...")
        result = generate_response(query_input, top_k=top_k, use_hybrid=use_hybrid, use_rerank=use_rerank)

        print("\n📘 Final Answer:\n", result["answer"])
        print("\n📊 Metrics:")
        print("  Token Count:", result["token_count"])
        print("  Latency (s):", result["latency"])
        print("  Answer Confidence:", result["answer_confidence"])
        print("  Retrieval Confidences:", result["retrieval_confidences"])
        print("\n🧾 Related Article IDs:", result["article_ids"])

    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")

