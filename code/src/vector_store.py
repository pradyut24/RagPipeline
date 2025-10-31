import os
import pickle
import chromadb
import numpy as np
from embedder import GeminiEmbedder
from config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
# from sentence_transformers import CrossEncoder
import logging
import google.generativeai as genai

# -------------------------------------------------
# Setup logger
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaRAG:
    """
    RAG Retriever using Chroma vector database with Hybrid and Re-ranking support.

    Features:
    - Persistent Chroma client
    - TF-IDF + BM25 lexical search
    - Hybrid retrieval (Vector + Lexical)
    - Optional re-ranking using CrossEncoder or LLM
    - Query expansion using Gemini
    - Logging & Exception handling
    """

    def __init__(self,
                 db_path=Config.CHROMA_DB,
                 collection_name=Config.collection_name,
                 embeddings_file=Config.EMBEDDINGS_FILE,
                 metadata_file=Config.METADATA_FILE,
                 chunks_file=Config.CHUNK_FILE):

        # --- Initialize Gemini LLM once ---
        try:
            genai.configure(api_key=os.getenv("GEMINI_FREE_API_KEY"))
            self.gemini_model = genai.GenerativeModel(Config.GEMINI_LLM_MODEL)
            logger.info("✅ Gemini model initialized for RAG.")
        except Exception as e:
            self.gemini_model = None
            logger.error(f"❌ Failed to initialize Gemini model: {e}")


        try:
            self.db_path = db_path
            self.collection_name = collection_name
            self.embeddings_file = embeddings_file
            self.metadata_file = metadata_file
            self.chunks_file = chunks_file

            # Ensure DB folder exists
            if not os.path.exists(self.db_path):
                os.makedirs(self.db_path, exist_ok=True)

            # Initialize persistent Chroma client
            self.client = chromadb.PersistentClient(path=self.db_path)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

            # --- TF-IDF and BM25 for hybrid search ---
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.bm25 = None
            self.docs = []
            self._init_lexical_models()

            # --- Optional re-ranking models ---
            # self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # Check if collection already has data
            existing_data = self.collection.get(include=["documents", "metadatas"])
            if len(existing_data.get("documents", [])) == 0:
                logger.info("No existing data found in Chroma — loading embeddings and adding documents...")
                self._load_and_add_documents()
            else:
                logger.info(f"✅ Found {len(existing_data['documents'])} existing documents in Chroma. Skipping insertion.")

        except FileNotFoundError as e:
            logger.error(f"❌ File not found during initialization: {e}")
        except Exception as e:
            logger.error(f"❌ Error initializing ChromaRAG: {e}")

    # -------------------------------------------------
    # Initialize TF-IDF and BM25 models
    # -------------------------------------------------
    def _init_lexical_models(self):
        """Initialize lexical search models (TF-IDF and BM25)."""
        try:
            if os.path.exists(self.chunks_file):
                with open(self.chunks_file, "rb") as f:
                    self.docs = pickle.load(f)

                if self.docs:
                    self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.docs)
                    tokenized_docs = [doc.split(" ") for doc in self.docs]
                    self.bm25 = BM25Okapi(tokenized_docs)
                    logger.info("✅ TF-IDF and BM25 models initialized.")
                else:
                    logger.warning("⚠️ No documents found for lexical models.")
            else:
                logger.warning("⚠️ Chunks file not found. Skipping lexical model initialization.")
        except Exception as e:
            logger.error(f"❌ Error initializing lexical models: {e}")

    def _load_and_add_documents(self):
        """Load pickled embeddings, metadata, and chunks and insert them into Chroma."""
        try:
            if not (os.path.exists(self.embeddings_file) and
                    os.path.exists(self.metadata_file) and
                    os.path.exists(self.chunks_file)):
                raise FileNotFoundError(
                    "Missing embeddings, metadata, or chunks pickle files. Please generate them first."
                )

            # Load files
            with open(self.embeddings_file, "rb") as f:
                embeddings = pickle.load(f)
            with open(self.metadata_file, "rb") as f:
                metadata_list = pickle.load(f)
            with open(self.chunks_file, "rb") as f:
                all_chunks = pickle.load(f)

            if not embeddings or not all_chunks:
                raise ValueError("Loaded embeddings or chunks are empty.")

            # Add to Chroma
            self.collection.add(
                documents=all_chunks,
                embeddings=embeddings,
                metadatas=metadata_list,
                ids=[str(i) for i in range(len(all_chunks))]
            )
            logger.info(f"✅ Successfully inserted {len(all_chunks)} documents into Chroma.")

        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"❌ Error loading pickle files: {e}")
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error while adding documents: {e}")

    # -------------------------------------------------
    # Query expansion using Gemini
    # -------------------------------------------------
    def expand_query(self, query):
        if self.gemini_model is None:
            logger.warning("⚠️ Gemini model not available; Skipping query expansion.")
            return query
        try:
            prompt = f"Suggest query expansions or synonyms for: {query}"
            response = self.gemini_model.generate_content(prompt)
            logger.info(f"✅ Expanded Query : {response.text.strip()}")
            return response.text.strip()

        except Exception as e:
            logger.error(f"❌ Unexpected error while expanding query: {e}")
            return query

    # -------------------------------------------------
    # Hybrid Retrieval (Vector + BM25 + Fusion)
    # -------------------------------------------------
    def hybrid_retrieve(self, query_text, top_k=5, alpha=0.5):
        """
        Hybrid retrieval combining vector similarity + BM25.
        Returns:
            top_docs: list of document chunks
            full_results: dict with documents, fused scores, and article IDs
        """
        try:
            expanded_query = self.expand_query(query_text)

            # --- Vector retrieval ---
            embedder = GeminiEmbedder()
            query_emb = embedder.embed_query(expanded_query)
            vector_results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=top_k * 2,
                include=["documents", "distances", "metadatas"]
            )

            # --- Lexical (BM25) retrieval ---
            bm25_scores = self.bm25.get_scores(expanded_query.split(" "))
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            lexical_docs = [self.docs[i] for i in bm25_top_indices]
            lexical_scores = [bm25_scores[i] for i in bm25_top_indices]

            # --- Fusion ---
            fused_results = {}
            doc_to_article_id = {}

            # Vector results
            for doc, dist, meta in zip(vector_results["documents"][0], vector_results["distances"][0],
                                       vector_results.get("metadatas", [[]])[0]):
                fused_results[doc] = alpha * (1 - dist)
                # always get actual article_id from metadata
                article_id = meta.get("article_id") or meta.get("id")
                if article_id is not None:
                    doc_to_article_id[doc] = article_id
                else:
                    doc_to_article_id[doc] = None  # will filter later if missing

            # Lexical results
            for doc, score in zip(lexical_docs, lexical_scores):
                fused_results[doc] = fused_results.get(doc, 0) + (1 - alpha) * score
                if doc not in doc_to_article_id:
                    # find article_id from Chroma collection metadata
                    # fallback to None if missing
                    doc_meta = next((m for m in vector_results.get("metadatas", [[]])[0] if m.get("document") == doc),
                                    {})
                    doc_to_article_id[doc] = doc_meta.get("article_id") or doc_meta.get("id")

            # --- Sort by fused score ---
            sorted_docs = sorted(fused_results.items(), key=lambda x: x[1], reverse=True)
            top_docs = [d[0] for d in sorted_docs[:top_k]]

            # --- Extract only non-None article IDs ---
            top_article_ids = [doc_to_article_id[d] for d in top_docs if doc_to_article_id[d] is not None]
            metadatas_for_hybrid = [{"id": aid} for aid in top_article_ids]

            # --- Full results dict ---
            full_results = {
                "documents": [top_docs],
                "fused_scores": [fused_results[d] for d in top_docs],
                "metadatas": [metadatas_for_hybrid],
            }

            # Debug log
            logger.info(f"✅ Hybrid retrieval returned {len(top_docs)} documents.")
            logger.info(f"🔹 Article IDs: {top_article_ids}")

            return top_docs, full_results

        except Exception as e:
            logger.error(f"❌ Error in hybrid retrieval: {e}")
            return [], {}

    # -------------------------------------------------
    # Re-ranking stage (CrossEncoder / LLM / Cohere)
    # -------------------------------------------------
    def rerank(self, query, candidate_docs, method="llm"):
        """Re-rank candidate documents using chosen method."""
        try:
            if not candidate_docs:
                return []

            if method == "crossencoder":
                pairs = [[query, doc] for doc in candidate_docs]
                scores = self.cross_encoder.predict(pairs)
                ranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in ranked]



            elif method == "llm":

                if self.gemini_model is None:
                    logger.warning("⚠️ Gemini model not available; skipping rerank.")

                    return candidate_docs

                scored = []

                for doc in candidate_docs:

                    prompt = (

                        f"Rate the relevance of the following document to the query on a scale 0 to 1:\n\n"

                        f"Query: {query}\n\nDocument: {doc}\n\nRelevance score:"

                    )

                    try:

                        response = self.gemini_model.generate_content(prompt)

                        score = float(response.text.strip())

                    except Exception:

                        score = 0.0

                    scored.append((doc, score))

                ranked = sorted(scored, key=lambda x: x[1], reverse=True)

                return [doc for doc, _ in ranked]

            elif method == "cohere":
                # Placeholder for Cohere Rerank API integration
                logger.info("Using Cohere Rerank API (placeholder).")
                return candidate_docs

            else:
                logger.warning("⚠️ Invalid rerank method; skipping.")
                return candidate_docs

        except Exception as e:
            logger.error(f"❌ Error during reranking: {e}")
            return candidate_docs

    # -------------------------------------------------
    # Simple vector retrieval (original)
    # -------------------------------------------------
    def retrieve(self, query_text, top_k):
        """
        Retrieve top_k most relevant chunks for the query_text (vector-only).
        Returns:
            retrieved_chunks: list of unique document chunks
            full_results: dict containing documents, metadatas, and distances
        """
        try:
            if not query_text or not isinstance(query_text, str):
                raise ValueError("Query text must be a non-empty string.")

            embedder = GeminiEmbedder()
            query_emb = embedder.embed_query(query_text)

            if query_emb is None:
                raise ValueError("Failed to generate embedding for the query.")

            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            if not results or "documents" not in results or not results["documents"]:
                logger.warning("⚠️ No documents retrieved for the query.")
                return [], {}

            retrieved_chunks = list(set(results["documents"][0]))
            return retrieved_chunks, results

        except ValueError as e:
            logger.error(f"❌ Invalid input or embedding issue: {e}")
            return [], {}
        except Exception as e:
            logger.error(f"❌ Error during retrieval: {e}")
            return [], {}
