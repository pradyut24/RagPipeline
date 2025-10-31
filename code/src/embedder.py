import os
import math
import pickle
import time
from google import genai
from config import Config
from dotenv import load_dotenv

load_dotenv()


class GeminiEmbedder:
    """
    GeminiEmbedder generates text embeddings using Google's Gemini API.
    Supports batching, resume on failure, and automatic checkpointing.
    Includes exception handling and logging for reliability.
    """

    def __init__(self, batch_size=50, max_bytes=3_500_000, checkpoint_path=Config.checkpoint_path):
        try:
            cfg = Config()
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.model_name = "gemini-embedding-001"
            self.batch_size = batch_size
            self.max_bytes = max_bytes
            self.checkpoint_path = checkpoint_path
            print("✅ GeminiEmbedder initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing GeminiEmbedder: {e}")
            raise

    def _split_large_text(self, text):
        """Split text into parts if it exceeds Gemini's max byte size."""
        try:
            if not isinstance(text, str):
                raise ValueError("Input text must be a string.")
            encoded = text.encode("utf-8")
            if len(encoded) <= self.max_bytes:
                return [text]

            parts, start = [], 0
            while start < len(encoded):
                end = min(start + self.max_bytes, len(encoded))
                parts.append(encoded[start:end].decode("utf-8", errors="ignore"))
                start = end
            return parts
        except Exception as e:
            print(f"❌ Error splitting text: {e}")
            return [text]

    def _load_checkpoint(self):
        """Load previous embedding progress if available."""
        try:
            if os.path.exists(self.checkpoint_path):
                with open(self.checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)
                if not isinstance(checkpoint, dict) or "embeddings" not in checkpoint:
                    raise ValueError("Invalid checkpoint format.")
                print(f"🔹 Resuming from checkpoint: {len(checkpoint['embeddings'])} batches completed.")
                return checkpoint
            return {"embeddings": [], "completed_batches": 0}
        except (pickle.PickleError, EOFError) as e:
            print(f"⚠️ Checkpoint corrupted: {e}. Starting fresh.")
            return {"embeddings": [], "completed_batches": 0}
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return {"embeddings": [], "completed_batches": 0}

    def _save_checkpoint(self, embeddings, completed_batches):
        """Save embedding progress for resuming later."""
        try:
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump({"embeddings": embeddings, "completed_batches": completed_batches}, f)
            print(f"💾 Saved checkpoint at batch {completed_batches}")
        except (OSError, pickle.PickleError) as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
        except Exception as e:
            print(f"❌ Unexpected error saving checkpoint: {e}")

    def embed_documents(self, documents):
        """
        Generate embeddings for a list of documents.
        Includes batching, checkpointing, and retry-safe failure handling.
        """
        try:
            if not isinstance(documents, list) or not all(isinstance(d, str) for d in documents):
                raise ValueError("Documents must be a list of strings.")
            if len(documents) == 0:
                raise ValueError("No documents provided for embedding.")

            all_embeddings = []
            safe_docs = []

            # Split large documents into manageable chunks
            for doc in documents:
                safe_docs.extend(self._split_large_text(doc))

            total_batches = math.ceil(len(safe_docs) / self.batch_size)
            checkpoint = self._load_checkpoint()

            all_embeddings = checkpoint["embeddings"]
            start_batch = checkpoint["completed_batches"]

            print(f"🚀 Starting embedding from batch {start_batch + 1}/{total_batches}")

            for batch_num in range(start_batch, total_batches):
                start = batch_num * self.batch_size
                end = start + self.batch_size
                batch = safe_docs[start:end]

                if not batch:
                    print(f"⚠️ Skipping empty batch {batch_num + 1}")
                    continue

                print(f"Embedding batch {batch_num + 1}/{total_batches} ({len(batch)} docs)...")

                try:
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch
                    )

                    if not hasattr(result, "embeddings") or not result.embeddings:
                        raise ValueError("No embeddings returned by Gemini API.")

                    all_embeddings.extend([emb.values for emb in result.embeddings])

                    # Save progress safely
                    self._save_checkpoint(all_embeddings, batch_num + 1)
                    time.sleep(5)  # pacing to avoid API rate limits

                except Exception as e:
                    print(f"❌ Error on batch {batch_num + 1}: {e}")
                    print("⏸️ Stopping... You can rerun later to continue from next batch.")
                    break

            print("✅ Embedding complete or paused.")
            return all_embeddings

        except ValueError as e:
            print(f"❌ Input validation error: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error during embedding: {e}")
            return []

    def embed_query(self, query):
        """Generate embedding for a single query string."""
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string.")

            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[query]
            )

            if not hasattr(result, "embeddings") or not result.embeddings:
                raise ValueError("No embeddings returned for query.")

            return result.embeddings[0].values

        except ValueError as e:
            print(f"❌ Query validation error: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error during query embedding: {e}")
            return []
