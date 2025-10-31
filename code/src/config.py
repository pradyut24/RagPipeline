import os

class Config:
    DATA_PATH = "data/input_article_details.jsonl"
    CHUNK_FILE = "data/chunks.pkl"
    EMBEDDINGS_FILE = "data/embeddings.pkl"
    METADATA_FILE = "data/metadata.pkl"
    checkpoint_path = "data/embeddings_checkpoint.pkl"
    CHROMA_DB = "./chromadb"
    collection_name = "articles"
    VECTOR_DB_DIR = "artifacts/chroma_db"
    GEMINI_EMBED_MODEL = "gemini-embedding-001"
    GEMINI_LLM_MODEL = "gemini-2.0-flash"  # For answer generation
    CHUNK_SIZE = 400                  # tokens
    CHUNK_OVERLAP = 50
    TOP_K = 5                         # retrieval
    EMBEDDING_CACHE_FILE = "artifacts/embedding_cache.json"

