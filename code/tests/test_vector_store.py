import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../src")))
import pickle
import pytest
from unittest.mock import MagicMock, patch
from vector_store import ChromaRAG

# -------------------------------
# 1. Setup temporary pickle files for testing
# -------------------------------

@pytest.fixture
def setup_pickles(tmp_path):
    chunks = ["doc1 content", "doc2 content"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    metadata = [{"id": "1"}, {"id": "2"}]

    chunks_file = tmp_path / "chunks.pkl"
    embeddings_file = tmp_path / "embeddings.pkl"
    metadata_file = tmp_path / "metadata.pkl"

    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    with open(embeddings_file, "wb") as f:
        pickle.dump(embeddings, f)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    return str(chunks_file), str(embeddings_file), str(metadata_file)

# -------------------------------
# 2. Test initialization and lexical models
# -------------------------------

@patch("vector_store.chromadb.PersistentClient")
@patch("vector_store.genai.configure")
@patch("vector_store.genai.GenerativeModel")
def test_chroma_rag_init(mock_gen_model, mock_gen_configure, mock_client, setup_pickles):
    chunks_file, embeddings_file, metadata_file = setup_pickles

    mock_collection = MagicMock()
    mock_collection.get.return_value = {"documents": [], "metadatas": []}
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    rag = ChromaRAG(
        db_path="fake_db",
        collection_name="test_col",
        embeddings_file=embeddings_file,
        metadata_file=metadata_file,
        chunks_file=chunks_file
    )

    assert rag.docs
    assert rag.tfidf_vectorizer is not None
    assert rag.bm25 is not None
    mock_collection.add.assert_called()

# -------------------------------
# 3. Test hybrid retrieval
# -------------------------------

@patch("vector_store.GeminiEmbedder")
@patch("vector_store.ChromaRAG.expand_query")
def test_hybrid_retrieve(mock_expand_query, mock_embedder_class, setup_pickles):
    chunks_file, embeddings_file, metadata_file = setup_pickles

    # Mock expand query
    mock_expand_query.return_value = "expanded query"

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2]
    mock_embedder_class.return_value = mock_embedder

    # Initialize ChromaRAG
    with patch("vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"documents": ["doc1", "doc2"], "metadatas": [{"id": "1"}, {"id": "2"}]}
        mock_collection.query.return_value = {"documents": [["doc1", "doc2"]],
                                              "distances": [[0.1, 0.2]],
                                              "metadatas": [[{"id": "1"}, {"id": "2"}]]}
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        rag = ChromaRAG(
            db_path="fake_db",
            collection_name="test_col",
            embeddings_file=embeddings_file,
            metadata_file=metadata_file,
            chunks_file=chunks_file
        )

        top_docs, full_results = rag.hybrid_retrieve("test query", top_k=2)
        assert isinstance(top_docs, list)
        assert "doc1" in top_docs
        assert "documents" in full_results

# -------------------------------
# 4. Test reranking
# -------------------------------

@patch("vector_store.genai.GenerativeModel")
def test_rerank_llm(mock_gen_model, setup_pickles):
    mock_response = MagicMock()
    mock_response.text = "0.9"
    mock_gen_model.return_value.generate_content.return_value = mock_response

    with patch("vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        rag = ChromaRAG(
            db_path="fake_db",
            collection_name="test_col",
            embeddings_file="fake",
            metadata_file="fake",
            chunks_file=setup_pickles[0]
        )
        rag.gemini_model = mock_gen_model.return_value
        candidate_docs = ["doc1", "doc2"]
        ranked = rag.rerank("query", candidate_docs, method="llm")
        assert ranked == candidate_docs  # Mock returns same, LLM mocked

# -------------------------------
# 5. Test simple vector retrieval
# -------------------------------

@patch("vector_store.GeminiEmbedder")
def test_retrieve(mock_embedder_class, setup_pickles):
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2]
    mock_embedder_class.return_value = mock_embedder

    with patch("vector_store.chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"id": "1"}, {"id": "2"}]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        rag = ChromaRAG(
            db_path="fake_db",
            collection_name="test_col",
            embeddings_file="fake",
            metadata_file="fake",
            chunks_file=setup_pickles[0]
        )

        retrieved, full_results = rag.retrieve("query", top_k=2)
        assert "doc1" in retrieved
        assert "documents" in full_results
