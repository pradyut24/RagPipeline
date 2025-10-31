import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../src")))
import pickle
import pytest
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import MagicMock, patch
from embedder import GeminiEmbedder


# -------------------------------
# 1. Tests for _split_large_text
# -------------------------------

def test_split_large_text_small():
    embedder = GeminiEmbedder()
    text = "small text"
    result = embedder._split_large_text(text)
    assert result == [text]


def test_split_large_text_large():
    embedder = GeminiEmbedder(max_bytes=10)
    text = "abcdefghij" * 5  # 50 bytes
    parts = embedder._split_large_text(text)
    assert len(parts) > 1
    assert "".join(parts).replace("\x00", "")[:50] in text


def test_split_large_text_invalid_input():
    embedder = GeminiEmbedder()
    parts = embedder._split_large_text(None)
    assert parts == [None]


# -------------------------------
# 2. Tests for checkpoint handling
# -------------------------------

def test_load_checkpoint_empty(tmp_path):
    checkpoint_file = tmp_path / "checkpoint.pkl"
    embedder = GeminiEmbedder(checkpoint_path=str(checkpoint_file))
    checkpoint = embedder._load_checkpoint()
    assert checkpoint == {"embeddings": [], "completed_batches": 0}


def test_save_and_load_checkpoint(tmp_path):
    checkpoint_file = tmp_path / "checkpoint.pkl"
    embedder = GeminiEmbedder(checkpoint_path=str(checkpoint_file))

    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    completed_batches = 1
    embedder._save_checkpoint(embeddings, completed_batches)

    loaded = embedder._load_checkpoint()
    assert loaded["embeddings"] == embeddings
    assert loaded["completed_batches"] == completed_batches


# -------------------------------
# 3. Tests for embed_documents with mock
# -------------------------------


@patch("embedder.genai.Client")
def test_embed_documents_success(mock_client_class, tmp_path):
    # Mock Gemini API response
    mock_client = MagicMock()

    # This function will return one embedding per document in batch
    def mock_embed_content(model, contents):
        return MagicMock(embeddings=[MagicMock(values=[0.1, 0.2]) for _ in contents])

    mock_client.models.embed_content.side_effect = mock_embed_content
    mock_client_class.return_value = mock_client

    checkpoint_file = tmp_path / "checkpoint.pkl"
    embedder = GeminiEmbedder(checkpoint_path=str(checkpoint_file), batch_size=2)

    documents = ["doc1", "doc2"]
    embeddings = embedder.embed_documents(documents)

    assert embeddings == [[0.1, 0.2], [0.1, 0.2]]  # Each document gets a mocked embedding


@patch("embedder.genai.Client")
def test_embed_documents_invalid_input(mock_client_class):
    mock_client_class.return_value = MagicMock()
    embedder = GeminiEmbedder()
    result = embedder.embed_documents("not a list")
    assert result == []


# -------------------------------
# 4. Tests for embed_query with mock
# -------------------------------

@patch("embedder.genai.Client")
def test_embed_query_success(mock_client_class):
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.embeddings = [MagicMock(values=[0.5, 0.6])]
    mock_client.models.embed_content.return_value = mock_result
    mock_client_class.return_value = mock_client

    embedder = GeminiEmbedder()
    result = embedder.embed_query("test query")
    assert result == [0.5, 0.6]


@patch("embedder.genai.Client")
def test_embed_query_invalid(mock_client_class):
    mock_client_class.return_value = MagicMock()
    embedder = GeminiEmbedder()
    result = embedder.embed_query("")
    assert result == []
