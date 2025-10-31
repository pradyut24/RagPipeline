import json
import pickle
import pytest
from tempfile import NamedTemporaryFile, TemporaryDirectory

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../src")))

from preprocess import load_data, chunk_text, prepare_chunks
from config import Config

# -------------------------------
# 1. Tests for load_data
# -------------------------------

def test_load_data_valid_json(tmp_path):
    # Create a temporary JSONL file
    data = [{"id": 1, "contents": "Test content"}]
    file_path = tmp_path / "test.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    loaded = load_data(str(file_path))
    assert isinstance(loaded, list)
    assert loaded == data

def test_load_data_invalid_json(tmp_path):
    # File with one invalid JSON line
    file_path = tmp_path / "invalid.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write('{"id": 1, "contents": "ok"}\n')
        f.write('invalid json line\n')
    loaded = load_data(str(file_path))
    assert len(loaded) == 1
    assert loaded[0]["id"] == 1

def test_load_data_file_not_found():
    loaded = load_data("non_existent_file.jsonl")
    assert loaded == []

# -------------------------------
# 2. Tests for chunk_text
# -------------------------------

def test_chunk_text_basic():
    text = "word " * 100
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) > 0
    # Check that overlap exists
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert first_chunk_words[-2:] == second_chunk_words[:2]

def test_chunk_text_invalid_input():
    assert chunk_text("") == []
    assert chunk_text(None) == []
    assert chunk_text("test", chunk_size=0) == []

# -------------------------------
# 3. Tests for prepare_chunks
# -------------------------------

def test_prepare_chunks_generates(tmp_path, monkeypatch):
    # Create a small input JSONL file
    input_file = tmp_path / "input.jsonl"
    articles = [{"id": "a1", "title": "Title1", "contents": "This is a test article."}]
    with open(input_file, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")

    # Use a temporary directory for chunk & metadata files
    temp_dir = tmp_path / "cache"
    temp_dir.mkdir()
    monkeypatch.setattr(Config, "CHUNK_FILE", str(temp_dir / "chunks.pkl"))
    monkeypatch.setattr(Config, "METADATA_FILE", str(temp_dir / "metadata.pkl"))

    chunks, metadata = prepare_chunks(str(input_file))
    assert len(chunks) > 0
    assert len(metadata) == len(chunks)
    assert metadata[0]["id"] == "a1"

def test_prepare_chunks_loads_existing(tmp_path, monkeypatch):
    temp_dir = tmp_path / "cache"
    temp_dir.mkdir()
    chunk_file = temp_dir / "chunks.pkl"
    metadata_file = temp_dir / "metadata.pkl"

    saved_chunks = ["chunk1", "chunk2"]
    saved_metadata = [{"id": "x"}, {"id": "y"}]

    with open(chunk_file, "wb") as f:
        pickle.dump(saved_chunks, f)
    with open(metadata_file, "wb") as f:
        pickle.dump(saved_metadata, f)

    monkeypatch.setattr(Config, "CHUNK_FILE", str(chunk_file))
    monkeypatch.setattr(Config, "METADATA_FILE", str(metadata_file))

    chunks, metadata = prepare_chunks("dummy_input.jsonl")
    assert chunks == saved_chunks
    assert metadata == saved_metadata
