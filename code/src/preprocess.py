import os
import pickle
from typing import Tuple, List, Dict
from config import Config


def load_data(file_path: str) -> List[Dict]:
    """
    Load JSONL data (each line is a JSON object).
    """
    import json
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = []
        for line in lines:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON line: {e}")
                continue

        if not data:
            raise ValueError("No valid JSON objects found in input file.")

        return data

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks.
    """
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")
        if chunk_size <= 0 or overlap < 0:
            raise ValueError("chunk_size must be > 0 and overlap >= 0.")

        words = text.split()
        if not words:
            return []

        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break

        return chunks

    except ValueError as e:
        print(f"❌ Invalid parameters in chunk_text: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error during text chunking: {e}")
        return []


def prepare_chunks(
    input_file: str = "input_article_details.jsonl"
) -> Tuple[List[str], List[Dict]]:
    """
    Generate or load chunks + metadata for the dataset.
    Returns all_chunks, metadata_list.
    """
    try:
        # Load cached chunks if available
        if os.path.exists(Config.CHUNK_FILE) and os.path.exists(Config.METADATA_FILE):
            print("✅ Chunks and metadata already exist. Loading them...")
            with open(Config.CHUNK_FILE, "rb") as f:
                all_chunks = pickle.load(f)
            with open(Config.METADATA_FILE, "rb") as f:
                metadata_list = pickle.load(f)

            if not all_chunks or not metadata_list:
                raise ValueError("Loaded chunk or metadata files are empty.")

            return all_chunks, metadata_list

        # Generate new chunks
        print("No existing chunks found — generating new ones from", input_file)
        data = load_data(input_file)

        if not data:
            raise ValueError("No valid data loaded from input file.")

        all_chunks, metadata_list = [], []
        for article in data:
            try:
                contents = article.get("contents", "")
                chunks = chunk_text(contents)
                if not chunks:
                    continue

                all_chunks.extend(chunks)
                metadata_list.extend(
                    [{
                        "id": article.get("id", "unknown"),
                        "title": article.get("title", "untitled"),
                        "url": article.get("url", "")
                    }] * len(chunks)
                )
            except Exception as e:
                print(f"⚠️ Skipping problematic article: {e}")
                continue

        if not all_chunks:
            raise ValueError("No chunks were generated from the dataset.")

        # Save for future runs
        with open(Config.CHUNK_FILE, "wb") as f:
            pickle.dump(all_chunks, f)
        with open(Config.METADATA_FILE, "wb") as f:
            pickle.dump(metadata_list, f)

        print(f"✅ Generated {len(all_chunks)} chunks and saved to {Config.CHUNK_FILE}.")
        return all_chunks, metadata_list

    except (pickle.PickleError, EOFError) as e:
        print(f"❌ Pickle file error: {e}")
        return [], []
    except ValueError as e:
        print(f"❌ Data validation error: {e}")
        return [], []
    except Exception as e:
        print(f"❌ Unexpected error in prepare_chunks: {e}")
        return [], []
