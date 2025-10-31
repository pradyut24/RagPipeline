import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../src")))

import pickle
from preprocess import load_data, chunk_text, prepare_chunks
from config import Config


def main():
    """
    Main entry point for running the RAG pipeline.
    Generates chunks if they don't exist and prints summary.
    """
    try:
        data_path = getattr(Config, "DATA_PATH", None)
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError("❌ DATA_PATH not found in Config or file missing.")

        # Prepare chunks and metadata
        all_chunks, metadata = prepare_chunks(data_path)
        print(f"✅ Loaded {len(all_chunks)} chunks and {len(metadata)} metadata entries.")

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Unexpected error in main(): {e}")


if __name__ == "__main__":
    main()
