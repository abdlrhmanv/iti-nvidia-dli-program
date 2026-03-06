"""
Initialize (or reset) the ChromaDB vector store.

Usage (from project root):
    python -m scripts.init_vectordb          # create / verify the collection
    python -m scripts.init_vectordb --reset  # delete existing data and recreate
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def init_vectordb(reset: bool = False) -> None:
    if reset and config.VECTORSTORE_DIR.exists():
        shutil.rmtree(config.VECTORSTORE_DIR)
        config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared existing vector store at %s", config.VECTORSTORE_DIR)

    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
    )

    logger.info(
        "Collection '%s' ready  ─  %d documents stored",
        config.CHROMA_COLLECTION_NAME,
        collection.count(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the vector store")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing vector store data before initializing",
    )
    args = parser.parse_args()
    init_vectordb(reset=args.reset)
