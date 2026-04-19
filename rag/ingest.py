"""
RAG Document Ingestion Pipeline.
Loads markdown documents, splits into chunks, embeds, and builds a FAISS index.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)

from typing import Optional

logger = logging.getLogger(__name__)


def _load_documents(docs_dir: Optional[Path] = None) -> list:
    """Load all markdown documents from the documents directory."""
    docs_dir = docs_dir or DOCUMENTS_DIR

    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from {docs_dir}")
    return documents


def _split_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(force: bool = False) -> FAISS:
    """
    Build the FAISS vector store from documents.

    Args:
        force: If True, rebuild even if index already exists.

    Returns:
        FAISS vector store instance.
    """
    index_path = VECTORSTORE_DIR / "index.faiss"

    if index_path.exists() and not force:
        logger.info("FAISS index already exists. Loading from disk.")
        return load_vectorstore()

    logger.info("Building FAISS vector store from documents...")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # Load and split
    documents = _load_documents()
    chunks = _split_documents(documents)

    # Embed and create index
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to disk
    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info(f"FAISS index saved to {VECTORSTORE_DIR}")

    return vectorstore


def load_vectorstore() -> FAISS:
    """
    Load an existing FAISS vector store from disk.

    Returns:
        FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index doesn't exist.
    """
    index_path = VECTORSTORE_DIR / "index.faiss"

    if not index_path.exists():
        logger.warning("FAISS index not found. Building from scratch...")
        return build_vectorstore(force=True)

    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS vector store loaded from disk.")
    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vs = build_vectorstore(force=True)
    print(f"Vector store built with {vs.index.ntotal} vectors")
