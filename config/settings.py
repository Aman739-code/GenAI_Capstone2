"""
Global configuration and settings for the Solar Grid Optimization Agent.
Handles API keys, model configs, and system constants.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "rag" / "documents"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
DATA_DIR = PROJECT_ROOT / "data"

# ──────────────────────────────────────────────
# LLM Configuration
# ──────────────────────────────────────────────
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_RETRIES = 3

# ──────────────────────────────────────────────
# Embedding Configuration
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# RAG Configuration
# ──────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 4

# ──────────────────────────────────────────────
# Agent Configuration
# ──────────────────────────────────────────────
MAX_ITERATIONS = 10
NODE_TIMEOUT_SECONDS = 120

from typing import Optional

# ──────────────────────────────────────────────
# API Key Handling
# ──────────────────────────────────────────────
def get_api_key() -> Optional[str]:
    """
    Retrieve the Google API key from multiple sources.
    Priority: Streamlit secrets > Environment variable > None
    """
    # Try Streamlit secrets first
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    # Fall back to environment variable
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key

    return None
