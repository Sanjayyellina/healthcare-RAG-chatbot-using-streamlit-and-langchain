
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

CFG = {
    "INDEX_DIR": os.getenv("INDEX_DIR", "data/index"),
    "RAW_DIR": os.getenv("RAW_DIR", "data/raw"),
    "EMBED_MODEL": os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    # LLM selection: one of ["openai", "llama.cpp", "gpt4all"]
    "LLM_BACKEND": os.getenv("LLM_BACKEND", "openai"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "LLAMACPP_MODEL_PATH": os.getenv("LLAMACPP_MODEL_PATH", ""),
    "GPT4ALL_MODEL": os.getenv("GPT4ALL_MODEL", "ggml-gpt4all-j-v1.3-groovy"),
    # Retrieval
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 900)),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 150)),
    "TOP_K": int(os.getenv("TOP_K", 4)),
    # Trust score weights
    "W_SIM": float(os.getenv("W_SIM", 0.6)),
    "W_SELF": float(os.getenv("W_SELF", 0.4)),
}
