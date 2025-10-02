
from __future__ import annotations
from config import CFG
from rag_pipeline import load_raw_documents, split_documents, build_or_load_vectorstore

if __name__ == "__main__":
    raw_dir = CFG["RAW_DIR"]
    print(f"Loading documents from {raw_dir} …")
    docs = load_raw_documents(raw_dir)
    print(f"Loaded {len(docs)} raw docs")

    print("Splitting into chunks …")
    chunks = split_documents(docs)
    print(f"Got {len(chunks)} chunks")

    print("Building FAISS index …")
    _ = build_or_load_vectorstore(chunks)
    print("✅ Index built and saved to", CFG["INDEX_DIR"])
