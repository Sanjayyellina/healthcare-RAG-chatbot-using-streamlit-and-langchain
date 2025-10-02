from __future__ import annotations
import os

def test_index_exists():
    assert os.path.exists("data/index/faiss.index"), "Run: python ingest.py"

