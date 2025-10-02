from __future__ import annotations
import os
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_openai import ChatOpenAI
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import CFG

SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".txt"}


def _loader_for(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path)
    if ext in {".html", ".htm"}:
        return UnstructuredHTMLLoader(path)
    if ext == ".txt":
        return TextLoader(path, autodetect_encoding=True)
    raise ValueError(f"Unsupported file type: {ext}")


def load_raw_documents(raw_dir: str) -> List[Document]:
    docs: List[Document] = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                loader = _loader_for(path)
                pages = loader.load()
                # Enrich metadata for better citation display (best-effort)
                if ext == ".pdf":
                    pages = enrich_pdf_metadata(path, pages)
                docs.extend(pages)
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG["CHUNK_SIZE"],
        chunk_overlap=CFG["CHUNK_OVERLAP"],
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def build_embeddings():
    return HuggingFaceEmbeddings(model_name=CFG["EMBED_MODEL"])


def build_or_load_vectorstore(chunks: List[Document] | None = None) -> FAISS:
    index_dir = CFG["INDEX_DIR"]
    os.makedirs(index_dir, exist_ok=True)
    faiss_path = os.path.join(index_dir, "faiss.index")
    store_path = os.path.join(index_dir, "index.pkl")

    if os.path.exists(faiss_path) and os.path.exists(store_path):
        return FAISS.load_local(
            index_dir, build_embeddings(), allow_dangerous_deserialization=True
        )

    assert chunks is not None and len(chunks) > 0, "No chunks provided to build index."
    vs = FAISS.from_documents(chunks, build_embeddings())
    vs.save_local(index_dir)
    return vs


def _llm():
    backend = CFG["LLM_BACKEND"].lower()
    if backend == "openai":
        return ChatOpenAI(model=CFG["OPENAI_MODEL"], temperature=0)
    if backend == "llama.cpp":
        model_path = CFG["LLAMACPP_MODEL_PATH"]
        if not model_path:
            raise ValueError("LLAMACPP_MODEL_PATH not set")
        return LlamaCpp(model_path=model_path, n_ctx=8192, temperature=0.0)
    if backend == "gpt4all":
        return GPT4All(model=CFG["GPT4ALL_MODEL"], n_ctx=4096, temp=0.0)
    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")


QA_PROMPT = PromptTemplate.from_template(
    """
You are MedGPT++, a careful healthcare assistant. Answer the user's question using ONLY the provided context.
- If the answer is not in context, say you don't know.
- Cite sources as [Author/Org, Year] with hyperlink when available.
- Keep answers concise and clinically cautious.

Context:
{context}

Question: {question}
Answer:
"""
)


def build_qa_chain(vs: FAISS):
    retriever = vs.as_retriever(
        search_type="similarity", search_kwargs={"k": CFG["TOP_K"]}
    )
    llm = _llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    return chain


def pretty_source(doc: Document) -> Dict[str, Any]:
    meta = doc.metadata or {}
    title = meta.get("title") or meta.get("source") or meta.get("file_path") or "Document"
    url = meta.get("source") if str(meta.get("source", "")).startswith("http") else None
    return {
        "title": title,
        "url": url,
        "snippet": doc.page_content[:300] + ("…" if len(doc.page_content) > 300 else ""),
        "score_hint": meta.get("score", None),
        "loc": {k: meta.get(k) for k in ["page", "start_index"] if k in meta},
    }


# ---------- Optional: nicer PDF metadata for citation panel ----------
def enrich_pdf_metadata(doc_path: str, pages: List[Document]) -> List[Document]:
    try:
        from pypdf import PdfReader  # lightweight import guard
    except Exception:
        return pages  # skip if pypdf not available

    try:
        meta_title, first_url = None, None
        reader = PdfReader(doc_path)
        if reader.metadata and reader.metadata.title:
            t = str(reader.metadata.title).strip()
            if t:
                meta_title = t

        try:
            first_txt = reader.pages[0].extract_text() or ""
            for tok in first_txt.split():
                if tok.startswith("http"):
                    first_url = tok.strip()
                    break
        except Exception:
            pass

        for d in pages:
            d.metadata = d.metadata or {}
            if meta_title and "title" not in d.metadata:
                d.metadata["title"] = meta_title
            if first_url:
                d.metadata["source"] = first_url
        return pages
    except Exception:
        return pages
