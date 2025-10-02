##  MedGPT++ — Healthcare RAG Chatbot (Streamlit + LangChain)
---

Ask clinical questions. Get sourced, grounded answers.
MedGPT++ is a Retrieval-Augmented Generation (RAG) app that indexes your medical PDFs and answers questions with citations. Built with Streamlit, LangChain, FAISS, and your choice of OpenAI or local LLMs (llama.cpp / GPT4All).
<img width="1432" height="851" alt="Screenshot 2025-10-02 at 3 12 09 PM" src="https://github.com/user-attachments/assets/8e939777-5ba9-48f1-8044-86022f157542" />


## 
Built a healthcare RAG chatbot with Streamlit and LangChain that indexes medical PDFs and generates source-grounded answers using FAISS vector search and HuggingFace embeddings.
Implemented backend-agnostic LLM orchestration supporting OpenAI and local models (llama.cpp / GPT4All) via environment-based configuration.
Optimized retrieval quality with top-k tuning and chunking strategies; designed reproducible ingestion pipeline and excluded artifacts from VCS for clean CI/CD.
Addressed safety & reliability by surfacing citations, controlling temperature, and isolating secrets via .env.

Production RAG pipeline: ingestion → chunking → embeddings → vector search → answer synthesis → source grounding.
Backend-agnostic LLMs: switch between OpenAI and local models (air-gapped option).
Engineering for scale: FAISS for fast similarity search, environment-based config, reproducible ingestion.
Safety-aware: cites sources, configurable retrieval depth, PHI disclaimer.

---

##  Architecture

flowchart LR
    A[PDFs in data/raw] -->|ingest.py| B[Chunking & Cleaning]
    B --> C[Embeddings<br/>(HuggingFace)]
    C --> D[FAISS Index<br/>data/index/]
    E[User Question] --> F[Retriever (k=TOP_K)]
    F --> D
    D --> G[Relevant Chunks]
    G --> H[LLM (OpenAI / Llama.cpp / GPT4All)]
    H --> I[Answer + Citations]
    I --> J[Streamlit UI]

    
Query flow (high level)
Vector search over FAISS using dense embeddings.
Top-k chunks stuffed into an LLM prompt.
LLM generates a concise answer with inline source attributions.

---

## Configuration

All runtime settings come from .env. Example:


# LLM backend: openai | llama.cpp | gpt4all
LLM_BACKEND=openai

# OpenAI (if using cloud)
OPENAI_API_KEY=sk-REPLACE_ME
OPENAI_MODEL=gpt-4o-mini

# Embeddings
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval
TOP_K=4
TOKENIZERS_PARALLELISM=false

# Local options (only if you choose them)
LLAMACPP_MODEL_PATH=/absolute/path/to/Meta-Llama-3.1-8B-instruct.Q4_K_M.gguf
GPT4ALL_MODEL=gguf-or-gpt4all-model-file


Backend switch: change LLM_BACKEND and (optionally) model paths, then restart.
Data Ingestion

# Drop your PDFs here:
data/raw/
-└── who_guidelines.pdf
-└── nih_copd_review.pdf
-└── your_notes.pdf

# Then:
python ingest.py


# Index built and saved to data/index
What happens inside:
PDFs → text → cleaned → chunked
Embeddings via HuggingFace
FAISS index written to data/index/

# Run the App
streamlit run app.py

# In the UI:
Ask questions like:
“What are early red-flag symptoms of sepsis?”
“First-line therapy for community-acquired pneumonia?”
The answer includes citations from your PDFs.

---

## Tech Stack

Frontend: Streamlit

RAG: LangChain RetrievalQA + FAISS

Embeddings: HuggingFace (configurable)

LLMs:
OpenAI (default; plug in your API key)

llama.cpp (local GGUF models; set LLAMACPP_MODEL_PATH)

GPT4All (local; set GPT4ALL_MODEL)

Persistence: Local filesystem (data/index/)

Heads-up: HuggingFaceEmbeddings has a deprecation warning in recent LangChain. You can later move to langchain-huggingface with:

pip install -U langchain-huggingface and

from langchain_huggingface import HuggingFaceEmbeddings.

Project Structure

medgptpp_clean/
├── app.py                  
├── rag_pipeline.py        
├── ingest.py              
├── requirements.txt
├── .env.example            
├── data/
│   ├── raw/                
│   └── index/             
└── docs/
    └── screenshot.png      


---





