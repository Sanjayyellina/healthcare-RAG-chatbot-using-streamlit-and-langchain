from __future__ import annotations
import time
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import CFG
from rag_pipeline import build_qa_chain, pretty_source, _llm
from trust import score_from_similarities, blend_trust, SELF_CHECK_PROMPT
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="MedGPT++", page_icon="🩺", layout="wide")

st.title("🩺 MedGPT++ — RAG Healthcare QA (MVP)")
st.caption("Research demo. Not medical advice.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K documents", 2, 8, CFG["TOP_K"])
    CFG["TOP_K"] = top_k
    backend = st.selectbox(
        "LLM backend",
        ["openai", "llama.cpp", "gpt4all"],
        index=["openai", "llama.cpp", "gpt4all"].index(CFG["LLM_BACKEND"]),
    )
    CFG["LLM_BACKEND"] = backend
    st.markdown(
        f"**Index dir:** `{CFG['INDEX_DIR']}`\n\n**Embed model:** `{CFG['EMBED_MODEL']}`"
    )

@st.cache_resource(show_spinner=True)
def _load_vs():
    return FAISS.load_local(
        CFG["INDEX_DIR"],
        HuggingFaceEmbeddings(model_name=CFG["EMBED_MODEL"]),
        allow_dangerous_deserialization=True,
    )

try:
    vs = _load_vs()
except Exception as e:
    st.error(f"Failed to load index: {e}. Run `python ingest.py` first.")
    st.stop()

qa = build_qa_chain(vs)

q = st.text_input("Ask a healthcare question (e.g., 'What are the key symptoms of measles?')")
ask = st.button("Ask")

if ask and q.strip():
    with st.spinner("Retrieving & generating…"):
        t0 = time.time()
        out = qa.invoke({"query": q})
        answer = out["result"].strip()
        sources: list[Document] = out.get("source_documents", [])

        # Best-effort similarity extraction (not all backends expose it)
        try:
            retriever = vs.as_retriever(search_kwargs={"k": CFG["TOP_K"]})
            hits = retriever.get_relevant_documents(q)
            sims = []
            for h in hits:
                sim = h.metadata.get("score")
                if sim is not None:
                    sims.append(float(sim))
        except Exception:
            sims = []

        sim_score = score_from_similarities(sims)

        llm = _llm()
        p = PromptTemplate.from_template(SELF_CHECK_PROMPT)
        sc_resp = llm.invoke(
            p.format(
                context="\n\n".join([s.page_content[:1000] for s in sources]),
                answer=answer,
            )
        )
        try:
            self_check_raw = int("".join([c for c in str(sc_resp) if c.isdigit()])[:1])
            self_check_raw = max(1, min(5, self_check_raw))
        except Exception:
            self_check_raw = 3

        trust = blend_trust(sim_score, self_check_raw)
        t1 = time.time()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Answer")
        st.write(answer)
        st.caption(
            f"⏱️ {t1 - t0:.2f}s | Trust Score: {trust:.2f} (sim={sim_score:.2f}, self={self_check_raw}/5)"
        )
    with col2:
        st.subheader("Citations")
        if not sources:
            st.info("No source documents returned.")
        else:
            for i, s in enumerate(sources, start=1):
                meta = pretty_source(s)
                title = meta["title"]
                url = meta["url"]
                snippet = meta["snippet"]
                loc = meta.get("loc", {})
                if url:
                    st.markdown(f"**[{i}. {title}]({url})**")
                else:
                    st.markdown(f"**{i}. {title}**")
                if loc:
                    st.caption(f"{loc}")
                st.write(snippet)
                st.divider()

    st.subheader("Disclaimer")
    st.caption(
        "This system is for research and education only and does not substitute professional medical advice."
    )
