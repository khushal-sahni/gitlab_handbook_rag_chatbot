import os
import sys
import streamlit as st

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TOP_K, MIN_SCORE, FEEDBACK_CSV, CHROMA_DIR
from src.retriever import Retriever
from src.model_providers.embeddings import get_embedding_fn
from src.model_providers.chat import get_chat_fn
import csv
import datetime

st.set_page_config(page_title="GitLab Handbook Chatbot", page_icon="💬", layout="wide")

# Ensure feedback log exists
os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "question", "answer", "useful", "sources"])  # useful in {1,0,-1}

# Lazy init
@st.cache_resource(show_spinner=False)
def _boot():
    embed = get_embedding_fn()
    chat = get_chat_fn()
    retr = Retriever()
    return embed, chat, retr

embed, chat, retr = _boot()

st.title("GitLab Handbook & Direction Chatbot")
st.caption("Answers strictly from https://about.gitlab.com/handbook/ and /direction/. Sources cited.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content}

with st.sidebar:
    st.header("About")
    st.markdown("This is a weekend-built RAG bot for GitLab's public docs.")
    st.markdown("**Scope**: Handbook & Direction. If unsure, it will say so.")
    st.divider()
    st.markdown(f"Vector store: `{CHROMA_DIR}`")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask about GitLab's Handbook/Direction…")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # Embed query and retrieve
            q_emb = embed([q])[0]
            res = retr.query(q_emb, top_k=TOP_K)

            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            # Convert Chroma distance to a pseudo-similarity (depends on backend). Normalize gently.
            # If Chroma returns cosine distance, similarity ~ 1 - dist.
            sims = [1 - (d or 0) for d in dists]

            context_blocks = []
            sources = []
            for doc, meta, sim in zip(docs, metas, sims):
                url = meta.get("url", "")
                snippet = (doc[:300] + "…") if len(doc) > 300 else doc
                context_blocks.append(f"[source] {url}\n{snippet}")
                sources.append(url)

            # Guardrail: if everything below threshold, ask user to rephrase and show best matches
            if not sims or max(sims) < MIN_SCORE:
                msg = (
                    "I couldn't confidently find this in GitLab's Handbook/Direction. "
                    "Try rephrasing, or browse the sources below.\n\n" + "\n".join(f"- {u}" for u in sources)
                )
                st.markdown(msg)
                st.session_state.history.append({"role": "assistant", "content": msg})
            else:
                context = "\n\n".join(context_blocks)
                prompt = (
                    f"SYSTEM: You answer using only the context. If unknown, say so.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"USER: {q}\nASSISTANT:"
                )
                answer = chat([{"role": "user", "content": prompt}], temperature=0.2)

                # Render answer and sources
                st.markdown(answer)
                with st.expander("Sources"):
                    for i, (u, s) in enumerate(zip(sources, sims), start=1):
                        st.markdown(f"**{i}.** {u} — similarity ~ {s:.2f}")

                st.session_state.history.append({"role": "assistant", "content": answer})

                # Feedback UI
                c1, c2, c3 = st.columns([1,1,6])
                if c1.button("👍 Helpful", key=f"up_{len(st.session_state.history)}"):
                    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([datetime.datetime.utcnow().isoformat(), q, answer, 1, ";".join(sources)])
                    st.toast("Thanks for the feedback!")
                if c2.button("👎 Not helpful", key=f"down_{len(st.session_state.history)}"):
                    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([datetime.datetime.utcnow().isoformat(), q, answer, 0, ";".join(sources)])
                    st.toast("Logged. We'll improve retrieval.")
