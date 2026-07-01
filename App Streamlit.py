"""
app_streamlit.py
-----------------
Web UI for the Power Systems RAG assistant:
  - Upload documents (PDF/TXT/MD) and build the knowledge base, directly in the browser.
  - Chat with the assistant, grounded in your uploaded material.
  - Browse and reopen previous chat sessions (persisted locally in chat_history.db).

Usage:
    streamlit run app_streamlit.py
"""
from pathlib import Path

import streamlit as st

import config
import knowledge_base as kb
import chat_history as ch
from rag_engine import PowerSystemsRAG, ConversationState

st.set_page_config(page_title="Power Systems RAG Assistant", page_icon="⚡", layout="wide")

config.DATA_DIR.mkdir(exist_ok=True)
config.DB_DIR.mkdir(exist_ok=True)

# ---------------- Session state setup ----------------
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None  # None = new, unsaved conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_state" not in st.session_state:
    st.session_state.conv_state = ConversationState()
if "rag" not in st.session_state:
    st.session_state.rag = None
if "rag_error" not in st.session_state:
    st.session_state.rag_error = None


def start_new_chat():
    st.session_state.current_session_id = None
    st.session_state.messages = []
    st.session_state.conv_state = ConversationState()


def load_chat(session_id: str):
    st.session_state.current_session_id = session_id
    stored = ch.get_messages(session_id)
    st.session_state.messages = stored
    # Rebuild the lightweight conversation memory (plain Q/A, no context blobs) for follow-ups
    conv = ConversationState()
    for m in stored:
        conv.add(m["role"], m["content"])
    st.session_state.conv_state = conv


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚡ Power Systems RAG")

    if st.button("➕ New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.subheader("💬 Previous chats")
    sessions = ch.list_sessions()
    if not sessions:
        st.caption("No previous chats yet — ask something to start one.")
    else:
        for s in sessions:
            is_active = s["id"] == st.session_state.current_session_id
            cols = st.columns([5, 1])
            with cols[0]:
                if st.button(
                    ("🟢 " if is_active else "") + (s["title"] or "Untitled"),
                    key=f"session_{s['id']}",
                    use_container_width=True,
                ):
                    load_chat(s["id"])
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", key=f"delete_{s['id']}"):
                    ch.delete_session(s["id"])
                    if st.session_state.current_session_id == s["id"]:
                        start_new_chat()
                    st.rerun()

    st.markdown("---")
    st.subheader("📚 Knowledge base")
    chunk_count = kb.collection_chunk_count()
    indexed_sources = kb.list_indexed_sources()

    if indexed_sources:
        st.success(f"{len(indexed_sources)} document(s) indexed, {chunk_count} chunks total")
        with st.expander("Indexed documents"):
            for src in indexed_sources:
                st.write(f"- {src}")
    else:
        st.info("No documents indexed yet. Upload files below.")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(f"📥 Build knowledge base from {len(uploaded_files)} file(s)", type="primary"):
            progress = st.progress(0, text="Starting...")
            collection = kb.get_collection()
            total_chunks = 0
            for i, uploaded in enumerate(uploaded_files):
                dest = Path(config.DATA_DIR) / uploaded.name
                dest.write_bytes(uploaded.getbuffer())
                progress.progress(i / len(uploaded_files), text=f"Reading {uploaded.name}...")
                docs = kb.load_documents(config.DATA_DIR)
                matching = [d for d in docs if d[0] == uploaded.name]
                if matching:
                    _, text = matching[0]
                    n = kb.ingest_file(uploaded.name, text, collection=collection)
                    total_chunks += n
                progress.progress((i + 1) / len(uploaded_files), text=f"Indexed {uploaded.name}")

            st.success(f"Done! Indexed {total_chunks} chunks across {len(uploaded_files)} file(s).")
            st.session_state.rag = None
            st.rerun()

    st.markdown("---")
    top_k = st.slider("Chunks to retrieve per question", min_value=1, max_value=10, value=config.TOP_K)

# ---------------- Main chat area ----------------
st.title("⚡ Power Systems Engineering RAG Assistant")
st.caption(
    "Upload your power systems notes and textbooks, build a knowledge base, and ask "
    "grounded, cited questions. Previous chats are saved automatically."
)

if kb.collection_chunk_count() == 0:
    st.warning("👈 Upload at least one document in the sidebar and build the knowledge base before asking questions.")
    st.stop()

if st.session_state.rag is None:
    try:
        st.session_state.rag = PowerSystemsRAG()
        st.session_state.rag_error = None
    except RuntimeError as e:
        st.session_state.rag_error = str(e)

if st.session_state.rag_error:
    st.error(st.session_state.rag_error)
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- **{s['source']}** — chunk {s['chunk_index']} (relevance {s['relevance']})")

question = st.chat_input("Ask a power systems engineering question...")

if question:
    # Lazily create a persisted session on the first message of a new chat
    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = ch.create_session(question)

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    ch.add_message(st.session_state.current_session_id, "user", question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & thinking..."):
            try:
                result = st.session_state.rag.ask(question, st.session_state.conv_state, top_k=top_k)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.write(f"- **{s['source']}** — chunk {s['chunk_index']} (relevance {s['relevance']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
    ch.add_message(
        st.session_state.current_session_id, "assistant", result["answer"], result["sources"]
    )
    st.rerun()
