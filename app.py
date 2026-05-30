import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_and_split
from rag.store import create_store, add_to_store
from rag.chain import stream_answer

load_dotenv()

st.set_page_config(page_title="Chat with Docs", page_icon="📄")
st.title("📄 Chat with your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = None
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []

# --- Sidebar ---
with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    new_files = [
        f for f in (uploaded_files or [])
        if f.name not in st.session_state.loaded_files
    ]

    if new_files and st.button("Process", type="primary"):
        with st.spinner(f"Indexing {len(new_files)} file(s)..."):
            all_chunks = []
            for uf in new_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uf.read())
                    tmp_path = f.name
                chunks = load_and_split(tmp_path)
                for chunk in chunks:
                    chunk.metadata["filename"] = uf.name
                all_chunks.extend(chunks)
                os.unlink(tmp_path)

            if st.session_state.store is None:
                st.session_state.store = create_store(all_chunks)
            else:
                add_to_store(st.session_state.store, all_chunks)

            for uf in new_files:
                st.session_state.loaded_files.append(uf.name)

        st.success(f"Indexed {len(all_chunks)} chunks.")

    if st.session_state.loaded_files:
        st.divider()
        st.caption("Loaded files")
        for name in st.session_state.loaded_files:
            st.markdown(f"- {name}")

        if st.button("Clear all"):
            st.session_state.store = None
            st.session_state.messages = []
            st.session_state.loaded_files = []
            st.rerun()

# --- Chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.caption(f"**{src['file']}** — page {src['page'] + 1}")
                    text = src["text"]
                    st.markdown(f"> {text[:400]}{'...' if len(text) > 400 else ''}")

if prompt := st.chat_input("Ask something about your document(s)..."):
    if st.session_state.store is None:
        st.warning("Upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        history = st.session_state.messages[:-1]
        token_stream, docs = stream_answer(prompt, st.session_state.store, history)

        with st.chat_message("assistant"):
            answer = st.write_stream(token_stream)

            seen = set()
            sources = []
            for doc in docs:
                file = doc.metadata.get("filename") or os.path.basename(
                    doc.metadata.get("source", "unknown")
                )
                page = doc.metadata.get("page", 0)
                if (file, page) not in seen:
                    seen.add((file, page))
                    sources.append({"file": file, "page": page, "text": doc.page_content})

            if sources:
                with st.expander("Sources", expanded=False):
                    for src in sources:
                        st.caption(f"**{src['file']}** — page {src['page'] + 1}")
                        text = src["text"]
                        st.markdown(f"> {text[:400]}{'...' if len(text) > 400 else ''}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
