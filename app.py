import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_and_split
from rag.store import add_documents
from rag.chain import get_answer

load_dotenv()

st.set_page_config(page_title="Chat with Docs", page_icon="📄")
st.title("📄 Chat with your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = None

with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Reading and indexing your document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name
            chunks = load_and_split(tmp_path)
            st.session_state.store = add_documents(chunks)
            st.session_state.messages = []
            os.unlink(tmp_path)
        st.success(f"Ready — {len(chunks)} chunks indexed.")

    if st.session_state.store and st.button("Clear"):
        st.session_state.store = None
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask something about your document..."):
    if st.session_state.store is None:
        st.warning("Upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(prompt, st.session_state.store)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
