from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_store(chunks):
    return Chroma.from_documents(chunks, get_embeddings())


def add_to_store(store, chunks):
    store.add_documents(chunks)
    return store
