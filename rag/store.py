from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def add_documents(chunks):
    embeddings = get_embeddings()
    store = Chroma.from_documents(chunks, embeddings)
    return store
