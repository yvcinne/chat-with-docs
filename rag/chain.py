import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


def _build(question: str, store, history: list[dict]):
    docs = store.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    lc_history = []
    for msg in (history or []):
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant. Answer the question based only on the context below.\n"
            "If the answer is not in the context, say \"I don't have enough information to answer that.\"\n\n"
            "Context:\n{context}"
        )),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm
    inputs = {"context": context, "history": lc_history, "question": question}
    return chain, inputs, docs


def stream_answer(question: str, store, history: list[dict] = None):
    chain, inputs, docs = _build(question, store, history)
    return (chunk.content for chunk in chain.stream(inputs) if chunk.content), docs
