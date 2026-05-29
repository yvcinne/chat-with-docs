import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


def get_answer(question: str, store) -> str:
    docs = store.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context provided.
If you cannot find the answer in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
""")

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
