# Chat with Docs

Upload a PDF and ask questions about it in plain English. Built with LangChain, ChromaDB, Groq, and Streamlit.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key (get one free at console.groq.com):

```bash
cp .env.example .env
# open .env and paste your GROQ_API_KEY
```

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a PDF, and start chatting.

## Stack

- **Streamlit** — UI
- **LangChain** — pipeline
- **ChromaDB** — vector database
- **Sentence Transformers** — local embeddings (free)
- **Groq / LLaMA 3** — LLM (free tier)
