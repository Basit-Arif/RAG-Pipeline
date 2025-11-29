## RAG Pipeline (LangChain + Python)

This project provides a minimal Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, using:

- **Chroma** as a local, persisted vector store
- **OpenAI** models (configurable) for embeddings and chat completion

### 1. Setup

1. Create and activate a virtual environment (optional but recommended):

```bash
cd /Users/basitarif/Documents/vertikal-agent/Upwork_Project27Nov
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your model configuration, for example:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

> By default, the code expects OpenAI models. You can adapt it to other providers if needed.

### 2. Add Documents

Place your documents under the `data/` directory (you can create subfolders if you want):

- `.txt`
- `.md`
- `.pdf`

The ingest script will recursively load supported files.

### 3. Build the Vector Store (Ingest)

Run the ingest script to load, split, embed, and persist your documents into a local Chroma DB:

```bash
python ingest.py
```

This will create a `chroma_db/` directory with the persisted vector store.

### 4. Ask Questions (RAG)

Use the RAG script to query your knowledge base:

```bash
python rag_cli.py "What is this project about?"
```

Or run it with no arguments and it will start an interactive loop:

```bash
python rag_cli.py
```

### 5. Files Overview

- `requirements.txt` – Python dependencies
- `config.py` – Loads environment variables / model configuration
- `ingest.py` – One-time or repeatable ingestion of documents into Chroma
- `rag_core.py` – Core RAG pipeline (retriever + LLM chain)
- `rag_cli.py` – Simple CLI interface for asking questions


