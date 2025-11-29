## Hybrid RAG + SQL Analytics Assistant (LangChain + Python)

This project provides a Retrieval-Augmented Generation (RAG) and Text-to-SQL analytics assistant built with **LangChain**, using:

- **Chroma** as a local, persisted vector store for semantic search over documents/CSV rows
- **MySQL** (local or cloud) for exact numeric/aggregate queries on tabular data
- **OpenAI** models (configurable) for embeddings and chat completion

### 1. Setup

1. Create and activate a virtual environment (optional but recommended):

```bash
cd /Users/basitarif/Documents/vertikal-agent/Upwork_Project27Nov
uv venv .venv
source .venv/bin/activate  # on macOS/Linux
```

2. Install dependencies:

```bash
uv pip install -r requirements.txt
# For SQL support:
uv pip install pymysql sqlalchemy
```

### 2. Environment variables (`.env`)

Create a `.env` file in the project root with your configuration, for example:

```bash
# OpenAI configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Data & vector store
DATA_DIR=data
CHROMA_DIR=chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
K=4

# MySQL / SQL configuration (local or cloud)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=dubai_analytics
MYSQL_USER=root          # or rag_user
MYSQL_PASSWORD=your_password
MYSQL_TABLE=dubai_hotels_daily

# Optional: full SQLAlchemy URL; if set, it overrides the pieces above
# Example for local:
# MYSQL_URL=mysql+pymysql://root:your_password@localhost:3306/dubai_analytics
```

> By default, the code expects OpenAI models. You can adapt it to other providers if needed.

### 3. Add Data

Place your data under the `data/` directory (you can create subfolders if you want):

- `.txt`
- `.md`
- `.pdf`
- `.csv` (e.g. `dubai_hotels_synthetic_daily_2y_enriched.csv`)

The ingestion script will recursively load supported files for RAG, and `load_mysql.py` will load the Dubai CSV into MySQL.

### 4. Build Stores (Chroma + MySQL)

- **Load CSV data into MySQL** (using settings from `.env`):

```bash
uv run load_mysql.py
```

- **Build/update the Chroma vector store for RAG**:

```bash
uv run ingest.py
```

This will create/update a `chroma_db/` directory with the persisted vector store.

### 5. Ask Questions (CLI)

- **RAG-only CLI**:

```bash
uv run rag_cli.py "What is this project about?"
uv run rag_cli.py
```

- **Hybrid SQL + RAG CLI**:

```bash
uv run hybrid_cli.py "Example numeric question about your Dubai dataset"
uv run hybrid_cli.py
```

For the hybrid CLI, the output shows:

- `Route`: which path was used (`sql`, `rag`, or `sql+rag`)
- `SQL query`: the generated SQL when SQL is used
- `SQL raw result`: the direct result returned from MySQL
- `Answer`: the final natural-language answer

### 6. Evaluation (optional)

You can evaluate numeric accuracy of the hybrid pipeline by editing `evaluate_hybrid.py`:

1. Add realistic `EvalExample` entries (question, expected_value, tolerance) in `main()`.
2. Run:

```bash
uv run evaluate_hybrid.py
```

This prints accuracy metrics and writes detailed per-question results to `hybrid_eval_results.csv`.

### 7. Files Overview

- `requirements.txt` – Python dependencies
- `config.py` – Loads environment variables / model configuration
- `rag_core.py` – Core RAG pipeline (retriever + LLM chain)
- `sql_core.py` – Text-to-SQL pipeline on top of MySQL
- `hybrid_qa.py` – Hybrid router combining SQL and RAG answers
- `ingest.py` – Ingestion of documents/CSVs into Chroma
- `load_mysql.py` – Load Dubai CSV into MySQL
- `rag_cli.py` – RAG-only CLI interface
- `hybrid_cli.py` – Hybrid SQL + RAG CLI interface
- `evaluate_hybrid.py` – Optional evaluation script for numeric accuracy

