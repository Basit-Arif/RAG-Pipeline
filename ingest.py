from rag_core import RAGPipeline


def main() -> None:
    rag = RAGPipeline()
    rag.ingest()


if __name__ == "__main__":
    main()


