import sys

from rag_core import RAGPipeline


def run_interactive(rag: RAGPipeline) -> None:
    print("RAG CLI. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        answer = rag.ask(question)
        print(f"\nAnswer:\n{answer}")


def main() -> None:
    rag = RAGPipeline()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        answer = rag.ask(question)
        print(answer)
    else:
        run_interactive(rag)


if __name__ == "__main__":
    main()


