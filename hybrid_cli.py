import sys

from hybrid_qa import HybridQAPipeline


def run_interactive() -> None:
    qa = HybridQAPipeline()
    print("Hybrid SQL + RAG CLI. Type 'exit' or 'quit' to stop.")

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

        result = qa.ask(question)
        print(f"\n[Route: {result.route}]")
        if result.sql_query:
            print(f"SQL query: {result.sql_query}")
        if result.sql_raw_result is not None:
            print(f"SQL raw result: {result.sql_raw_result}")
        print(f"\nAnswer:\n{result.answer}")


def main() -> None:
    qa = HybridQAPipeline()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        result = qa.ask(question)
        print(f"[Route: {result.route}]")
        if result.sql_query:
            print(f"SQL query: {result.sql_query}")
        if result.sql_raw_result is not None:
            print(f"SQL raw result: {result.sql_raw_result}")
        print(f"\nAnswer:\n{result.answer}")
    else:
        # Reuse interactive loop but with the same QA instance
        print("Hybrid SQL + RAG CLI. Type 'exit' or 'quit' to stop.")
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

            result = qa.ask(question)
            print(f"\n[Route: {result.route}]")
            if result.sql_query:
                print(f"SQL query: {result.sql_query}")
            if result.sql_raw_result is not None:
                print(f"SQL raw result: {result.sql_raw_result}")
            print(f"\nAnswer:\n{result.answer}")


if __name__ == "__main__":
    main()


