"""
Simple evaluation script to test the hybrid SQL + RAG pipeline on a set of
numeric questions, using MySQL as the ground truth source of numbers.

This is a skeleton you can extend with a proper eval_set.csv if desired.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from hybrid_qa import HybridQAPipeline


@dataclass
class EvalExample:
    question: str
    expected_value: float
    tolerance: float  # relative tolerance, e.g. 0.05 for 5%


def extract_first_number(text: str) -> Optional[float]:
    import re

    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def evaluate(examples: List[EvalExample]) -> None:
    qa = HybridQAPipeline()

    total = len(examples)
    correct = 0
    per_example = []

    for ex in examples:
        result = qa.ask(ex.question)
        pred_val = extract_first_number(result.answer)

        if pred_val is None:
            is_correct = False
            rel_error = None
        else:
            rel_error = abs(pred_val - ex.expected_value) / max(abs(ex.expected_value), 1e-9)
            is_correct = rel_error <= ex.tolerance

        if is_correct:
            correct += 1

        per_example.append(
            {
                "question": ex.question,
                "expected": ex.expected_value,
                "predicted": pred_val,
                "relative_error": rel_error,
                "correct": is_correct,
                "route": result.route,
                "sql_query": result.sql_query,
            }
        )

    accuracy = correct / total if total else 0.0
    print(f"Evaluated {total} examples.")
    print(f"Accuracy within tolerance: {accuracy * 100:.2f}%")

    df = pd.DataFrame(per_example)
    df.to_csv("hybrid_eval_results.csv", index=False)
    print("Saved detailed results to hybrid_eval_results.csv")


def main() -> None:
    # Placeholder examples – you should replace these with real,
    # dataset-specific questions and ground-truth values.
    examples = [
        # EvalExample(
        #     question="What is the total revenue for hotel XYZ on 2023-01-01?",
        #     expected_value=12345.67,
        #     tolerance=0.05,
        # ),
    ]

    if not examples:
        print("No evaluation examples defined yet in evaluate_hybrid.py.")
        print("Edit the 'examples' list in main() to add realistic questions and ground truth.")
        return

    evaluate(examples)


if __name__ == "__main__":
    main()


