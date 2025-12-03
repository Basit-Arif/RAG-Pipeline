"""
Simple evaluation script to test the hybrid SQL + RAG pipeline on a set of
numeric questions, using MySQL as the ground truth source of numbers.

This is a skeleton you can extend with a proper eval_set.csv if desired.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def evaluate(examples: List[EvalExample]) -> Tuple[float, List[dict]]:
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

    return accuracy, per_example


def main() -> None:
    # Dataset-specific evaluation questions with numeric ground truth.
    #
    # IMPORTANT:
    # - expected_value is the *first* number we expect the model to output
    #   in its natural-language answer (see extract_first_number()).
    # - tolerance is relative (e.g. 0.02 → ±2%).
    examples = [
        # Simple, single-row lookups (easy)
        EvalExample(
            question="What was the occupancy percentage for St Regis Dubai on 1 January 2025?",
            expected_value=85.8,  # from CSV row: 01/01/2025, Occupancy_% = 85.8
            tolerance=0.001,
        ),
        EvalExample(
            question="What was the ADR for St Regis Dubai on 1 January 2025?",
            expected_value=1160.33,  # from CSV row: 01/01/2025, ADR = 1160.33
            tolerance=0.001,
        ),

        # Harder aggregate questions (extremes / year-over-year logic)
        EvalExample(
            question="In 2025, which day did St Regis Dubai have the highest occupancy, and what was that occupancy percentage?",
            expected_value=95.4,  # from SQL: max 2025 occupancy for St Regis Dubai
            tolerance=0.001,
        ),
        EvalExample(
            question="When did Premier Inn Al Furjan have the highest occupancy in 2025, and what was the occupancy percentage?",
            expected_value=97.0,  # from SQL: multiple 2025 dates with 97% occupancy
            tolerance=0.001,
        ),
    ]

    if not examples:
        print("No evaluation examples defined yet in evaluate_hybrid.py.")
        print("Edit the 'examples' list in main() to add realistic questions and ground truth.")
        return

    evaluate(examples)


if __name__ == "__main__":
    main()


