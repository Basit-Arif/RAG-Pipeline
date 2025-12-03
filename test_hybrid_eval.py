import re

import pytest

from evaluate_hybrid import EvalExample, evaluate
from hybrid_qa import HybridQAPipeline


# Reuse the same basic numeric examples used in evaluate_hybrid.main()
EVAL_EXAMPLES = [
    EvalExample(
        question="What was the occupancy percentage for St Regis Dubai on 1 January 2025?",
        expected_value=85.8,
        tolerance=0.001,
    ),
    EvalExample(
        question="What was the ADR for St Regis Dubai on 1 January 2025?",
        expected_value=1160.33,
        tolerance=0.001,
    ),
    EvalExample(
        question="In 2025, which day did St Regis Dubai have the highest occupancy, and what was that occupancy percentage?",
        expected_value=95.4,
        tolerance=0.001,
    ),
    EvalExample(
        question="When did Premier Inn Al Furjan have the highest occupancy in 2025, and what was the occupancy percentage?",
        expected_value=97.0,
        tolerance=0.001,
    ),
]


def test_hybrid_numeric_accuracy():
    """End-to-end evaluation of a small numeric test set using pytest."""
    accuracy, per_example = evaluate(EVAL_EXAMPLES)

    # Require perfect accuracy on this small, curated set.
    assert accuracy == pytest.approx(1.0, rel=1e-6)

    # Ensure each example produced a numeric prediction.
    for row in per_example:
        assert row["predicted"] is not None, f"No numeric answer for: {row['question']}"


def test_yoy_adr_sql_structure():
    """
    Hard technical case:
    Check that the YoY ADR question generates a reasonable SQL with AVG(ADR)
    and year-based grouping, without needing to know the exact numeric answer.
    """
    qa = HybridQAPipeline()
    question = (
        "Which hotel had the strongest year-on-year ADR improvement from 2024 "
        "to 2025, and what were its average ADRs in each year?"
    )
    result = qa.ask(question)

    # Should route through SQL or SQL+RAG
    assert result.route in {"sql", "sql+rag"}
    assert result.sql_query is not None

    sql = result.sql_query.lower()
    assert "avg(adr" in sql
    assert "year(parsed_date_temp)" in sql
    # Query should touch both years
    assert "2024" in sql and "2025" in sql


def test_same_day_last_year_join_uses_parsed_date_temp_and_hotel():
    """
    Hard technical case:
    For a 'same day last year' type question, ensure the generated SQL
    joins on both hotel_name and parsed_date_temp parts.
    """
    qa = HybridQAPipeline()
    question = (
        "In 2025, which day did St Regis Dubai have the highest occupancy, "
        "and what was the occupancy on the same day in 2024?"
    )
    result = qa.ask(question)

    assert result.route in {"sql", "sql+rag"}
    assert result.sql_query is not None

    sql = result.sql_query
    # Should join the table to itself
    assert "join" in sql.lower()
    # Should use hotel_name in the join or where conditions
    assert "hotel_name" in sql
    # Should use parsed_date_temp with DAY/MONTH and YEAR logic
    assert re.search(r"day\\s*\\(.*parsed_date_temp", sql, re.IGNORECASE)
    assert re.search(r"month\\s*\\(.*parsed_date_temp", sql, re.IGNORECASE)
    assert "year(parsed_date_temp)" in sql.lower()


def test_revenue_uses_adr_times_rooms_sold():
    """
    Hard technical case:
    Ensure revenue questions use ADR * Rooms_Sold rather than a non-existent
    Revenue column.
    """
    qa = HybridQAPipeline()
    question = (
        "What was the total room revenue for St Regis Dubai in 2025 "
        "based on ADR times rooms sold?"
    )
    result = qa.ask(question)

    assert result.route in {"sql", "sql+rag"}
    assert result.sql_query is not None

    sql = result.sql_query.lower()
    # Must use adr * rooms_sold somewhere
    assert "adr * rooms_sold" in sql or "rooms_sold * adr" in sql
    # Should not reference a fake revenue column
    assert "revenue" not in sql


def test_highest_occupancy_query_does_not_limit_ties():
    """
    Hard technical case:
    For extreme-value occupancy questions, ensure the SQL does not use LIMIT 1,
    so that ties (multiple max-occupancy dates) are preserved.
    """
    qa = HybridQAPipeline()
    question = "When did Premier Inn Al Furjan have the highest occupancy in 2025?"
    result = qa.ask(question)

    assert result.route in {"sql", "sql+rag"}
    assert result.sql_query is not None

    sql = result.sql_query.lower()
    # Should use a MAX(occupancy) subquery
    assert "max(occupancy" in sql
    # Should not arbitrarily truncate with LIMIT 1
    assert "limit 1" not in sql


