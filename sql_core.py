from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings


def get_mysql_uri() -> str:
    """
    Build the SQLAlchemy MySQL URI from settings if MYSQL_URL is not explicitly set.
    """
    if settings.mysql_url:
        return settings.mysql_url

    return (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_db}"
    )


@dataclass
class SQLAnswer:
    sql: str
    rows: List[Dict[str, Any]]
    raw_result: Any


class SQLPipeline:
    """
    Simple Text-to-SQL pipeline on top of a MySQL database using LangChain.
    """

    def __init__(self, table: Optional[str] = None) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please configure it in your environment or .env file.")

        uri = get_mysql_uri()
        # Limit tables to the main Dubai hotels table for safety by default.
        include_tables = [table or settings.mysql_table]

        self.db = SQLDatabase.from_uri(uri, include_tables=include_tables)
        self.llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)

        # Prompt to generate SQL directly from schema + question
        self.sql_prompt = ChatPromptTemplate.from_template(
            """
          You are an expert MySQL analyst. Generate ONLY valid MySQL queries.

====================================================
DATE COLUMN RULE (CRITICAL)
====================================================
The table contains a REAL parsed DATE column:

    parsed_date_temp  (type: DATE)

There may also be a raw TEXT column `date` (format 'DD/MM/YYYY'), but
you should NOT use it for logic anymore.

Rules:
- ALWAYS use parsed_date_temp for ANY date filtering, comparison, grouping, ordering.
- NEVER invent or use columns like PARSED_DATE or PARSED_DATE_TEMP
  (only parsed_date_temp exists, with this exact lowercase name).
- To extract date parts, always use:
    YEAR(parsed_date_temp)
    MONTH(parsed_date_temp)
    DAY(parsed_date_temp)

====================================================
AGGREGATION RULES
====================================================
1) NEVER SUM structural fields  
   - Rooms_Available → ALWAYS use MAX(Rooms_Available)

2) Allowed to SUM or AVG (daily-changing fields):
   - Rooms_Sold
   - ADR
   - ADR_Competition
   - Occupancy
   - Occupancy_Competition
   - Revenue → (ADR * Rooms_Sold)

3) Do NOT SUM:
   - Occupancy
   - ADR
   - Percentages  
   (Use AVG instead.)

4) HOTEL LIST RULE  
   If question asks for hotel list or count:
       SELECT DISTINCT hotel_name

====================================================
MYSQL ONLY_FULL_GROUP_BY RULE
====================================================
In this MySQL server, sql_mode includes ONLY_FULL_GROUP_BY.

Therefore:
 - If you use ANY aggregate function (SUM, AVG, MAX, MIN, COUNT, etc.)
   together with non-aggregated columns in the SELECT list, you MUST add
   a GROUP BY clause.
 - In that GROUP BY, list EVERY non-aggregated column from SELECT.
 - Example:
       SELECT t1.parsed_date_temp, t1.occupancy, AVG(t2.occupancy) AS occupancy_last_year
       ...
   MUST have:
       GROUP BY t1.parsed_date_temp, t1.occupancy;
 - Do NOT select non-aggregated columns that are not listed in GROUP BY.

====================================================
EXTREME VALUE RULE (MAX/MIN/HIGHEST/BEST/WORST)
====================================================
If the question asks for:
- highest
- lowest
- best
- worst
- max
- min
- peak
- record
- strongest
- weakest
- busiest

Then you MUST:
1. Compute extreme value using a subquery.
2. Return ALL matching rows.
3. NEVER use LIMIT 1.
4. ALWAYS allow ties.

====================================================
YEAR-OVER-YEAR SAME-DAY RULE
====================================================
When user asks:
"same day last year"  
"what was 2024 value on that day"  
etc.

Then:
- Match rows using same DAY + same MONTH **for the same hotel**.
- Compare YEAR(parsed_date_temp) = X AND YEAR(parsed_date_temp) = X-1
- NEVER use aliases like PARSED_DATE in JOIN conditions.
- When comparing a metric like occupancy between years on the same day,
  aggregate the "last year" side with AVG(...) so there is exactly ONE
  value per day for that year (e.g. AVG(t2.occupancy) AS occupancy_last_year).

Pattern:
    JOIN <same_table_name> t2
    ON  t1.hotel_name = t2.hotel_name
    AND DAY(t1.parsed_date_temp) = DAY(t2.parsed_date_temp)
    AND MONTH(t1.parsed_date_temp) = MONTH(t2.parsed_date_temp)
    AND YEAR(t1.parsed_date_temp) = X
    AND YEAR(t2.parsed_date_temp) = X - 1

====================================================
SAFETY RULES
====================================================
- NEVER invent columns.
- NEVER guess table names.
- NEVER output explanations.
- Output ONLY a valid SQL query.

====================================================
SCHEMA
{schema}

====================================================
USER QUESTION
{question}

Write ONLY the SQL query:""".strip()
        )

    def ask_sql(self, question: str) -> SQLAnswer:
        """
        Generate SQL for a natural language question, execute it, and return the results.
        """
        # Get schema info for better SQL generation
        schema = self.db.get_table_info()

        # Ask LLM to generate SQL
        msg = self.sql_prompt.format(schema=schema, question=question)
        sql_query = self.llm.invoke(msg).content.strip()

        # Defensive cleanup: strip markdown fences like ```sql ... ``` while keeping the inner query.
        if "```" in sql_query:
            import re

            match = re.search(r"```(?:sql)?\s*(.*?)```", sql_query, re.IGNORECASE | re.DOTALL)
            if match:
                sql_query = match.group(1).strip()
            else:
                # Fallback: remove backticks only
                sql_query = sql_query.replace("```", "").strip()

        if not sql_query:
            raise ValueError("Generated SQL query was empty. Check the LLM prompt or question.")

        # Execute the SQL query
        raw_result = self.db.run(sql_query)

        # SQLDatabase.run may return a string representation; we keep it as-is and
        # also expose it in a rows-like field for convenience (best-effort parsing).
        rows: List[Dict[str, Any]] = []

        return SQLAnswer(sql=sql_query, rows=rows, raw_result=raw_result)



