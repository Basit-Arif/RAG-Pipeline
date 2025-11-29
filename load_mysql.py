"""
Helper script to load the Dubai hotels CSV into a local MySQL database.

Usage (after configuring MySQL and .env):

    uv run load_mysql.py
"""

import pandas as pd
from sqlalchemy import create_engine

from config import settings


def get_mysql_url() -> str:
    """
    Build the SQLAlchemy MySQL URL from settings if MYSQL_URL is not explicitly set.
    """
    if settings.mysql_url:
        return settings.mysql_url

    return (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_db}"
    )


def main() -> None:
    csv_path = f"{settings.data_dir}/dubai_hotels_synthetic_daily_2y_enriched.csv"
    print(f"Loading CSV from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV.")

    mysql_url = get_mysql_url()
    print(f"Connecting to MySQL at: {mysql_url}")

    engine = create_engine(mysql_url)
    table_name = settings.mysql_table

    # if_exists="replace" so you can rerun this script to refresh data
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Wrote {len(df)} rows into table '{table_name}' in database '{settings.mysql_db}'.")


if __name__ == "__main__":
    main()


