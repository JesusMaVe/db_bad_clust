"""
apply_comments.py — Apply the documentation overlay to the Oracle DB.

Runs COMMENT ON TABLE / COMMENT ON COLUMN statements from
anti_patterns.TABLE_COMMENTS and anti_patterns.COLUMN_COMMENTS.
Idempotent: re-running simply overwrites the comments.

Usage:
    python3 apply_comments.py
"""

from __future__ import annotations

from anti_patterns import COLUMN_COMMENTS, TABLE_COMMENTS
from db_connector import OracleConnector


def apply_comments() -> None:
    conn = OracleConnector(config_path="config.yaml").connect()
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM user_tables")
    existing = {row[0] for row in cur.fetchall()}

    statements = 0
    # ponytail: no bind detection in COMMENT statements; inline with escaping
    for table, comment in TABLE_COMMENTS.items():
        if table not in existing:
            print(f"  skipped {table} (not in DB)")
            continue
        cur.execute(f"COMMENT ON TABLE {table} IS '{comment.replace(chr(39), chr(39)*2)}'")
        statements += 1

    for table, columns in COLUMN_COMMENTS.items():
        if table not in existing:
            continue
        for column, comment in columns.items():
            cur.execute(
                f'COMMENT ON COLUMN {table}."{column}" '
                f"IS '{comment.replace(chr(39), chr(39)*2)}'"
            )
            statements += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Applied {statements} comments "
          f"({len(TABLE_COMMENTS)} tables, "
          f"{sum(len(v) for v in COLUMN_COMMENTS.values())} columns)")


if __name__ == "__main__":
    apply_comments()
