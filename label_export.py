"""
label_export.py — Export columns from Oracle to a manual ground-truth CSV.

Connects to the Oracle in config.yaml, extracts every table column, and writes
output/manual_labels.csv with an empty `label` column for a human to fill.

Usage:
    .venv/bin/python label_export.py --output output/manual_labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from db_connector import OracleConnector
from schema_extractor import SchemaExtractor

LABEL_VOCABULARY = [
    "clean",
    "wrong_data_types",
    "reserved_words",
    "self_contradictory",
    "impossible_data",
    "self_referencing",
    "polymorphic",
    "giant_table",
    "inconsistent_naming",
    "eav",
]


def export_labels(output: Path) -> None:
    connector = OracleConnector(config_path="config.yaml")
    connection = connector.connect()
    try:
        schema = SchemaExtractor(connection).extract_all()
    finally:
        connection.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "column", "data_type", "label"])
        for table in schema.tables:
            for col in table.columns:
                writer.writerow([table.name, col.name, col.data_type, ""])

    print(f"Wrote {output} ({schema.total_columns()} columns)")
    print(f"Label vocabulary: {', '.join(LABEL_VOCABULARY)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="output/manual_labels.csv",
        help="Path to write the labeling CSV.",
    )
    args = parser.parse_args()
    export_labels(Path(args.output))


if __name__ == "__main__":
    main()
