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

# Rule-engine raw labels -> manual vocabulary
RULE_TO_MANUAL = {
    "date_as_text": "wrong_data_types",
    "number_as_text": "wrong_data_types",
    "bad_boolean": "self_contradictory",
    "reserved_word": "reserved_words",
    "eav_pattern": "eav",
}


def export_labels(output: Path, prefill: bool = False) -> None:
    connector = OracleConnector(config_path="config.yaml")
    connection = connector.connect()
    try:
        schema = SchemaExtractor(connection).extract_all()
    finally:
        connection.close()

    labels: dict[str, str] = {}
    if prefill:
        from rule_engine import classify as classify_schema

        for r in classify_schema(schema=schema):
            key = f"{r.table_name}.{r.column_name}".upper()
            labels[key] = RULE_TO_MANUAL.get(r.predicted_label, r.predicted_label)

    output.parent.mkdir(parents=True, exist_ok=True)
    filled = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "column", "data_type", "label"])
        for table in schema.tables:
            for col in table.columns:
                label = labels.get(f"{table.name}.{col.name}".upper(), "")
                if label:
                    filled += 1
                writer.writerow([table.name, col.name, col.data_type, label])

    print(f"Wrote {output} ({schema.total_columns()} columns)")
    if prefill:
        print(f"Prefilled {filled} labels from rule engine — review and correct them")
    print(f"Label vocabulary: {', '.join(LABEL_VOCABULARY)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="output/manual_labels.csv",
        help="Path to write the labeling CSV.",
    )
    parser.add_argument(
        "--prefill",
        action="store_true",
        help="Fill labels with rule-engine predictions for review.",
    )
    args = parser.parse_args()
    export_labels(Path(args.output), prefill=args.prefill)


if __name__ == "__main__":
    main()
