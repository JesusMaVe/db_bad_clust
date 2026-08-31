"""
benchmark.py — Synthetic-schema benchmark for rule-engine coverage and scale.

Generates Oracle schemas of increasing size, converts them to DatabaseSchema,
runs the rule engine, and measures precision/recall against the generator's
manifest (ground truth). No Oracle container or BERT download is required.

Usage:
    .venv/bin/python benchmark.py --sizes 100 500 1000 --output output/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from db_bad_clust.generation.schema_generator import (
    SchemaGeneratorConfig,
    SyntheticSchemaGenerator,
)
from db_bad_clust.rules.rule_engine import ColumnRuleEngine, TableRuleEngine


@dataclass
class BenchmarkReport:
    """Aggregated benchmark result for one schema size."""

    num_tables: int
    num_columns: int
    generation_time_ms: float
    detection_time_ms: float
    column_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    report_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def _collect_detections(schema: Any) -> list[Any]:
    """Run all rule-engine detectors (classify + report-level signals)."""
    from db_bad_clust.rules.rule_engine import classify as classify_schema

    col_engine = ColumnRuleEngine()
    table_engine = TableRuleEngine()
    all_columns = [c for t in schema.tables for c in t.columns]

    results: list[Any] = []
    results.extend(classify_schema(schema=schema))
    results.extend(col_engine.detect_implicit_fks(all_columns, schema))
    results.extend(col_engine.detect_obsolete_types(all_columns, schema))
    results.extend(col_engine.detect_oversized_varchars(all_columns, schema))
    results.extend(table_engine.detect_missing_pk(schema))
    results.extend(table_engine.detect_redundant_indexes(schema))
    results.extend(table_engine.detect_fk_without_index(schema))
    results.extend(table_engine.detect_stale_statistics(schema))
    results.extend(table_engine.detect_disabled_constraints(schema))
    results.extend(table_engine.detect_partition_candidates(schema))
    return results


def _build_ground_truth(
    tables: list[Any], manifest: dict[str, Any]
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    """Build ground-truth sets from the manifest.

    Returns:
        column_truth: "TABLE.COLUMN" -> label
        table_truth: label -> {table names}
        column_report_truth: label -> { "TABLE.COLUMN" }
    """
    column_truth: dict[str, str] = {}
    table_truth: dict[str, set[str]] = defaultdict(set)
    column_report_truth: dict[str, set[str]] = defaultdict(set)

    for t in manifest["tables"]:
        table_name = t["name"]
        flags = t.get("report_flags", {})

        for col in t["columns"]:
            key = f"{table_name}.{col['name']}".upper()
            column_truth[key] = col["anti_pattern"]

        for label, present in {
            "fk_without_index": flags.get("fk_without_index"),
            "stale_statistics": flags.get("stale_statistics"),
            "disabled_constraints": flags.get("disabled_constraints"),
            "partition_candidate": flags.get("partition_candidate"),
        }.items():
            if present:
                table_truth[label].add(table_name)

        if flags.get("obsolete_types"):
            for col in t["columns"]:
                dtype = col["data_type"].upper()
                if any(dt in dtype for dt in {"LONG", "RAW"}):
                    column_report_truth["obsolete_type"].add(f"{table_name}.{col['name']}".upper())

        oversized = flags.get("oversized_varchars", {})
        for col_name in oversized:
            column_report_truth["oversized_varchar"].add(f"{table_name}.{col_name}".upper())

    return column_truth, dict(table_truth), dict(column_report_truth)


def _label_metrics(true_set: set[str], pred_set: set[str]) -> dict[str, float]:
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": len(true_set)}


def _evaluate_column_level(
    column_truth: dict[str, str], detections: list[Any]
) -> dict[str, dict[str, float]]:
    """Compute per-label metrics for classify() output."""
    true_by_label: dict[str, set[str]] = defaultdict(set)
    pred_by_label: dict[str, set[str]] = defaultdict(set)

    for key, label in column_truth.items():
        true_by_label[label].add(key)

    for r in detections:
        if r.predicted_label in {"missing_pk", "redundant_index", "fk_without_index",
                                  "stale_statistics", "disabled_constraint", "partition_candidate",
                                  "implicit_fk", "obsolete_type", "oversized_varchar"}:
            continue  # report-level only
        key = f"{r.table_name}.{r.column_name}".upper()
        pred_by_label[r.predicted_label].add(key)

    labels = set(true_by_label) | set(pred_by_label)
    return {
        label: _label_metrics(true_by_label.get(label, set()), pred_by_label.get(label, set()))
        for label in labels
    }


def _evaluate_report_level(
    table_truth: dict[str, set[str]],
    column_report_truth: dict[str, set[str]],
    detections: list[Any],
) -> dict[str, dict[str, float]]:
    """Compute per-label metrics for report-level detectors."""
    report_labels = {
        "fk_without_index", "stale_statistics", "disabled_constraint",
        "partition_candidate", "obsolete_type", "oversized_varchar",
        "missing_pk", "redundant_index", "implicit_fk",
    }

    pred_table: dict[str, set[str]] = defaultdict(set)
    pred_column: dict[str, set[str]] = defaultdict(set)

    for r in detections:
        if r.predicted_label not in report_labels:
            continue
        if r.predicted_label in {"obsolete_type", "oversized_varchar"}:
            key = f"{r.table_name}.{r.column_name}".upper()
            pred_column[r.predicted_label].add(key)
        else:
            pred_table[r.predicted_label].add(r.table_name)

    metrics: dict[str, dict[str, float]] = {}
    for label in report_labels:
        if label in {"obsolete_type", "oversized_varchar"}:
            metrics[label] = _label_metrics(
                column_report_truth.get(label, set()),
                pred_column.get(label, set()),
            )
        else:
            metrics[label] = _label_metrics(
                table_truth.get(label, set()),
                pred_table.get(label, set()),
            )
    return metrics


def run_benchmark(
    num_tables: int,
    seed: int = 42,
    density: float = 0.4,
) -> BenchmarkReport:
    """Run one benchmark iteration and return the report."""
    config = SchemaGeneratorConfig(
        num_tables=num_tables, seed=seed, anti_pattern_density=density
    )
    generator = SyntheticSchemaGenerator(seed=seed)

    t0 = time.perf_counter()
    tables, manifest = generator.generate(config)
    schema = SyntheticSchemaGenerator.to_database_schema(tables)
    generation_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    detections = _collect_detections(schema)
    detection_time = (time.perf_counter() - t0) * 1000

    column_truth, table_truth, column_report_truth = _build_ground_truth(tables, manifest)

    return BenchmarkReport(
        num_tables=num_tables,
        num_columns=manifest["summary"]["total_columns"],
        generation_time_ms=round(generation_time, 2),
        detection_time_ms=round(detection_time, 2),
        column_metrics=_evaluate_column_level(column_truth, detections),
        report_metrics=_evaluate_report_level(
            table_truth, column_report_truth, detections
        ),
    )


def _report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return {
        "num_tables": report.num_tables,
        "num_columns": report.num_columns,
        "generation_time_ms": report.generation_time_ms,
        "detection_time_ms": report.detection_time_ms,
        "column_metrics": report.column_metrics,
        "report_metrics": report.report_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark rule-engine coverage and scale on synthetic schemas."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Schema sizes (number of tables) to benchmark.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.4,
        help="Anti-pattern density for the generator.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/benchmark.json",
        help="Path to write the JSON benchmark report.",
    )
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    for size in args.sizes:
        print(f"Benchmarking {size} tables...")
        report = run_benchmark(size, seed=args.seed, density=args.density)
        reports.append(_report_to_dict(report))
        print(
            f"  {report.num_columns} columns | "
            f"gen {report.generation_time_ms}ms | "
            f"detect {report.detection_time_ms}ms"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "seed": args.seed,
                    "density": args.density,
                },
                "runs": reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
