"""
cli.py — Command-line entry point for schema audits.

The notebooks remain the research record; this is the product surface. Two
commands:

  audit    Connect to Oracle, classify every column, print the findings and
           optionally emit the corrective SQL.
  compare  Re-run the rule engine against the clustering pipeline on stored
           results. Needs no database.

Usage:
    db-bad-clust audit --sql output/fix.sql
    db-bad-clust compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from db_bad_clust.exceptions import BadDBError


def _audit(args: argparse.Namespace) -> int:
    """Extract → classify → recommend → (optionally) corrective SQL."""
    from db_bad_clust.data.db_connector import OracleConnector
    from db_bad_clust.data.schema_extractor import SchemaExtractor
    from db_bad_clust.generation.ddl_generator import DDLGenerator
    from db_bad_clust.rules.recommender import Recommender

    connector = OracleConnector(config_path=args.config)
    try:
        connection = connector.connect()
        schema = SchemaExtractor(connection).extract_all()
    finally:
        connector.close()

    print(f"Schema: {len(schema.tables)} tables, {schema.total_columns()} columns\n")

    recommendations = Recommender().recommend(schema)
    print(Recommender.print_recommendations(recommendations))

    if args.sql:
        sql = DDLGenerator().from_schema(schema)
        Path(args.sql).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sql).write_text(sql)
        print(f"\nCorrective SQL → {args.sql}")

    if args.json:
        payload: dict[str, Any] = {
            "tables": len(schema.tables),
            "columns": schema.total_columns(),
            "recommendations": recommendations,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"Findings → {args.json}")

    return 0


def _compare(args: argparse.Namespace) -> int:
    """Rule engine vs clustering, both scored with both rulers."""
    from db_bad_clust.evaluation.head_to_head import (
        alpha_sweep,
        format_report,
        run_comparison,
        write_csv,
    )

    result = run_comparison(pickle_path=args.pickle, labels_path=args.labels)
    print(format_report(result))

    if args.sweep:
        print("\nContribution of BERT (alpha sweep)")
        print("-" * 72)
        print(f"{'alpha':>8} {'ARI':>8} {'NMI':>8} {'Accuracy':>10} {'Clusters':>10}")
        for row in alpha_sweep(pickle_path=args.pickle, labels_path=args.labels):
            print(
                f"{row['alpha']:>8.2f} {row['ari']:>8.4f} {row['nmi']:>8.4f} "
                f"{row['accuracy']:>10.4f} {row['n_clusters']:>10}"
            )

    if args.baselines:
        from db_bad_clust.evaluation.ml_baselines import (
            clustering_algorithm_comparison,
            format_baselines,
            random_forest_baseline,
        )

        print()
        print(
            format_baselines(
                random_forest_baseline(pickle_path=args.pickle, labels_path=args.labels),
                clustering_algorithm_comparison(pickle_path=args.pickle, labels_path=args.labels),
            )
        )

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        write_csv(result, args.csv)
        print(f"\nPer-column verdicts → {args.csv}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="db-bad-clust",
        description="Detect schema anti-patterns in Oracle databases.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="audit a live Oracle schema")
    audit.add_argument("--config", default="config.yaml", help="connection config (config.yaml)")
    audit.add_argument("--sql", metavar="PATH", help="write corrective SQL to PATH")
    audit.add_argument("--json", metavar="PATH", help="write findings as JSON to PATH")
    audit.set_defaults(func=_audit)

    compare = sub.add_parser("compare", help="rule engine vs clustering (no database)")
    compare.add_argument("--pickle", default="output/intermediate_02.pkl")
    compare.add_argument("--labels", default="output/manual_labels.csv")
    compare.add_argument("--csv", metavar="PATH", default="output/head_to_head.csv")
    compare.add_argument(
        "--sweep",
        action="store_true",
        help="also report what the BERT embeddings contribute",
    )
    compare.add_argument(
        "--baselines",
        action="store_true",
        help="also run the ML approaches that lost (Random Forest, clustering algorithms)",
    )
    compare.set_defaults(func=_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BadDBError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc.filename} not found", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
