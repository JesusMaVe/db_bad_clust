"""
cli.py — Command-line entry point for the clustering experiments.

The notebooks (01, 02) build the feature blocks and pickle them; this is the
surface that turns those blocks into the numbers the write-up quotes. One
command:

  experiment  Score one or more clustering configurations against the 243
              hand-labelled columns. Needs no database.

Usage:
    db-bad-clust experiment
    db-bad-clust experiment --baselines --csv output/per_column.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from db_bad_clust.exceptions import BadDBError


def _experiment(args: argparse.Namespace) -> int:
    """Score the clustering pipeline against the manual ground truth."""
    from db_bad_clust.evaluation.cluster_scoring import write_per_column_csv
    from db_bad_clust.evaluation.experiments import (
        STRUCTURE_ONLY,
        evaluate,
        format_diagnostics,
        format_table,
        load_dataset,
    )

    dataset = load_dataset(pickle_path=args.pickle, labels_path=args.labels)
    print(f"{dataset.n_columns} columns, {len(set(dataset.table_of))} tables\n")

    name = "structure only (alpha=0)"
    run = evaluate(name, dataset, STRUCTURE_ONLY)
    print(format_table([run.score]))
    print()
    print(format_diagnostics(dataset, {name: STRUCTURE_ONLY}))

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
        write_per_column_csv(
            args.csv,
            column_index=dataset.column_index,
            truth=dataset.truth,
            cluster_ids=run.cluster_ids,
            predicted=run.predicted,
        )
        print(f"\nPer-column verdicts → {args.csv}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="db-bad-clust",
        description="Cluster Oracle schema columns into anti-pattern groups.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    experiment = sub.add_parser("experiment", help="score clustering configurations (no database)")
    experiment.add_argument("--pickle", default="output/intermediate_02.pkl")
    experiment.add_argument("--labels", default="output/manual_labels.csv")
    experiment.add_argument("--csv", metavar="PATH", default="output/per_column.csv")
    experiment.add_argument(
        "--baselines",
        action="store_true",
        help="also run the reference points the pipeline has to beat",
    )
    experiment.set_defaults(func=_experiment)

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
