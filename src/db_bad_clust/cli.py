"""
cli.py — Command-line entry point for the clustering experiments.

The notebooks (01, 02) build the feature blocks and pickle them; this is the
surface that turns those blocks into the numbers the write-up quotes. One
command:

  experiment  Score one or more clustering configurations against the 243
              hand-labelled columns. Needs no database.

Usage:
    db-bad-clust experiment --sweep
    db-bad-clust experiment --without-giants --ablation output/intermediate_*.pkl
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
        CONFLICT,
        CONFLICT_FUSED,
        GIANT_TABLES,
        STRUCTURE_ONLY,
        ablation,
        blind_scorer,
        bootstrap_compare,
        cluster_composition,
        evaluate,
        evaluate_blind,
        format_blind_table,
        format_bootstrap,
        format_diagnostics,
        format_robustness,
        format_stability,
        format_table,
        load_dataset,
        mean_over_wordings_scorer,
        robustness_over_wordings,
        stability_sweep,
        sweep,
        weights_for,
    )

    dataset = load_dataset(pickle_path=args.pickle, labels_path=args.labels)
    if args.without_giants:
        dataset = dataset.without_tables(GIANT_TABLES)
        print(f"excluding {', '.join(sorted(GIANT_TABLES))} — giant_table is a property")
        print("of the table, not of the column, so no per-column encoder can see it\n")
    print(f"{dataset.n_columns} columns, {len(set(dataset.table_of))} tables\n")

    name = "structure only (alpha=0)"
    run = evaluate(name, dataset, STRUCTURE_ONLY)
    print(format_table([run.score]))
    print()
    print(format_diagnostics({name: (dataset, STRUCTURE_ONLY)}))

    if args.sweep:
        print()
        print("What does the semantic block contribute? (corrected fusion)")
        print(format_table(sweep(dataset)))

    if args.ablation:
        others = {
            Path(path).stem: load_dataset(pickle_path=path, labels_path=args.labels)
            for path in args.ablation
        }
        if args.without_giants:
            others = {k: v.without_tables(GIANT_TABLES) for k, v in others.items()}
        print()
        print(f"Semantic representations, all at alpha={args.alpha:.2f}")
        print(format_table(ablation(others, alpha=args.alpha)))
        print()
        print(
            format_diagnostics(
                {
                    name: (dataset, STRUCTURE_ONLY),
                    **{k: (d, weights_for(args.alpha)) for k, d in others.items()},
                }
            )
        )

    if args.conflict or args.stability or args.robustness or args.bootstrap:
        if dataset.e_conflict is None:
            print(
                f"error: {args.pickle} carries no 'e_conflict' block — rebuild it with "
                "scripts/build_embeddings.py (--from-pickle works without Oracle)",
                file=sys.stderr,
            )
            return 1

    if (args.robustness or args.bootstrap) and not dataset.e_conflict_variants:
        print(
            f"error: {args.pickle} carries no 'e_conflict_variants' — rebuild it with "
            "scripts/build_embeddings.py (--from-pickle works without Oracle)",
            file=sys.stderr,
        )
        return 1

    if args.conflict:
        print()
        print("Conflict representation: what the name expects minus what the type declares")
        print("(every hyper-parameter chosen blind)")
        configs = [
            ("document (alpha=1)", weights_for(1.0)),
            ("conflict", CONFLICT),
            ("conflict fused", CONFLICT_FUSED),
        ]
        runs = [
            (name, evaluate_blind(name, dataset, weights, algorithm=algorithm))
            for name, weights in configs
            for algorithm in ("ward", "hdbscan")
        ]
        print(format_blind_table(runs))
        fused_ward = next(
            run for name, run in runs if name == "conflict fused" and run.algorithm == "ward"
        )
        print()
        print("Cluster composition, conflict fused / ward (tables spanned is the check):")
        print(cluster_composition(dataset, fused_ward.evaluation.cluster_ids))

    if args.robustness:
        for algorithm in ("ward", "hdbscan"):
            print()
            print(f"Robustness over anchor wordings — conflict fused / {algorithm}")
            print(format_robustness(robustness_over_wordings(dataset, CONFLICT_FUSED, algorithm)))

    if args.bootstrap:
        print()
        scorers = {
            "document / ward": blind_scorer(weights_for(1.0), "ward"),
            "conflict default / ward": blind_scorer(CONFLICT_FUSED, "ward", "orig"),
            "conflict mean / ward": mean_over_wordings_scorer(CONFLICT_FUSED, "ward"),
            "conflict mean / hdbscan": mean_over_wordings_scorer(CONFLICT_FUSED, "hdbscan"),
        }
        result = bootstrap_compare(
            dataset, scorers, n_boot=args.bootstrap, frac=args.bootstrap_frac, seed=0
        )
        print(format_bootstrap(result, "document / ward"))

    if args.stability:
        print()
        print("HDBSCAN grid, conflict representation — the operating point is chosen blind")
        print(format_stability(stability_sweep(dataset, CONFLICT)))

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


def _score(args: argparse.Namespace) -> int:
    """Health score + per-column structural recommendations for a database."""
    from db_bad_clust.evaluation.health_report import (
        fit_reference_model,
        format_health_report,
        score_reference,
        score_target,
        write_health_csv,
    )

    model = fit_reference_model(
        reference_pickle=args.reference, labels_path=args.labels, folds=args.folds
    )
    if args.target is None or Path(args.target) == Path(args.reference):
        result = score_reference(model, folds=args.folds)
    else:
        result = score_target(model, args.target, folds=args.folds)

    print(format_health_report(result, folds=args.folds))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        write_health_csv(args.csv, result)
        print(f"\nPer-column health verdicts → {args.csv}")

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
        "--sweep",
        action="store_true",
        help="score across the semantic block's share of the space",
    )
    experiment.add_argument(
        "--ablation",
        nargs="+",
        metavar="PICKLE",
        help="compare semantic representations, one pickle per representation",
    )
    experiment.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="weight given to the semantic block in --ablation (default 1.0)",
    )
    experiment.add_argument(
        "--without-giants",
        action="store_true",
        help="drop the two tables that alone supply every giant_table label",
    )
    experiment.add_argument(
        "--baselines",
        action="store_true",
        help="also run the reference points the pipeline has to beat",
    )
    experiment.add_argument(
        "--conflict",
        action="store_true",
        help="score the conflict representation (name expectation minus declared type) "
        "next to the document, with the ARI-against-table leakage column",
    )
    experiment.add_argument(
        "--robustness",
        action="store_true",
        help="re-run the blind conflict evaluation once per anchor wording and report "
        "the mean — the number to quote",
    )
    experiment.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="paired bootstrap: re-run the blind pipeline on N subsamples of the columns "
        "and report 95%% intervals for every configuration and its difference to the document",
    )
    experiment.add_argument(
        "--bootstrap-frac",
        type=float,
        default=0.8,
        help="share of columns in each bootstrap subsample (default 0.8)",
    )
    experiment.add_argument(
        "--stability",
        action="store_true",
        help="walk the HDBSCAN grid on the conflict representation and pick the "
        "operating point by relative validity, never by the labels",
    )
    experiment.set_defaults(func=_experiment)

    score = sub.add_parser(
        "score", help="health score + structural recommendations for a database"
    )
    score.add_argument(
        "--reference",
        default="output/intermediate_docs.pkl",
        help="labeled pickle the classifier is trained on",
    )
    score.add_argument("--labels", default="output/manual_labels.csv")
    score.add_argument(
        "--target",
        default=None,
        metavar="PICKLE",
        help="pickle for the database being scored; omit or pass --reference's path "
        "to self-check via out-of-fold prediction instead",
    )
    score.add_argument("--folds", type=int, default=5)
    score.add_argument("--csv", metavar="PATH", default="output/health_report.csv")
    score.set_defaults(func=_score)

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
