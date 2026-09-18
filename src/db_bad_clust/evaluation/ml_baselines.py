"""
ml_baselines.py — The baselines the semantic pipeline has to beat.

Two reference points, both scored against the same manual ground truth in
`output/manual_labels.csv` so every figure sits on one scale:

  random_forest_baseline()          supervised ceiling: how much of the label
                                    can a classifier extract from the composite
                                    vector at all? A clustering that lands far
                                    below this is limited by the clustering, not
                                    by the features.
  clustering_algorithm_comparison()  algorithm control: is the density-based
                                    choice actually doing the work, or would
                                    K-Means on the same vector do as well?

Neither is the contribution — they are the floor. A claim that the semantic
component helps means beating these, not beating nothing.

Usage:
    db-bad-clust experiment --baselines
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from db_bad_clust.evaluation.report_table import Column, render_table

# The structure-only configuration: alpha=0 switches the embeddings off, so
# this is the baseline the semantic component is measured against.
CLUSTERING_WEIGHTS = {"alpha": 0.00, "beta": 0.35, "gamma": 0.45, "delta": 0.20}
PCA_COMPONENTS = 20
CV_FOLDS = 5


def _load_phi(pickle_path: str | Path, labels_path: str | Path) -> tuple[np.ndarray, list[str]]:
    """`random_forest_baseline` alone needs the precomputed `phi` a notebook-02
    -style pickle stores — that field isn't one of the four raw blocks
    `evaluation.experiments.load_dataset` loads, so it still opens the file
    directly rather than going through `Dataset`."""
    from db_bad_clust.evaluation.cluster_scoring import load_manual_labels

    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)
    return data["phi"], load_manual_labels(labels_path, data["column_index"])


# ── Supervised baseline ───────────────────────────────────────────────


def random_forest_baseline(
    pickle_path: str | Path = "output/intermediate_02.pkl",
    labels_path: str | Path = "output/manual_labels.csv",
    folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """Can a supervised classifier learn the anti-pattern labels?

    Runs stratified cross-validation on the stored composite vector. The
    interesting part is not the score but *why* it is low: several classes have
    one or two examples in the whole database, and no algorithm learns a pattern
    it saw once.
    """
    import warnings
    from collections import Counter

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    phi, truth = _load_phi(pickle_path, labels_path)
    support = Counter(truth)

    # sklearn warns rather than fails when a class has fewer members than folds,
    # and that warning IS the finding — capture it instead of hiding it.
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = cross_val_score(clf, phi, truth, cv=folds, scoring="f1_macro")
    split_warnings = [str(w.message) for w in caught if "least populated class" in str(w.message)]

    clf.fit(phi, truth)
    importances = clf.feature_importances_
    top = np.argsort(importances)[-10:][::-1]

    return {
        "f1_macro_mean": float(scores.mean()),
        "f1_macro_std": float(scores.std()),
        "folds": folds,
        "n_features": int(phi.shape[1]),
        "rarest_classes": sorted(support.items(), key=lambda kv: kv[1])[:4],
        "top_features": [(int(i), float(importances[i])) for i in top],
        "split_warnings": split_warnings,
    }


# ── Unsupervised baseline ─────────────────────────────────────────────


def clustering_algorithm_comparison(
    pickle_path: str | Path = "output/intermediate_02.pkl",
    labels_path: str | Path = "output/manual_labels.csv",
    methods: tuple[str, ...] = ("kmeans", "dbscan", "agglomerative", "meanshift", "hdbscan"),
) -> list[dict[str, Any]]:
    """Is the density-based algorithm doing the work, or would K-Means do as well?

    All algorithms run on the same PCA-reduced composite vector and are scored
    against the same manual labels, so the comparison is internally consistent.
    """
    from sklearn.decomposition import PCA

    from db_bad_clust.clustering.cluster_engine import ClusterEngine
    from db_bad_clust.evaluation.experiments import build_phi, load_dataset
    from db_bad_clust.evaluation.validation import GroundTruthValidator

    # load_dataset widens the stored float32 blocks to float64 — see its
    # docstring on why a degenerate structural representation is precision-
    # sensitive. Building phi here through the same path as `cli experiment`
    # keeps this table on the same arithmetic width as the rest of the report
    # it's printed alongside, rather than silently reading float32.
    dataset = load_dataset(pickle_path=pickle_path, labels_path=labels_path)
    phi = build_phi(dataset, CLUSTERING_WEIGHTS)
    reduced = PCA(n_components=min(PCA_COMPONENTS, phi.shape[1]), random_state=42).fit_transform(phi)

    validator = GroundTruthValidator()
    truth = dataset.truth
    truth_arr = np.asarray(truth)
    n_true_classes = len(set(truth))

    rows: list[dict[str, Any]] = []
    for method in methods:
        try:
            labels = ClusterEngine(method=method, n_clusters=n_true_classes).fit_predict(reduced)
        except Exception as exc:  # a method may be unavailable or degenerate
            rows.append({"method": method, "error": str(exc)[:60]})
            continue
        n_clusters = len(set(labels.tolist()) - {-1})
        rows.append(
            {
                "method": method,
                "ari": float(validator.adjusted_rand_index(labels, truth_arr)),
                "nmi": float(validator.normalized_mutual_info(labels, truth_arr)),
                "n_clusters": n_clusters,
                "noise": int((labels == -1).sum()),
            }
        )
    return sorted(rows, key=lambda r: r.get("ari", -99), reverse=True)


# ── Reporting ─────────────────────────────────────────────────────────


def format_baselines(rf: dict[str, Any], clustering: list[dict[str, Any]]) -> str:
    """Human-readable summary of both baselines."""
    lines: list[str] = []
    add = lines.append

    add("Supervised baseline — Random Forest")
    add("-" * 72)
    add(
        f"  F1-macro {rf['f1_macro_mean']:.4f} +/- {rf['f1_macro_std']:.4f} "
        f"({rf['folds']}-fold CV, {rf['n_features']} features)"
    )
    rarest = ", ".join(f"{name} n={n}" for name, n in rf["rarest_classes"])
    add(f"  rarest classes: {rarest}")
    if rf["split_warnings"]:
        add("  sklearn: the folds are degenerate — a class has fewer members than folds,")
        add("  which is why the standard deviation is large and the score unstable")
    add("  a class seen once cannot be learned — this is a data limit, not a tuning one")
    add("")

    add("Unsupervised baseline — clustering algorithms on the same vector")
    columns = [
        Column("method", 16, "<"),
        Column("ARI", 9),
        Column("NMI", 9),
        Column("clusters", 10),
        Column("noise", 8),
    ]
    ok_rows = [
        [row["method"], f"{row['ari']:.4f}", f"{row['nmi']:.4f}", str(row["n_clusters"]), str(row["noise"])]
        for row in clustering
        if "error" not in row
    ]
    table = render_table(columns, ok_rows)
    add("\n".join("  " + line for line in table.splitlines()))
    for row in clustering:
        if "error" in row:
            add(f"  {row['method']:<16}{'unavailable':>9}  {row['error']}")
    add("")
    add("  Scored against the 243 manual labels. These are reference points, not")
    add("  results: the semantic pipeline has to clear them to have said anything.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_baselines(random_forest_baseline(), clustering_algorithm_comparison()))
