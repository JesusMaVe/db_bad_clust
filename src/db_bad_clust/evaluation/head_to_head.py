"""
head_to_head.py — Rule engine (branch A) vs clustering (branch B), same ruler.

The two branches are not natively comparable: the rule engine emits *labels*
("wrong_data_types"), the clustering pipeline emits *unnamed groups* ("cluster
4"). Quoting accuracy for one and ARI for the other compares two rulers and
invites the objection that the ruler was picked to favour the winner.

This module scores BOTH branches with BOTH rulers against the same manual
ground truth:

  1. Branch A is lowered onto B's turf (ARI / NMI). A set of labels is also a
     partition, so the rule engine's output can be scored as a clustering.
  2. Branch B is raised onto A's turf (accuracy / F1). Each cluster is named
     after the most common true label among its members ("cluster-then-label").

Step 2 *hands B free information*: it uses the ground truth to name B's own
clusters, which B never had access to. B's accuracy here is therefore an upper
bound — its best possible case. A win for A under this protocol is a win
against a handicapped opponent.

Usage:
  from db_bad_clust.evaluation.head_to_head import run_comparison, format_report

  result = run_comparison()
  print(format_report(result))
"""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Best clustering configuration (re-validated, issue #1): PCA 20D + HDBSCAN.
# alpha=0.0 means the BERT embeddings are switched off — this is branch B at
# its own measured optimum, not a configuration chosen to make it look bad.
BEST_CLUSTERING_WEIGHTS = {"alpha": 0.00, "beta": 0.35, "gamma": 0.45, "delta": 0.20}
PCA_COMPONENTS = 20
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 2


@dataclass
class BranchScore:
    """Scores for one branch under both rulers."""

    name: str
    ari: float
    nmi: float
    accuracy: float
    f1_macro: float
    n_groups: int

    def as_row(self) -> dict[str, Any]:
        return {
            "branch": self.name,
            "ari": round(self.ari, 4),
            "nmi": round(self.nmi, 4),
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "n_groups": self.n_groups,
        }


@dataclass
class ComparisonResult:
    """Full head-to-head outcome."""

    rules: BranchScore
    clustering: BranchScore
    n_columns: int
    truth: list[str]
    rules_pred: list[str]
    clustering_pred: list[str]
    column_index: list[str]
    cluster_raw: list[int]
    per_class_f1: dict[str, dict[str, float]] = field(default_factory=dict)
    cluster_contents: dict[int, Counter] = field(default_factory=dict)

    def disagreements(self, limit: int | None = None) -> list[dict[str, str]]:
        """Columns where A is right and B is wrong (the evidence exhibits)."""
        rows = [
            {
                "column": col,
                "truth": t,
                "rules": a,
                "clustering": b,
                "cluster": str(c),
            }
            for col, t, a, b, c in zip(
                self.column_index,
                self.truth,
                self.rules_pred,
                self.clustering_pred,
                self.cluster_raw,
                strict=True,
            )
            if a == t and b != t
        ]
        return rows[:limit] if limit is not None else rows

    def reverse_disagreements(self, limit: int | None = None) -> list[dict[str, str]]:
        """Columns where B is right and A is wrong — reported for fairness."""
        rows = [
            {
                "column": col,
                "truth": t,
                "rules": a,
                "clustering": b,
                "cluster": str(c),
            }
            for col, t, a, b, c in zip(
                self.column_index,
                self.truth,
                self.rules_pred,
                self.clustering_pred,
                self.cluster_raw,
                strict=True,
            )
            if b == t and a != t
        ]
        return rows[:limit] if limit is not None else rows


# ── Cluster → label mapping ───────────────────────────────────────────


def majority_vote_map(
    cluster_labels: np.ndarray,
    true_labels: list[str],
) -> dict[int, str]:
    """Name each cluster after the most common true label among its members.

    HDBSCAN noise (-1) is treated as an ordinary cluster and also gets a
    name — the generous reading, since discarding it would only lower B.
    """
    mapping: dict[int, str] = {}
    for cid in set(int(c) for c in cluster_labels):
        members = [t for c, t in zip(cluster_labels, true_labels, strict=True) if int(c) == cid]
        mapping[cid] = Counter(members).most_common(1)[0][0]
    return mapping


def clusters_to_labels(
    cluster_labels: np.ndarray,
    true_labels: list[str],
) -> list[str]:
    """Turn unnamed clusters into predicted labels via majority vote."""
    mapping = majority_vote_map(cluster_labels, true_labels)
    return [mapping[int(c)] for c in cluster_labels]


# ── Scoring ───────────────────────────────────────────────────────────


def _score(
    name: str,
    pred_labels: list[str],
    true_labels: list[str],
    group_ids: np.ndarray | list[int] | None = None,
) -> BranchScore:
    """Score one branch with both rulers."""
    from sklearn.metrics import accuracy_score, f1_score

    from db_bad_clust.evaluation.validation import GroundTruthValidator

    validator = GroundTruthValidator()
    # ARI/NMI are computed on the *grouping*, not the names: for the rule
    # engine the grouping is its own labels; for clustering it is the raw
    # cluster ids (naming them cannot change the partition, but using the
    # raw ids keeps the two branches symmetric).
    grouping = np.asarray(group_ids if group_ids is not None else pred_labels)
    truth_arr = np.asarray(true_labels)

    return BranchScore(
        name=name,
        ari=validator.adjusted_rand_index(grouping, truth_arr),
        nmi=validator.normalized_mutual_info(grouping, truth_arr),
        accuracy=float(accuracy_score(true_labels, pred_labels)),
        f1_macro=float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        n_groups=len(set(grouping.tolist())),
    )


# ── Branch runners ────────────────────────────────────────────────────


def _run_branch_a(schema: Any, column_index: list[str]) -> list[str]:
    """Rule engine labels, aligned to column_index order.

    The engine emits fine-grained labels (`bad_boolean`, `date_as_text`, ...)
    that must be translated to the manual vocabulary through RULE_TO_MANUAL
    before scoring — otherwise A is penalised for being *more* specific than
    the ground truth, which is not an error.
    """
    from db_bad_clust.generation.ground_truth import RULE_TO_MANUAL
    from db_bad_clust.rules.rule_engine import classify

    results = classify(schema=schema)
    by_key = {
        f"{r.table_name}.{r.column_name}".upper(): RULE_TO_MANUAL.get(
            r.predicted_label, r.predicted_label
        )
        for r in results
    }
    return [by_key.get(key.upper(), "clean") for key in column_index]


def _run_branch_b(
    data: dict[str, Any],
    weights: dict[str, float],
) -> np.ndarray:
    """Clustering pipeline at its measured optimum → raw cluster ids."""
    import hdbscan
    from sklearn.decomposition import PCA

    from db_bad_clust.features.feature_builder import FeatureBuilder

    builder = FeatureBuilder(**weights)
    phi = builder.build(
        e_text=data["e_text"],
        e_type=data["e_type"],
        e_rest=data["e_rest"],
        e_stat=data["e_stat"],
    )
    reduced = PCA(n_components=min(PCA_COMPONENTS, phi.shape[1]), random_state=42).fit_transform(phi)
    return hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    ).fit_predict(reduced)


# ── Entry point ───────────────────────────────────────────────────────


def load_manual_labels(path: str | Path, column_index: list[str]) -> list[str]:
    """Read manual_labels.csv into column_index order."""
    import csv

    with open(path, newline="") as fh:
        by_key = {
            f"{row['table']}.{row['column']}".upper(): row["label"].strip()
            for row in csv.DictReader(fh)
        }
    missing = [k for k in column_index if k.upper() not in by_key]
    if missing:
        raise ValueError(f"{len(missing)} columns lack a manual label, e.g. {missing[:3]}")
    return [by_key[k.upper()] for k in column_index]


def run_comparison(
    pickle_path: str | Path = "output/intermediate_02.pkl",
    labels_path: str | Path = "output/manual_labels.csv",
    weights: dict[str, float] | None = None,
) -> ComparisonResult:
    """Run both branches against the manual ground truth. No database needed."""
    from sklearn.metrics import f1_score

    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)

    column_index: list[str] = data["column_index"]
    truth = load_manual_labels(labels_path, column_index)

    rules_pred = _run_branch_a(data["schema"], column_index)
    cluster_raw = _run_branch_b(data, weights or BEST_CLUSTERING_WEIGHTS)
    clustering_pred = clusters_to_labels(cluster_raw, truth)

    rules = _score("Rule engine (A)", rules_pred, truth)
    clustering = _score("Clustering (B)", clustering_pred, truth, group_ids=cluster_raw)

    classes = sorted(set(truth))
    per_class = {
        "rules": dict(
            zip(
                classes,
                f1_score(truth, rules_pred, labels=classes, average=None, zero_division=0),
                strict=True,
            )
        ),
        "clustering": dict(
            zip(
                classes,
                f1_score(truth, clustering_pred, labels=classes, average=None, zero_division=0),
                strict=True,
            )
        ),
    }

    contents: dict[int, Counter] = {}
    for cid, t in zip(cluster_raw, truth, strict=True):
        contents.setdefault(int(cid), Counter())[t] += 1

    return ComparisonResult(
        rules=rules,
        clustering=clustering,
        n_columns=len(truth),
        truth=truth,
        rules_pred=rules_pred,
        clustering_pred=clustering_pred,
        column_index=column_index,
        cluster_raw=[int(c) for c in cluster_raw],
        per_class_f1=per_class,
        cluster_contents=contents,
    )


def alpha_sweep(
    pickle_path: str | Path = "output/intermediate_02.pkl",
    labels_path: str | Path = "output/manual_labels.csv",
    alphas: tuple[float, ...] = (0.00, 0.15, 0.30, 0.50, 0.70),
) -> list[dict[str, Any]]:
    """What does BERT actually contribute? Sweep alpha, hold the rest fixed.

    The remaining weights are rescaled to keep their relative proportions, so
    the only thing changing is how much say the semantic embeddings get.
    """
    from sklearn.metrics import accuracy_score

    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)
    truth = load_manual_labels(labels_path, data["column_index"])

    base = BEST_CLUSTERING_WEIGHTS
    struct_total = base["beta"] + base["gamma"] + base["delta"]
    rows = []
    for alpha in alphas:
        scale = (1.0 - alpha) / struct_total
        weights = {
            "alpha": alpha,
            "beta": base["beta"] * scale,
            "gamma": base["gamma"] * scale,
            "delta": base["delta"] * scale,
        }
        cluster_raw = _run_branch_b(data, weights)
        pred = clusters_to_labels(cluster_raw, truth)
        score = _score(f"alpha={alpha:.2f}", pred, truth, group_ids=cluster_raw)
        rows.append(
            {
                "alpha": alpha,
                "ari": round(score.ari, 4),
                "nmi": round(score.nmi, 4),
                "accuracy": round(accuracy_score(truth, pred), 4),
                "n_clusters": score.n_groups,
            }
        )
    return rows


# ── Reporting ─────────────────────────────────────────────────────────


def format_report(result: ComparisonResult, n_examples: int = 8) -> str:
    """Human-readable head-to-head table."""
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add(f"HEAD TO HEAD — {result.n_columns} columns, manual ground truth")
    add("=" * 72)
    add("")
    add(f"{'Branch':<20} {'ARI':>8} {'NMI':>8} {'Accuracy':>10} {'F1-macro':>10} {'Groups':>8}")
    add("-" * 72)
    for score in (result.rules, result.clustering):
        add(
            f"{score.name:<20} {score.ari:>8.4f} {score.nmi:>8.4f} "
            f"{score.accuracy:>10.4f} {score.f1_macro:>10.4f} {score.n_groups:>8}"
        )
    add("")
    add("B's accuracy/F1 use ground-truth-assisted cluster naming — an upper bound.")
    add("")

    add("Per-class F1")
    add("-" * 72)
    add(f"{'Class':<24} {'Rules (A)':>12} {'Clustering (B)':>16}")
    for cls in sorted(result.per_class_f1["rules"]):
        add(
            f"{cls:<24} {result.per_class_f1['rules'][cls]:>12.3f} "
            f"{result.per_class_f1['clustering'][cls]:>16.3f}"
        )
    add("")

    add("What each cluster actually contains")
    add("-" * 72)
    for cid in sorted(result.cluster_contents):
        counts = result.cluster_contents[cid]
        mix = ", ".join(f"{k} x{v}" for k, v in counts.most_common(4))
        tag = "noise" if cid == -1 else f"cluster {cid}"
        add(f"  {tag:<12} n={sum(counts.values()):<4} {mix}")
    add("")

    wrong = result.disagreements()
    right = result.reverse_disagreements()
    add(f"A right / B wrong: {len(wrong)} columns")
    add(f"B right / A wrong: {len(right)} columns")
    add("")
    add(f"Examples where A is right and B is wrong (first {n_examples})")
    add("-" * 72)
    for row in wrong[:n_examples]:
        add(f"  {row['column']:<38} truth={row['truth']:<20} B said={row['clustering']}")
    if right:
        add("")
        add(f"Examples where B is right and A is wrong (first {n_examples})")
        add("-" * 72)
        for row in right[:n_examples]:
            add(f"  {row['column']:<38} truth={row['truth']:<20} A said={row['rules']}")
    return "\n".join(lines)


def write_csv(result: ComparisonResult, path: str | Path) -> None:
    """Per-column verdict, so the numbers can be audited by hand."""
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["column", "truth", "rules_pred", "cluster_id", "clustering_pred"])
        for col, t, a, cid, b in zip(
            result.column_index,
            result.truth,
            result.rules_pred,
            result.cluster_raw,
            result.clustering_pred,
            strict=True,
        ):
            writer.writerow([col, t, a, cid, b])


if __name__ == "__main__":
    outcome = run_comparison()
    print(format_report(outcome))
    print()
    print("Contribution of BERT (alpha sweep)")
    print("-" * 72)
    print(f"{'alpha':>8} {'ARI':>8} {'NMI':>8} {'Accuracy':>10} {'Clusters':>10}")
    for row in alpha_sweep():
        print(
            f"{row['alpha']:>8.2f} {row['ari']:>8.4f} {row['nmi']:>8.4f} "
            f"{row['accuracy']:>10.4f} {row['n_clusters']:>10}"
        )
    write_csv(outcome, "output/head_to_head.csv")
    print("\nPer-column verdicts → output/head_to_head.csv")
