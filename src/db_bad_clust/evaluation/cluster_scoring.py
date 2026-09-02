"""
cluster_scoring.py — Scoring a clustering against the manual ground truth.

A clustering pipeline emits *unnamed groups* ("cluster 4"); the ground truth in
`output/manual_labels.csv` is a set of *labels* ("wrong_data_types"). The two
are not natively comparable, so this module supplies both rulers:

  1. The partition ruler (ARI / NMI): a labelling is also a partition, so the
     cluster ids can be scored directly against the true labels.
  2. The label ruler (accuracy / F1): each cluster is named after the most
     common true label among its members ("cluster-then-label").

Step 2 *hands the clustering free information* — it uses the ground truth to
name the clusters, which the pipeline itself never had access to. Any accuracy
or F1 reported this way is therefore an **upper bound**, not an achieved score.
Say so wherever the number is quoted.

Usage:
    from db_bad_clust.evaluation.cluster_scoring import (
        clusters_to_labels, load_manual_labels, score,
    )

    truth = load_manual_labels("output/manual_labels.csv", column_index)
    pred = clusters_to_labels(cluster_ids, truth)
    print(score("BERT + structure", pred, truth, group_ids=cluster_ids).as_row())
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ClusterScore:
    """One configuration scored under both rulers."""

    name: str
    ari: float
    nmi: float
    accuracy: float
    f1_macro: float
    n_groups: int
    ami: float = 0.0
    v_measure: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "branch": self.name,
            "ari": round(self.ari, 4),
            "nmi": round(self.nmi, 4),
            "ami": round(self.ami, 4),
            "v_measure": round(self.v_measure, 4),
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "n_groups": self.n_groups,
        }


# ── Cluster → label mapping ───────────────────────────────────────────


def majority_vote_map(
    cluster_labels: np.ndarray,
    true_labels: list[str],
) -> dict[int, str]:
    """Name each cluster after the most common true label among its members.

    HDBSCAN noise (-1) is treated as an ordinary cluster and also gets a
    name — the generous reading, since discarding it would only lower the
    clustering's score.
    """
    mapping: dict[int, str] = {}
    for cid in {int(c) for c in cluster_labels}:
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


def score(
    name: str,
    pred_labels: list[str],
    true_labels: list[str],
    group_ids: np.ndarray | list[int] | None = None,
) -> ClusterScore:
    """Score one configuration with both rulers.

    ARI/NMI/AMI/V are computed on the *grouping*, not the names: pass
    `group_ids` to score the raw cluster ids, since majority-vote naming can
    merge two clusters into one label and would otherwise misreport the
    partition.

    AMI is reported alongside NMI because the configurations compared here
    produce anywhere from 2 to 19 clusters, and NMI rises with the cluster
    count on its own; AMI corrects for the agreement expected by chance at
    that count, so it is the one to quote when the counts differ.
    """
    from sklearn.metrics import (
        accuracy_score,
        adjusted_mutual_info_score,
        f1_score,
        v_measure_score,
    )

    from db_bad_clust.evaluation.validation import GroundTruthValidator

    validator = GroundTruthValidator()
    grouping = np.asarray(group_ids if group_ids is not None else pred_labels)
    truth_arr = np.asarray(true_labels)

    return ClusterScore(
        name=name,
        ari=validator.adjusted_rand_index(grouping, truth_arr),
        nmi=validator.normalized_mutual_info(grouping, truth_arr),
        accuracy=float(accuracy_score(true_labels, pred_labels)),
        f1_macro=float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        n_groups=len(set(grouping.tolist())),
        ami=float(adjusted_mutual_info_score(truth_arr, grouping)),
        v_measure=float(v_measure_score(truth_arr, grouping)),
    )


# ── Ground truth ──────────────────────────────────────────────────────


def load_manual_labels(path: str | Path, column_index: list[str]) -> list[str]:
    """Read manual_labels.csv into column_index order.

    Raises if any column in the index lacks a label: a silently missing label
    would be scored as a wrong prediction and quietly deflate every metric.
    """
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


# ── Reporting ─────────────────────────────────────────────────────────


def write_per_column_csv(
    path: str | Path,
    column_index: list[str],
    truth: list[str],
    cluster_ids: list[int],
    predicted: list[str],
) -> None:
    """Per-column verdict, so the aggregate numbers can be audited by hand."""
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["column", "truth", "cluster_id", "predicted"])
        for col, t, cid, p in zip(column_index, truth, cluster_ids, predicted, strict=True):
            writer.writerow([col, t, cid, p])
