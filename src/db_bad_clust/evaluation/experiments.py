"""
experiments.py — Run the clustering pipeline under a configuration and score it.

One place to answer "what does this configuration achieve?", so that every
number in the write-up comes from the same code path and the same ground truth.
A configuration is the four fusion weights plus the reduction and clustering
settings; the answer is a `ClusterScore` under both rulers.

The evaluation is always against the 243 hand-labelled columns in
`output/manual_labels.csv` — never against a detector's own output.

Usage:
    from db_bad_clust.evaluation.experiments import load_dataset, evaluate

    data = load_dataset("output/intermediate_02.pkl", "output/manual_labels.csv")
    print(evaluate("structure only", data, STRUCTURE_ONLY).score.as_row())
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from db_bad_clust.evaluation.cluster_scoring import (
    ClusterScore,
    clusters_to_labels,
    load_manual_labels,
    score,
)

# Structure only: alpha=0 switches the embeddings off. This is the reference
# the semantic component has to beat, and the configuration the project was
# tuned to before the fusion was corrected.
STRUCTURE_ONLY = {"alpha": 0.00, "beta": 0.35, "gamma": 0.45, "delta": 0.20}

PCA_COMPONENTS = 20
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 2

# The two tables that alone supply every `giant_table` label (90 of 243
# columns). Excluding them isolates the anti-patterns a per-column encoder can
# actually see — see `Dataset.without_tables`.
GIANT_TABLES = frozenset({"TABLA_BASE_DATOS", "BACKUP_DATOS"})


@dataclass
class Dataset:
    """The four feature blocks plus the ground truth, all in one column order."""

    e_text: np.ndarray
    e_type: np.ndarray
    e_rest: np.ndarray
    e_stat: np.ndarray
    column_index: list[str]
    truth: list[str]

    def __post_init__(self) -> None:
        lengths = {
            len(self.column_index),
            len(self.truth),
            *(b.shape[0] for b in self.blocks),
        }
        if len(lengths) != 1:
            raise ValueError(f"every block must have the same number of rows, got {sorted(lengths)}")

    @property
    def blocks(self) -> tuple[np.ndarray, ...]:
        return (self.e_text, self.e_type, self.e_rest, self.e_stat)

    @property
    def n_columns(self) -> int:
        return len(self.column_index)

    @property
    def table_of(self) -> list[str]:
        """Table name per column, taken from the "TABLE.COLUMN" index."""
        return [key.split(".", 1)[0] for key in self.column_index]

    def without_tables(self, tables: set[str] | frozenset[str]) -> Dataset:
        """A copy with every column of the named tables removed."""
        keep = [i for i, t in enumerate(self.table_of) if t not in tables]
        return Dataset(
            e_text=self.e_text[keep],
            e_type=self.e_type[keep],
            e_rest=self.e_rest[keep],
            e_stat=self.e_stat[keep],
            column_index=[self.column_index[i] for i in keep],
            truth=[self.truth[i] for i in keep],
        )


@dataclass
class Evaluation:
    """A scored run, with the per-column detail the score is aggregated from."""

    score: ClusterScore
    cluster_ids: list[int]
    predicted: list[str]


def load_dataset(
    pickle_path: str | Path = "output/intermediate_02.pkl",
    labels_path: str | Path = "output/manual_labels.csv",
) -> Dataset:
    """Read the stored feature blocks and pair them with the manual labels."""
    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)
    column_index: list[str] = data["column_index"]
    return Dataset(
        e_text=data["e_text"],
        e_type=data["e_type"],
        e_rest=data["e_rest"],
        e_stat=data["e_stat"],
        column_index=column_index,
        truth=load_manual_labels(labels_path, column_index),
    )


def run_clustering(
    dataset: Dataset,
    weights: dict[str, float],
    normalize: str = "zscore",
    n_components: int = PCA_COMPONENTS,
    reducer: str = "pca",
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = "euclidean",
) -> np.ndarray:
    """Fuse the blocks, reduce, cluster. Returns raw cluster ids (-1 = noise)."""
    from db_bad_clust.clustering.cluster_engine import ClusterEngine
    from db_bad_clust.clustering.dimensionality_reducer import DimensionalityReducer
    from db_bad_clust.features.feature_builder import FeatureBuilder

    phi = FeatureBuilder(**weights).build(
        e_text=dataset.e_text,
        e_type=dataset.e_type,
        e_rest=dataset.e_rest,
        e_stat=dataset.e_stat,
    )
    reduced = DimensionalityReducer(
        method=reducer,
        n_components=min(n_components, *phi.shape),
        random_state=42,
    ).fit_transform(phi)

    return ClusterEngine(
        method="hdbscan",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    ).fit_predict(reduced)


def evaluate(
    name: str,
    dataset: Dataset,
    weights: dict[str, float],
    **kwargs: object,
) -> Evaluation:
    """Run one configuration end to end and score it under both rulers.

    Accuracy and F1 come from majority-vote cluster naming, which uses the
    ground truth the pipeline never saw — they are an upper bound. ARI and NMI
    are computed on the raw partition and carry no such handicap.
    """
    cluster_ids = run_clustering(dataset, weights, **kwargs)  # type: ignore[arg-type]
    predicted = clusters_to_labels(cluster_ids, dataset.truth)
    return Evaluation(
        score=score(name, predicted, dataset.truth, group_ids=cluster_ids),
        cluster_ids=[int(c) for c in cluster_ids],
        predicted=predicted,
    )


def format_table(rows: list[ClusterScore]) -> str:
    """Render scored configurations as one comparable table."""
    lines = [
        f"{'Configuration':<28} {'ARI':>8} {'NMI':>8} {'Accuracy':>10} {'F1-macro':>10} {'Groups':>8}",
        "-" * 76,
    ]
    lines += [
        f"{r.name:<28} {r.ari:>8.4f} {r.nmi:>8.4f} {r.accuracy:>10.4f} "
        f"{r.f1_macro:>10.4f} {r.n_groups:>8}"
        for r in rows
    ]
    lines.append("")
    lines.append("Accuracy/F1 use ground-truth-assisted cluster naming — an upper bound.")
    return "\n".join(lines)
