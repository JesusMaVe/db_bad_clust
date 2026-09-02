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
    """Read the stored feature blocks and pair them with the manual labels.

    The blocks are stored as float32 (BERT's output dtype) and are widened to
    float64 here on purpose: with a degenerate structural representation the
    clustering is decided by rounding, and float32 arithmetic moves the
    structure-only ARI by 0.3154 (see `precision_sensitivity`). Pinning the
    working width makes every reported number reproducible; the narrow width
    stays available as a diagnostic rather than as an accident.
    """
    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)
    column_index: list[str] = data["column_index"]
    return Dataset(
        e_text=np.asarray(data["e_text"], dtype=np.float64),
        e_type=np.asarray(data["e_type"], dtype=np.float64),
        e_rest=np.asarray(data["e_rest"], dtype=np.float64),
        e_stat=np.asarray(data["e_stat"], dtype=np.float64),
        column_index=column_index,
        truth=load_manual_labels(labels_path, column_index),
    )


def build_phi(
    dataset: Dataset,
    weights: dict[str, float],
    normalize: str = "block",
) -> np.ndarray:
    """Fuse the four blocks into the composite vector φ.

    Separated from `run_clustering` because the fusion is where the weights
    either mean something or do not — see `FeatureBuilder`'s docstring — and
    an ablation needs to inspect it without clustering.
    """
    from db_bad_clust.features.feature_builder import FeatureBuilder

    return FeatureBuilder(**weights, normalize=normalize).build(
        e_text=dataset.e_text,
        e_type=dataset.e_type,
        e_rest=dataset.e_rest,
        e_stat=dataset.e_stat,
    )


def block_variance_shares(dataset: Dataset, phi: np.ndarray) -> dict[str, float]:
    """The share of φ's variance each block actually holds.

    The measurement that shows a nominal weight is not the effective one.
    """
    widths = [b.shape[1] for b in dataset.blocks]
    names = ["e_text", "e_type", "e_rest", "e_stat"]
    variances, start = {}, 0
    for name, width in zip(names, widths, strict=True):
        variances[name] = float(phi[:, start : start + width].var(axis=0).sum())
        start += width
    total = sum(variances.values()) or 1.0
    return {k: v / total for k, v in variances.items()}


# ── Diagnostics: is the representation able to say anything? ──────────


def representation_degeneracy(
    dataset: Dataset,
    weights: dict[str, float],
    normalize: str = "block",
    decimals: int = 9,
) -> dict[str, float | int]:
    """How many of the columns does this representation actually tell apart?

    A clustering can only separate points the features distinguish. If two
    columns map to the same vector, no algorithm can ever assign them different
    labels — they are one point wearing two names. Measured on the real corpus,
    the structure-only configuration collapses 243 columns into 24 distinct
    vectors (97% of rows are a duplicate of another), with one tie group of 108.
    Its ARI is therefore a statement about 24 points, not 243.

    Returns n_columns, n_distinct, duplicate_fraction and largest_tie_group.
    """
    from collections import Counter

    phi = build_phi(dataset, weights, normalize=normalize)
    groups = Counter(tuple(np.round(row, decimals)) for row in phi)
    duplicated = sum(count for count in groups.values() if count > 1)
    return {
        "n_columns": len(phi),
        "n_distinct": len(groups),
        "duplicate_fraction": duplicated / len(phi) if len(phi) else 0.0,
        "largest_tie_group": max(groups.values()) if groups else 0,
    }


def precision_sensitivity(
    dataset: Dataset,
    weights: dict[str, float],
    normalize: str = "block",
    **kwargs: object,
) -> dict[str, float]:
    """Score the same configuration in float32 and float64 arithmetic.

    Tied points have no well-defined neighbourhood, so a representation full of
    duplicates leaves the clustering to be decided by rounding. The gap between
    the two precisions is how much of the score is arithmetic rather than
    signal: on the real corpus, structure-only moves 0.1877 → 0.5031 (gap
    0.3154) while any configuration with the embeddings on moves not at all.
    """
    scores = {}
    for name, dtype in (("float32", np.float32), ("float64", np.float64)):
        cast = Dataset(
            e_text=dataset.e_text.astype(dtype),
            e_type=dataset.e_type.astype(dtype),
            e_rest=dataset.e_rest.astype(dtype),
            e_stat=dataset.e_stat.astype(dtype),
            column_index=dataset.column_index,
            truth=dataset.truth,
        )
        phi = build_phi(cast, weights, normalize=normalize).astype(dtype)
        scores[name] = _score_phi(phi, cast, **kwargs).ari

    return {
        "ari_float32": scores["float32"],
        "ari_float64": scores["float64"],
        "gap": abs(scores["float32"] - scores["float64"]),
    }


def run_clustering(
    dataset: Dataset,
    weights: dict[str, float],
    normalize: str = "block",
    n_components: int = PCA_COMPONENTS,
    reducer: str = "pca",
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = "euclidean",
) -> np.ndarray:
    """Fuse the blocks, reduce, cluster. Returns raw cluster ids (-1 = noise)."""
    return _cluster(
        build_phi(dataset, weights, normalize=normalize),
        n_components=n_components,
        reducer=reducer,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )


def _cluster(
    phi: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    reducer: str = "pca",
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = "euclidean",
) -> np.ndarray:
    """Reduce an already-fused matrix and cluster it."""
    from db_bad_clust.clustering.cluster_engine import ClusterEngine
    from db_bad_clust.clustering.dimensionality_reducer import DimensionalityReducer

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


def _score_phi(
    phi: np.ndarray,
    dataset: Dataset,
    n_components: int = PCA_COMPONENTS,
    reducer: str = "pca",
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = "euclidean",
) -> ClusterScore:
    """Reduce, cluster and score an already-fused feature matrix."""
    cluster_ids = _cluster(
        phi,
        n_components=n_components,
        reducer=reducer,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    predicted = clusters_to_labels(cluster_ids, dataset.truth)
    return score("phi", predicted, dataset.truth, group_ids=cluster_ids)


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


def weights_for(alpha: float, base: dict[str, float] | None = None) -> dict[str, float]:
    """Weights at a given alpha, the rest sharing what is left in fixed ratio.

    Keeps the structural blocks' relative proportions constant so a sweep
    varies exactly one thing: how much say the semantic block gets.
    """
    base = base or STRUCTURE_ONLY
    structural = base["beta"] + base["gamma"] + base["delta"]
    scale = (1.0 - alpha) / structural if structural else 0.0
    return {
        "alpha": alpha,
        "beta": base["beta"] * scale,
        "gamma": base["gamma"] * scale,
        "delta": base["delta"] * scale,
    }


def sweep(
    dataset: Dataset,
    alphas: tuple[float, ...] = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00),
    normalize: str = "block",
    **kwargs: object,
) -> list[ClusterScore]:
    """Score the pipeline across the semantic block's share of the space.

    With `normalize="block"` this is a real sweep. With `normalize="zscore"`
    it is the historical switch: every alpha above zero hands the embeddings
    ~90% of the variance regardless of what the number says.
    """
    return [
        evaluate(f"alpha={alpha:.2f}", dataset, weights_for(alpha, STRUCTURE_ONLY), normalize=normalize, **kwargs).score  # type: ignore[arg-type]
        for alpha in alphas
    ]


def ablation(
    datasets: dict[str, Dataset],
    alpha: float = 1.00,
    **kwargs: object,
) -> list[ClusterScore]:
    """Score each semantic representation at the same weight, plus the floor.

    The structure-only row is computed from whichever dataset comes first:
    at alpha=0 the semantic block is switched off, so every dataset gives the
    same answer, and computing it once keeps the comparison honest.
    """
    if not datasets:
        return []
    first = next(iter(datasets.values()))
    rows = [
        evaluate("structure only (alpha=0)", first, weights_for(0.0), **kwargs).score  # type: ignore[arg-type]
    ]
    rows += [
        evaluate(name, dataset, weights_for(alpha), **kwargs).score  # type: ignore[arg-type]
        for name, dataset in datasets.items()
    ]
    return rows


def format_diagnostics(configs: dict[str, tuple[Dataset, dict[str, float]]]) -> str:
    """Report, per configuration, whether its score can mean anything.

    Each entry pairs a name with the dataset AND the weights it was scored
    under — an ablation compares different feature blocks, so a diagnostic
    that reused one dataset for every name would report the same numbers
    under three different labels.

    Two questions, in order. How many columns does the representation tell
    apart? And does the score survive a change of floating-point width? A
    configuration that fails the first will usually fail the second, because
    tied points leave the clustering to be settled by rounding.
    """
    lines = [
        "Can this representation say anything?",
        "-" * 76,
        f"{'Configuration':<28} {'distinct':>12} {'duplicated':>11} {'largest tie':>12}",
    ]
    for name, (dataset, weights) in configs.items():
        d = representation_degeneracy(dataset, weights)
        lines.append(
            f"{name:<28} {d['n_distinct']:>5}/{d['n_columns']:<6} "
            f"{d['duplicate_fraction']:>10.0%} {d['largest_tie_group']:>12}"
        )
    lines += [
        "",
        f"{'Configuration':<28} {'ARI float32':>12} {'ARI float64':>12} {'gap':>12}",
    ]
    for name, (dataset, weights) in configs.items():
        p = precision_sensitivity(dataset, weights)
        lines.append(
            f"{name:<28} {p['ari_float32']:>12.4f} {p['ari_float64']:>12.4f} {p['gap']:>12.4f}"
        )
    lines += [
        "",
        "Columns that share a vector cannot be given different labels by any",
        "algorithm, and a gap between the two precisions is the part of the score",
        "that is arithmetic rather than signal.",
    ]
    return "\n".join(lines)


def format_table(rows: list[ClusterScore]) -> str:
    """Render scored configurations as one comparable table."""
    lines = [
        f"{'Configuration':<28} {'ARI':>8} {'NMI':>8} {'AMI':>8} {'V':>8} "
        f"{'Accuracy':>9} {'F1-macro':>9} {'k':>4}",
        "-" * 92,
    ]
    lines += [
        f"{r.name:<28} {r.ari:>8.4f} {r.nmi:>8.4f} {r.ami:>8.4f} {r.v_measure:>8.4f} "
        f"{r.accuracy:>9.4f} {r.f1_macro:>9.4f} {r.n_groups:>4}"
        for r in rows
    ]
    lines.append("")
    lines.append("Accuracy/F1 use ground-truth-assisted cluster naming — an upper bound.")
    lines.append("AMI, not NMI, is the one to compare across rows with different k.")
    return "\n".join(lines)
