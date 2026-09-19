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
from typing import Any

import numpy as np

from db_bad_clust.evaluation.cluster_scoring import (
    ClusterScore,
    clusters_to_labels,
    load_manual_labels,
    score,
)
from db_bad_clust.evaluation.report_table import Column, render_table

# Structure only: alpha=0 switches the embeddings off. This is the reference
# the semantic component has to beat, and the configuration the project was
# tuned to before the fusion was corrected.
STRUCTURE_ONLY = {"alpha": 0.00, "beta": 0.35, "gamma": 0.45, "delta": 0.20}

# min_samples=3 beats the previous default of 2 on every metric (ARI/NMI/AMI/
# accuracy/F1-macro) on the honest without-giants evaluation, and is a wash on
# the full corpus — measured via a grid sweep against cluster_selection_method
# too (eom vs leaf made no difference at this corpus size). See
# docs/research_improving_clustering.md, candidato #1.
PCA_COMPONENTS = 20
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3

# The conflict representation (features/semantic_anchors.py, ConflictBlock):
# the embeddings are off and the space is name-expectation-minus-declared-type
# plus the constraints. Every clean column sits near the origin whatever its
# topic, so the clusters that form are kinds of conflict, not tables — the
# ARI against the table drops from 0.593 (document) to ~0.02. CONFLICT_FUSED
# lets a little of the document and the per-table aggregates back in.
CONFLICT = {"alpha": 0.0, "beta": 0.0, "gamma": 0.5, "delta": 0.0, "epsilon": 0.0, "zeta": 1.0}
CONFLICT_FUSED = {"alpha": 0.2, "beta": 0.0, "gamma": 0.5, "delta": 0.0, "epsilon": 0.3, "zeta": 1.0}

# The HDBSCAN grid `stability_sweep` walks. The conflict space is 18-dimensional
# and 153-243 points wide, and its density structure is knife-edge: ARI moves
# 0.09 → 0.21 between min_cluster_size 5 and 8. The sweep exists so that the
# operating point is chosen by HDBSCAN's own relative validity, not by the
# labels — and so the whole grid is reported, not the best cell.
STABILITY_GRID: tuple[tuple[int, int], ...] = tuple(
    (mcs, ms) for mcs in (5, 6, 7, 8, 10) for ms in (2, 3, 5)
)

# Ward's k is chosen by silhouette over this range (candidato #7). On the
# conflict representation the silhouette peaks at k=17 over the whole 2..40
# range, so the upper bound is not what picks it; 2 stays in so that a
# representation that really does split in two is allowed to say so.
WARD_K_RANGE: tuple[int, int] = (2, 30)
BLIND_ALGORITHMS = ("ward", "hdbscan")

# The two tables that alone supply every `giant_table` label (90 of 243
# columns). Excluding them isolates the anti-patterns a per-column encoder can
# actually see — see `Dataset.without_tables`.
GIANT_TABLES = frozenset({"TABLA_BASE_DATOS", "BACKUP_DATOS"})


@dataclass
class Dataset:
    """The four feature blocks, all in one column order.

    `truth` is optional: a labeled reference corpus carries it, but a target
    database being scored by `evaluation.health_report` has none — the pipeline
    up to this point (extraction, documents, embeddings) never needed labels
    (see `load_dataset`), and this dataclass is the single place that loading
    happens, so it should not force a label that isn't there.

    `schema` is the raw `DatabaseSchema` from the pickle, when present — needed
    by callers that read table-level metadata (e.g. `health_report.py`'s
    per-table column count), not by the clustering/scoring path itself.

    `e_table` is the per-table aggregate block (features/table_aggregates.py),
    computed once by `load_dataset` from `schema` — pure computation, no need
    to touch Oracle/BERT or regenerate any pickle. `build_phi` passes it to
    `FeatureBuilder` alongside epsilon; every existing weights dict lacks an
    "epsilon" key, so it defaults to 0.0 and this block contributes nothing
    unless a caller explicitly asks for it.

    `e_conflict` is the raw conflict block (features/semantic_anchors.py,
    ConflictBlock), stored by build_embeddings.py because it needs the encoder.
    Same contract as `e_table`: weighted by "zeta", absent from every existing
    weights dict, so nothing changes until a caller asks. None for pickles
    built before it existed.

    `e_conflict_variants` holds the same block under each anchor wording
    (TYPE_FAMILY_ANCHOR_VARIANTS), for `robustness_over_wordings`.
    """

    e_text: np.ndarray
    e_type: np.ndarray
    e_rest: np.ndarray
    e_stat: np.ndarray
    column_index: list[str]
    truth: list[str] | None = None
    schema: Any = None
    e_table: np.ndarray | None = None
    e_conflict: np.ndarray | None = None
    e_conflict_variants: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        lengths = {len(self.column_index), *(b.shape[0] for b in self.blocks)}
        if self.truth is not None:
            lengths.add(len(self.truth))
        if self.e_table is not None:
            lengths.add(self.e_table.shape[0])
        if self.e_conflict is not None:
            lengths.add(self.e_conflict.shape[0])
        for variant in (self.e_conflict_variants or {}).values():
            lengths.add(variant.shape[0])
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
            truth=[self.truth[i] for i in keep] if self.truth is not None else None,
            schema=self.schema,
            e_table=self.e_table[keep] if self.e_table is not None else None,
            e_conflict=self.e_conflict[keep] if self.e_conflict is not None else None,
            e_conflict_variants=(
                {name: block[keep] for name, block in self.e_conflict_variants.items()}
                if self.e_conflict_variants is not None
                else None
            ),
        )

    def with_conflict(self, e_conflict: np.ndarray) -> Dataset:
        """A copy whose conflict block is replaced — every other block shared."""
        return Dataset(
            e_text=self.e_text,
            e_type=self.e_type,
            e_rest=self.e_rest,
            e_stat=self.e_stat,
            column_index=self.column_index,
            truth=self.truth,
            schema=self.schema,
            e_table=self.e_table,
            e_conflict=e_conflict,
            e_conflict_variants=self.e_conflict_variants,
        )


@dataclass
class Evaluation:
    """A scored run, with the per-column detail the score is aggregated from."""

    score: ClusterScore
    cluster_ids: list[int]
    predicted: list[str]


def load_dataset(
    pickle_path: str | Path = "output/intermediate_02.pkl",
    labels_path: str | Path | None = "output/manual_labels.csv",
) -> Dataset:
    """Read the stored feature blocks, optionally paired with the manual labels.

    This is the one place a build_embeddings.py-shaped pickle gets opened and
    its blocks widened to float64 — every other reader of these pickles
    (ml_baselines.py, evaluation/health_report.py) goes through this function
    rather than re-opening the file, so the float64 pin below applies
    everywhere a pickle is read, not just here.

    The blocks are stored as float32 (BERT's output dtype) and are widened to
    float64 here on purpose: with a degenerate structural representation the
    clustering is decided by rounding, and float32 arithmetic moves the
    structure-only ARI by 0.3154 (see `precision_sensitivity`). Pinning the
    working width makes every reported number reproducible; the narrow width
    stays available as a diagnostic rather than as an accident.

    labels_path=None skips loading ground truth — the extraction/embedding
    pipeline never needed labels to begin with (see CLAUDE.md), and a target
    database being scored by health_report.py has none to load.
    """
    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)
    column_index: list[str] = data["column_index"]
    truth = load_manual_labels(labels_path, column_index) if labels_path is not None else None
    schema = data.get("schema")
    e_table = None
    if schema is not None:
        from db_bad_clust.features.table_aggregates import build_table_block

        e_table = build_table_block(schema, column_index)
    e_conflict = data.get("e_conflict")
    variants = data.get("e_conflict_variants")
    return Dataset(
        e_text=np.asarray(data["e_text"], dtype=np.float64),
        e_type=np.asarray(data["e_type"], dtype=np.float64),
        e_rest=np.asarray(data["e_rest"], dtype=np.float64),
        e_stat=np.asarray(data["e_stat"], dtype=np.float64),
        column_index=column_index,
        truth=truth,
        schema=schema,
        e_table=e_table,
        e_conflict=np.asarray(e_conflict, dtype=np.float64) if e_conflict is not None else None,
        e_conflict_variants=(
            {name: np.asarray(block, dtype=np.float64) for name, block in variants.items()}
            if variants is not None
            else None
        ),
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
        e_table=dataset.e_table,
        e_conflict=dataset.e_conflict,
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
            e_table=dataset.e_table.astype(dtype) if dataset.e_table is not None else None,
            e_conflict=(
                dataset.e_conflict.astype(dtype) if dataset.e_conflict is not None else None
            ),
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
    reducer_kwargs: dict[str, Any] | None = None,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> np.ndarray:
    """Fuse the blocks, reduce, cluster. Returns raw cluster ids (-1 = noise)."""
    return _cluster(
        build_phi(dataset, weights, normalize=normalize),
        n_components=n_components,
        reducer=reducer,
        reducer_kwargs=reducer_kwargs,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )


def _cluster(
    phi: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    reducer: str = "pca",
    reducer_kwargs: dict[str, Any] | None = None,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> np.ndarray:
    """Reduce an already-fused matrix and cluster it.

    `reducer_kwargs` reaches `DimensionalityReducer` unchanged — e.g.
    `{"n_neighbors": 10, "min_dist": 0.0}` for `reducer="umap"`. Without it,
    UMAP silently runs on defaults tuned in the reducer for much larger
    datasets than this corpus (see docs/research_improving_clustering.md,
    Tema 3.2).
    """
    labels, _ = _cluster_with_info(
        phi,
        n_components=n_components,
        reducer=reducer,
        reducer_kwargs=reducer_kwargs,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    return labels


def _cluster_with_info(
    phi: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    reducer: str = "pca",
    reducer_kwargs: dict[str, Any] | None = None,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> tuple[np.ndarray, dict[str, Any]]:
    """`_cluster`, plus the engine's `cluster_info_` (relative validity, noise)."""
    from db_bad_clust.clustering.cluster_engine import ClusterEngine
    from db_bad_clust.clustering.dimensionality_reducer import DimensionalityReducer

    reduced = DimensionalityReducer(
        method=reducer,
        n_components=min(n_components, *phi.shape),
        random_state=42,
        **(reducer_kwargs or {}),
    ).fit_transform(phi)

    engine = ClusterEngine(
        method="hdbscan",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = engine.fit_predict(reduced)
    return labels, dict(engine.cluster_info_ or {})


def _score_phi(
    phi: np.ndarray,
    dataset: Dataset,
    n_components: int = PCA_COMPONENTS,
    reducer: str = "pca",
    reducer_kwargs: dict[str, Any] | None = None,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = HDBSCAN_MIN_SAMPLES,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> ClusterScore:
    """Reduce, cluster and score an already-fused feature matrix."""
    cluster_ids = _cluster(
        phi,
        n_components=n_components,
        reducer=reducer,
        reducer_kwargs=reducer_kwargs,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    predicted = clusters_to_labels(cluster_ids, dataset.truth)
    return score("phi", predicted, dataset.truth, group_ids=cluster_ids, table_of=dataset.table_of)


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
        score=score(
            name, predicted, dataset.truth, group_ids=cluster_ids, table_of=dataset.table_of
        ),
        cluster_ids=[int(c) for c in cluster_ids],
        predicted=predicted,
    )


@dataclass
class StabilityRow:
    """One cell of the HDBSCAN grid: its score and its label-free quality."""

    min_cluster_size: int
    min_samples: int
    score: ClusterScore
    relative_validity: float
    noise_fraction: float


def stability_sweep(
    dataset: Dataset,
    weights: dict[str, float],
    grid: tuple[tuple[int, int], ...] = STABILITY_GRID,
    normalize: str = "block",
    **kwargs: object,
) -> list[StabilityRow]:
    """Score one representation across the HDBSCAN grid, with a label-free pick.

    Every earlier hyper-parameter choice in this repo (min_samples=3 among
    them) was made by looking at the labelled score, which is tuning on the
    test set by another name. Each row here also carries HDBSCAN's own
    `relative_validity_` — a density-based quality estimate that never sees
    the ground truth — so `choose_operating_point` can pick a cell without
    peeking. The full grid is the deliverable; the chosen cell is just the one
    that gets quoted.
    """
    phi = build_phi(dataset, weights, normalize=normalize)
    rows = []
    for min_cluster_size, min_samples in grid:
        cluster_ids, info = _cluster_with_info(
            phi,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            **kwargs,  # type: ignore[arg-type]
        )
        predicted = clusters_to_labels(cluster_ids, dataset.truth)
        rows.append(
            StabilityRow(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                score=score(
                    f"mcs={min_cluster_size} ms={min_samples}",
                    predicted,
                    dataset.truth,
                    group_ids=cluster_ids,
                    table_of=dataset.table_of,
                ),
                relative_validity=float(info.get("relative_validity", float("nan"))),
                noise_fraction=float((np.asarray(cluster_ids) == -1).mean()),
            )
        )
    return rows


def choose_operating_point(rows: list[StabilityRow]) -> StabilityRow | None:
    """The grid cell with the highest relative validity — chosen blind to the labels.

    Cells whose validity is undefined (all noise, a single cluster) are never
    chosen; if every cell is undefined there is no unsupervised pick and the
    caller has to say so rather than fall back to the best labelled score.
    """
    candidates = [r for r in rows if r.relative_validity == r.relative_validity]
    return max(candidates, key=lambda r: r.relative_validity) if candidates else None


def cluster_composition(dataset: Dataset, cluster_ids: list[int] | np.ndarray) -> str:
    """One line per cluster: size, top labels, and how many tables it spans.

    The last number is the check that matters for this branch. A cluster that
    is one table has recognised the table; one that spans twelve has found
    something the columns share across tables — an anti-pattern, if the top
    label agrees.
    """
    from collections import Counter

    ids = np.asarray(cluster_ids)
    truth = np.asarray(dataset.truth)
    tables = np.asarray(dataset.table_of)
    names = [key.split(".", 1)[1] for key in dataset.column_index]
    lines = []
    for cid in sorted(set(ids.tolist())):
        mask = ids == cid
        top = ", ".join(f"{label} {n}" for label, n in Counter(truth[mask]).most_common(3))
        members = [n for n, m in zip(names, mask, strict=True) if m]
        sample = ", ".join(members[:6])
        lines.append(
            f"  {'noise' if cid == -1 else f'cluster {cid}':>10}  n={int(mask.sum()):3d}  "
            f"tables={len(set(tables[mask])):2d}  [{top}]  e.g. {sample}"
        )
    return "\n".join(lines)


# ── Blind evaluation: every hyper-parameter chosen without the labels ──


def _reduce(phi: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    """The same PCA step `_cluster` applies, exposed for the blind paths."""
    from db_bad_clust.clustering.dimensionality_reducer import DimensionalityReducer

    return DimensionalityReducer(
        method="pca", n_components=min(n_components, *phi.shape), random_state=42
    ).fit_transform(phi)


def _cluster_ward(
    reduced: np.ndarray, k_range: tuple[int, int] = WARD_K_RANGE
) -> tuple[np.ndarray, int, float]:
    """Ward at every k in the range; keep the partition with the best silhouette.

    Returns (labels, chosen k, its silhouette). Ward assigns every column, so
    there is no noise class to lump — the reason it replaced HDBSCAN as the
    main algorithm in candidato #7, where a third of the columns were noise.
    """
    from sklearn.metrics import silhouette_score

    from db_bad_clust.clustering.cluster_engine import ClusterEngine

    low, high = k_range
    high = min(high, reduced.shape[0] - 1)
    best: tuple[np.ndarray, int, float] | None = None
    for k in range(max(2, low), high + 1):
        labels = ClusterEngine(method="agglomerative", n_clusters=k).fit_predict(reduced)
        if len(set(labels.tolist())) < 2:
            continue
        sil = float(silhouette_score(reduced, labels))
        if best is None or sil > best[2]:
            best = (labels, k, sil)
    if best is None:
        return np.zeros(reduced.shape[0], dtype=int), 1, float("nan")
    return best


def reassign_noise_knn(reduced: np.ndarray, labels: np.ndarray, k: int = 3) -> np.ndarray:
    """Give every HDBSCAN noise point the majority cluster of its k clustered neighbours.

    The scoring treats -1 as one more cluster, so leaving a third of the corpus
    as noise lumps unrelated columns into one fake group. Clustered points keep
    their label. Without any clustered point there is nothing to vote with and
    the labels come back unchanged.
    """
    from sklearn.neighbors import KNeighborsClassifier

    labels = np.asarray(labels).copy()
    noise = labels == -1
    clustered = ~noise
    if not noise.any() or clustered.sum() == 0:
        return labels
    knn = KNeighborsClassifier(n_neighbors=min(k, int(clustered.sum())))
    knn.fit(reduced[clustered], labels[clustered])
    labels[noise] = knn.predict(reduced[noise])
    return labels


@dataclass
class BlindEvaluation:
    """A run whose hyper-parameter was chosen by an internal criterion alone."""

    evaluation: Evaluation
    algorithm: str
    chosen: str  # "k=17" or "mcs=6 ms=3"
    criterion: float  # silhouette (ward) or relative validity (hdbscan)


def evaluate_blind(
    name: str,
    dataset: Dataset,
    weights: dict[str, float],
    algorithm: str = "ward",
    grid: tuple[tuple[int, int], ...] = STABILITY_GRID,
    k_range: tuple[int, int] = WARD_K_RANGE,
) -> BlindEvaluation:
    """Fuse, reduce, cluster with a label-free choice of hyper-parameter, then score.

    ward:    k by silhouette over `k_range`.
    hdbscan: the grid cell with the highest relative validity, then noise
             reassigned to its nearest clusters (`reassign_noise_knn`).

    The labels are used only after the partition exists, to score it.
    """
    if algorithm not in BLIND_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {BLIND_ALGORITHMS}, got {algorithm!r}")
    reduced = _reduce(build_phi(dataset, weights))

    if algorithm == "ward":
        labels, k, criterion = _cluster_ward(reduced, k_range)
        chosen = f"k={k}"
    else:
        from db_bad_clust.clustering.cluster_engine import ClusterEngine

        best: tuple[float, np.ndarray, int, int] | None = None
        for mcs, ms in grid:
            engine = ClusterEngine(method="hdbscan", min_cluster_size=mcs, min_samples=ms)
            cell = engine.fit_predict(reduced)
            validity = float((engine.cluster_info_ or {}).get("relative_validity", float("nan")))
            if validity != validity:
                continue
            if best is None or validity > best[0]:
                best = (validity, cell, mcs, ms)
        if best is None:
            labels, criterion, chosen = np.full(reduced.shape[0], -1), float("nan"), "none"
        else:
            criterion, cell, mcs, ms = best
            labels = reassign_noise_knn(reduced, cell)
            chosen = f"mcs={mcs} ms={ms}"

    predicted = clusters_to_labels(labels, dataset.truth)
    evaluation = Evaluation(
        score=score(name, predicted, dataset.truth, group_ids=labels, table_of=dataset.table_of),
        cluster_ids=[int(c) for c in labels],
        predicted=predicted,
    )
    return BlindEvaluation(evaluation, algorithm, chosen, criterion)


@dataclass
class RobustnessReport:
    """One blind run per anchor wording, and what they add up to."""

    algorithm: str
    rows: list[tuple[str, BlindEvaluation]]

    def _values(self, attr: str) -> np.ndarray:
        return np.array([getattr(run.evaluation.score, attr) for _, run in self.rows])

    def summary(self, attr: str) -> tuple[float, float, float]:
        """(mean, min, max) of a ClusterScore field across wordings."""
        values = self._values(attr)
        return float(values.mean()), float(values.min()), float(values.max())

    @property
    def smallest_k(self) -> int:
        return min(run.evaluation.score.n_groups for _, run in self.rows)


def robustness_over_wordings(
    dataset: Dataset,
    weights: dict[str, float],
    algorithm: str = "ward",
) -> RobustnessReport:
    """Re-run `evaluate_blind` once per anchor wording in `e_conflict_variants`.

    The anchor wording is part of the conflict representation, and the one in
    use was chosen while looking at the labelled score. The mean over the
    wordings is the number to quote; the spread is how much of it is wording.
    """
    if not dataset.e_conflict_variants:
        raise ValueError("this dataset carries no e_conflict_variants")
    rows = [
        (
            wording,
            evaluate_blind(wording, dataset.with_conflict(block), weights, algorithm=algorithm),
        )
        for wording, block in dataset.e_conflict_variants.items()
    ]
    return RobustnessReport(algorithm, rows)


def evaluate_late_fusion(
    name: str,
    dataset: Dataset,
    views: list[dict[str, float]],
    distance_threshold: float = 0.5,
    linkage: str = "average",
    **kwargs: object,
) -> Evaluation:
    """Cluster each view separately, fuse the PARTITIONS via co-association,
    score the consensus partition — late/partition-level fusion, an
    alternative to `evaluate`'s early/feature-level (weighted concatenation)
    fusion. See clustering/late_fusion.py and
    docs/research_improving_clustering.md, candidato #5.

    `views` is a list of weights dicts, one per view (e.g. a document-only
    view and a structure-only view) — each gets its own `run_clustering` call
    under the SAME reduction/HDBSCAN settings (`**kwargs`), so the only thing
    that varies between views is which blocks are switched on.
    """
    from db_bad_clust.clustering.late_fusion import consensus_clustering

    view_labels = [run_clustering(dataset, w, **kwargs) for w in views]  # type: ignore[arg-type]
    cluster_ids = consensus_clustering(
        view_labels, distance_threshold=distance_threshold, linkage=linkage
    )
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
    degeneracy_columns = [
        Column("Configuration", 28, "<"),
        Column("distinct", 12),
        Column("duplicated", 11),
        Column("largest tie", 12),
    ]
    degeneracy_rows = []
    for name, (dataset, weights) in configs.items():
        d = representation_degeneracy(dataset, weights)
        degeneracy_rows.append(
            [
                name,
                f"{d['n_distinct']}/{d['n_columns']}",
                f"{d['duplicate_fraction']:.0%}",
                str(d["largest_tie_group"]),
            ]
        )

    precision_columns = [
        Column("Configuration", 28, "<"),
        Column("ARI float32", 12),
        Column("ARI float64", 12),
        Column("gap", 12),
    ]
    precision_rows = []
    for name, (dataset, weights) in configs.items():
        p = precision_sensitivity(dataset, weights)
        precision_rows.append(
            [name, f"{p['ari_float32']:.4f}", f"{p['ari_float64']:.4f}", f"{p['gap']:.4f}"]
        )

    lines = [
        "Can this representation say anything?",
        render_table(degeneracy_columns, degeneracy_rows),
        "",
        render_table(precision_columns, precision_rows),
        "",
        "Columns that share a vector cannot be given different labels by any",
        "algorithm, and a gap between the two precisions is the part of the score",
        "that is arithmetic rather than signal.",
    ]
    return "\n".join(lines)


TABLE_SCORE_COLUMNS = [
    Column("Configuration", 28, "<"),
    Column("ARI", 8),
    Column("NMI", 8),
    Column("AMI", 8),
    Column("V", 8),
    Column("Accuracy", 9),
    Column("F1-macro", 9),
    Column("k", 4),
    Column("ARI~tbl", 8),
]


def _fmt_table_ari(value: float) -> str:
    return f"{value:.4f}" if value == value else "n/a"


def format_table(rows: list[ClusterScore]) -> str:
    """Render scored configurations as one comparable table."""
    table_rows = [
        [
            r.name,
            f"{r.ari:.4f}",
            f"{r.nmi:.4f}",
            f"{r.ami:.4f}",
            f"{r.v_measure:.4f}",
            f"{r.accuracy:.4f}",
            f"{r.f1_macro:.4f}",
            str(r.n_groups),
            _fmt_table_ari(r.table_ari),
        ]
        for r in rows
    ]
    lines = [
        render_table(TABLE_SCORE_COLUMNS, table_rows),
        "",
        "Accuracy/F1 use ground-truth-assisted cluster naming — an upper bound.",
        "AMI, not NMI, is the one to compare across rows with different k.",
        "ARI~tbl is the partition's ARI against the TABLE of each column: table",
        "identity leaking into the clusters, not anti-pattern detection.",
    ]
    return "\n".join(lines)


STABILITY_COLUMNS = [
    Column("mcs", 4),
    Column("ms", 3),
    Column("ARI", 8),
    Column("AMI", 8),
    Column("F1-macro", 9),
    Column("k", 4),
    Column("noise", 6),
    Column("ARI~tbl", 8),
    Column("validity", 9),
]


def format_stability(rows: list[StabilityRow]) -> str:
    """The HDBSCAN grid, with the label-free pick marked."""
    chosen = choose_operating_point(rows)
    table_rows = []
    for r in rows:
        s = r.score
        validity = f"{r.relative_validity:.4f}" if r.relative_validity == r.relative_validity else "n/a"
        if chosen is r:
            validity += " *"
        table_rows.append(
            [
                str(r.min_cluster_size),
                str(r.min_samples),
                f"{s.ari:.4f}",
                f"{s.ami:.4f}",
                f"{s.f1_macro:.4f}",
                str(s.n_groups),
                f"{r.noise_fraction:.0%}",
                _fmt_table_ari(s.table_ari),
                validity,
            ]
        )
    columns = [Column(c.header, c.width + (2 if c.header == "validity" else 0), c.align) for c in STABILITY_COLUMNS]
    lines = [render_table(columns, table_rows), ""]
    if chosen is None:
        lines.append("No cell has a defined relative validity — no unsupervised pick is possible.")
    else:
        lines.append(
            f"* chosen by HDBSCAN relative validity (label-free): "
            f"mcs={chosen.min_cluster_size} ms={chosen.min_samples} → "
            f"ARI {chosen.score.ari:.4f}, AMI {chosen.score.ami:.4f}. The best labelled"
        )
        lines.append("  cell is NOT the result; quoting it would be tuning on the test set.")
    return "\n".join(lines)


BLIND_COLUMNS = [
    Column("Configuration", 22, "<"),
    Column("algorithm", 9, "<"),
    Column("chosen", 11, "<"),
    Column("ARI", 8),
    Column("AMI", 8),
    Column("F1-macro", 9),
    Column("k", 4),
    Column("ARI~tbl", 8),
]


def _blind_row(label: str, run: BlindEvaluation) -> list[str]:
    s = run.evaluation.score
    return [
        label,
        run.algorithm,
        run.chosen,
        f"{s.ari:.4f}",
        f"{s.ami:.4f}",
        f"{s.f1_macro:.4f}",
        str(s.n_groups),
        _fmt_table_ari(s.table_ari),
    ]


def format_blind_table(runs: list[tuple[str, BlindEvaluation]]) -> str:
    """Blind runs side by side, with the hyper-parameter each one chose."""
    lines = [
        render_table(BLIND_COLUMNS, [_blind_row(label, run) for label, run in runs]),
        "",
        "Every hyper-parameter above was chosen without the labels: Ward's k by",
        "silhouette, HDBSCAN's cell by relative validity (noise then reassigned by kNN).",
        "F1 is still an upper bound — majority-vote naming reads the labels.",
    ]
    return "\n".join(lines)


def format_robustness(report: RobustnessReport) -> str:
    """Per-wording rows, then mean / min / max — the mean is the number to quote."""
    table_rows = [_blind_row(wording, run) for wording, run in report.rows]
    lines = [render_table(BLIND_COLUMNS, table_rows), ""]
    for attr, label in (("ari", "ARI"), ("ami", "AMI"), ("table_ari", "ARI~tbl")):
        mean, low, high = report.summary(attr)
        lines.append(f"  {label:8s} mean {mean:.4f}   min {low:.4f}   max {high:.4f}")
    lines.append(f"  smallest k across wordings: {report.smallest_k}")
    lines.append("")
    lines.append("The mean is the headline. The default wording was chosen while looking")
    lines.append("at the labelled score, so its row alone would overstate the method.")
    return "\n".join(lines)
