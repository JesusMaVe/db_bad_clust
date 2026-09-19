"""
table_level.py — The second level: which tables are anomalous, and which together.

A flat clustering of columns cannot serve two kinds of anti-pattern at once.
Type conflicts gather columns from many tables (their ARI against the table is
~0); `eav`, `giant_table` and whole-table `inconsistent_naming` gather the
columns of one table. Pulling both into one partition breaks the conflict
clusters (docs/research_improving_clustering.md). So the design has two levels:

  columns  clustered on the conflict representation (Ward, k by silhouette);
  tables   described by BERT naming features (features.name_signals) and
           split into normal and anomalous with a parameter-free rule.

A column of a normal table keeps its column cluster. A column of an anomalous
table joins its table's anomaly group instead — the anti-pattern is the
table's, so every column in it shares it.

The rule, and why this one. With 21-23 tables, "cluster the tables and call
the biggest cluster normal" failed as soon as the silhouette chose many small
clusters: 11 of 21 tables ended up "anomalous". The rule here has no knob to
tune:

  anomalous  a table whose distance to the coordinate-wise median table is
             above Tukey's fence, Q3 + 1.5 * IQR of those distances;
  grouped    anomalous tables join the same group when they are closer to each
             other than that same fence (single linkage cut at the fence);
  held out   a table with fewer than MIN_TABLE_COLUMNS columns is never judged:
             its sibling signals are zero by construction (no siblings), an
             extreme value that is an artefact, not a finding. It stays normal
             and is left out of the standardisation and the fence. On the full
             corpus no table is that small; in the bootstrap subsamples, 23 of
             the flagged tables had 1-2 columns left before this rule, and the
             two giant tables were grouped together in 31 of 60 subsamples
             instead of 44.

It was the third rule tried, plus the hold-out; the choice is disclosed in the
research doc, and the bootstrap measures sampling noise, not that selection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NORMAL = -1
MIN_TABLE_COLUMNS = 3


def _standardise(features: np.ndarray) -> np.ndarray:
    """Z-score each feature, then scale the block to unit total variance."""
    from db_bad_clust.features.feature_builder import FeatureBuilder

    return FeatureBuilder._unit_variance_block(FeatureBuilder._zscore(features))


def tukey_fence(distances: np.ndarray) -> float:
    """Q3 + 1.5 * IQR — the conventional outlier fence, with nothing to tune."""
    q1, q3 = np.percentile(distances, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


def flag_anomalous_tables(
    features: np.ndarray,
    n_columns: Sequence[int] | None = None,
    min_columns: int = MIN_TABLE_COLUMNS,
) -> np.ndarray:
    """A group id per table: NORMAL (-1) or an anomaly group 0, 1, ...

    Args:
        features: (T, f) one row per table, raw — standardised here.
        n_columns: columns per table; tables below `min_columns` are held out
            (never flagged, not part of the standardisation or the fence).
    """
    features = np.asarray(features, dtype=np.float64)
    groups = np.full(len(features), NORMAL, dtype=int)
    judged = (
        np.asarray(n_columns) >= min_columns
        if n_columns is not None
        else np.ones(len(features), dtype=bool)
    )
    groups[judged] = _flag(features[judged])
    return groups


def _flag(features: np.ndarray) -> np.ndarray:
    """The fence and the grouping, over the tables being judged."""
    from scipy.cluster.hierarchy import fcluster, linkage

    groups = np.full(len(features), NORMAL, dtype=int)
    if len(features) < 4:
        return groups  # no meaningful quartiles
    X = _standardise(features)
    distances = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    fence = tukey_fence(distances)
    anomalous = np.where(distances > fence)[0]
    if len(anomalous) == 1:
        groups[anomalous] = 0
    elif len(anomalous) > 1:
        cut = fcluster(linkage(X[anomalous], method="single"), t=fence, criterion="distance")
        groups[anomalous] = cut - 1
    return groups


def combine_levels(
    column_ids: Sequence[int],
    column_tables: Sequence[str],
    table_order: Sequence[str],
    table_groups: np.ndarray,
) -> np.ndarray:
    """The two-level partition of the columns.

    A column of a normal table keeps its column cluster id; a column of an
    anomalous table takes its table's anomaly group, offset so the two id
    spaces cannot collide.
    """
    column_ids = np.asarray(column_ids, dtype=int)
    offset = int(column_ids.max()) + 1 if len(column_ids) else 0
    group_of = dict(zip(table_order, np.asarray(table_groups, dtype=int), strict=True))
    return np.array(
        [
            cid if group_of[table] == NORMAL else offset + group_of[table]
            for cid, table in zip(column_ids, column_tables, strict=True)
        ],
        dtype=int,
    )
