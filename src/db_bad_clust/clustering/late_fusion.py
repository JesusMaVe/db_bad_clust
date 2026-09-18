"""
late_fusion.py — Consensus clustering via co-association, an alternative to
weighted-concatenation (early) fusion.

`FeatureBuilder` fuses blocks before clustering — concatenate, then cluster once
on the combined vector ("early"/feature-level fusion, in multi-view clustering's
own taxonomy; see docs/research_improving_clustering.md, Tema 4). This module
instead clusters each view SEPARATELY, then fuses the resulting PARTITIONS into
one consensus partition — no alpha/beta/gamma/delta weight has to be chosen a
priori, because nothing is concatenated.

Late Fusion Multi-view Clustering via Global and Local Alignment Maximization
(arXiv:2208.01198) motivates the general approach (cluster per view, fuse
partitions, not vectors). This is a simpler, off-the-shelf version of that idea
using only what is already a dependency: co-association counting plus
`AgglomerativeClustering(metric="precomputed")` — not the paper's optimisation
objective.

Usage:
    from db_bad_clust.clustering.late_fusion import consensus_clustering

    labels = consensus_clustering([doc_view_labels, structure_view_labels])
"""

from __future__ import annotations

import numpy as np


def co_association_matrix(view_labels: list[np.ndarray]) -> np.ndarray:
    """C[i,j] = fraction of views in which columns i and j share a cluster.

    HDBSCAN noise (-1) never counts as agreement with anything, including
    another noise point — two columns both being "unclustered" in one view
    says nothing about whether they belong together.
    """
    if not view_labels:
        raise ValueError("need at least one view")
    n = len(view_labels[0])
    counts = np.zeros((n, n), dtype=np.float64)
    for labels in view_labels:
        if len(labels) != n:
            raise ValueError("every view must label the same number of columns")
        same = (labels[:, None] == labels[None, :]) & (labels[:, None] != -1)
        counts += same
    return counts / len(view_labels)


def consensus_clustering(
    view_labels: list[np.ndarray],
    distance_threshold: float = 0.5,
    linkage: str = "average",
) -> np.ndarray:
    """Fuse per-view partitions into one consensus partition.

    Builds the co-association matrix, turns it into a distance (1 - C), and
    cuts an agglomerative tree at `distance_threshold` — two columns end up
    together only if they were judged together in enough views. No `n_clusters`
    is assumed: `distance_threshold` (not the true label count) decides where
    the tree gets cut, so this stays genuinely unsupervised.
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(view_labels[0]) == 0:
        return np.array([], dtype=int)

    matrix = co_association_matrix(view_labels)
    distance = 1.0 - matrix
    np.fill_diagonal(distance, 0.0)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage,
    )
    return model.fit_predict(distance).astype(int)
