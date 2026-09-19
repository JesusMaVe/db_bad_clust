"""
name_signals.py — What BERT says about a column's name relative to its table.

The conflict block (semantic_anchors.ConflictBlock) sees anti-patterns that live
in one column: the name promises a date, the type delivers text. Three of the
remaining four classes do not live in one column. In this corpus `eav` is the
whole of CONFIGURACION, `inconsistent_naming` is the whole of TBL_DATOS and most
of ORDENES_COMPRA, `giant_table` is the whole of two tables. They are
properties of how a TABLE names its columns, so they are read here at the
table level and handed to `clustering.table_level`.

Every signal comes from the same encoder as the rest of the branch, reading
the bare column name, with no labels and no hand-written rule:

  per column, computed once at build time (they need the encoder):
    emptiness    1 - best cosine against the six storage-family anchors.
                 High for names that mean nothing: C1, COL_001, SELECT.
    divergence   the largest distance between the words inside one name.
                 High for FECHA_O_DIRECCION.

  per column, computed from the stored name vectors against the siblings
  actually present (so a bootstrap subsample recomputes them honestly):
    atypical     distance to the centroid of the other names in the table.
    clash        best cosine to a sibling spelt differently (FLAG_S_N/FLAG_Y_N).
    heterogeneity  mean pairwise distance among the table's names.

  per table: the mean of each (the max for clash), plus the share of the
  table's columns whose name contradicts their declared type.

Measured on their own, these signals separate little at the COLUMN level
(docs/research_improving_clustering.md, the "señal nueva" section): added to
the column representation they either move nothing or break the conflict
clusters. At the TABLE level, where these anti-patterns actually live, they
flag the right tables. See `clustering.table_level`.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Words that join the parts of a name without carrying meaning of their own.
_JOINERS = frozenset({"de", "del", "la", "el", "los", "las", "o", "y"})

TABLE_NAMING_FEATURES = (
    "emptiness",
    "divergence",
    "atypical",
    "clash",
    "heterogeneity",
    "conflict_rate",
)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ── Per column, at build time ──────────────────────────────────────────


def semantic_emptiness(name_vectors: np.ndarray, anchor_vectors: np.ndarray) -> np.ndarray:
    """1 minus the best cosine of each name against the family anchors. Shape (N,)."""
    sims = _unit_rows(name_vectors) @ _unit_rows(anchor_vectors).T
    return 1.0 - sims.max(axis=1)


def internal_divergence(names: Sequence[str], embedder: Embedder) -> np.ndarray:
    """Largest cosine distance between the content words of each name. Shape (N,).

    Zero for a name with fewer than two content words. Words of one letter and
    joiners ("de", "o", ...) are not content words.
    """
    out = np.zeros(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        words = [w for w in name.split() if len(w) > 1 and w not in _JOINERS]
        if len(words) < 2:
            continue
        vectors = _unit_rows(embedder.encode(words))
        out[i] = 1.0 - min(float(vectors[a] @ vectors[b]) for a, b in combinations(range(len(words)), 2))
    return out


# ── Per column, against the siblings present ───────────────────────────


def sibling_signals(
    name_vectors: np.ndarray,
    names: Sequence[str],
    tables: Sequence[str],
) -> np.ndarray:
    """[atypical, clash, heterogeneity] per column. Shape (N, 3).

    Computed over the columns passed in, so a subsample of a table is treated
    as if it were the whole table. A table with a single column scores zero on
    all three: it has no siblings to be compared with.
    """
    vectors = _unit_rows(name_vectors)
    tables_arr = np.asarray(tables)
    out = np.zeros((len(names), 3), dtype=np.float64)
    for table in dict.fromkeys(tables):
        idx = np.where(tables_arr == table)[0]
        if len(idx) < 2:
            continue
        sims = vectors[idx] @ vectors[idx].T
        out[idx, 2] = 1.0 - float(sims[np.triu_indices(len(idx), 1)].mean())
        for a, i in enumerate(idx):
            others = [b for b in range(len(idx)) if b != a]
            centroid = _unit_rows(vectors[idx[others]].mean(axis=0, keepdims=True))[0]
            out[i, 0] = 1.0 - float(vectors[i] @ centroid)
            spelt_differently = [b for b in others if names[idx[b]] != names[i]]
            if spelt_differently:
                out[i, 1] = max(float(sims[a, b]) for b in spelt_differently)
    return out


# ── Per table ──────────────────────────────────────────────────────────


def table_naming_features(
    intrinsic: np.ndarray,
    name_vectors: np.ndarray,
    names: Sequence[str],
    tables: Sequence[str],
    conflicting: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """One row per table, in first-appearance order. Columns: TABLE_NAMING_FEATURES.

    Args:
        intrinsic: (N, 2) per-column [emptiness, divergence] from build time.
        name_vectors: (N, d) embeddings of the bare column names.
        names: the column names, to tell a different spelling from a repeat.
        tables: the table of each column.
        conflicting: (N,) 1 where the name contradicts the declared type.
    """
    siblings = sibling_signals(name_vectors, names, tables)
    tables_arr = np.asarray(tables)
    conflicting = np.asarray(conflicting, dtype=np.float64)
    order = list(dict.fromkeys(tables))
    rows = []
    for table in order:
        m = tables_arr == table
        rows.append(
            [
                intrinsic[m, 0].mean(),
                intrinsic[m, 1].mean(),
                siblings[m, 0].mean(),
                siblings[m, 1].max(),
                siblings[m, 2].mean(),
                conflicting[m].mean(),
            ]
        )
    return order, np.asarray(rows, dtype=np.float64)
