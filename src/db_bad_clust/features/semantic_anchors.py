"""
semantic_anchors.py — BERT as a detector, not as 384 coordinates.

Concatenating the raw embedding gives the fusion a 384-dimensional block to
weigh against an 18-dimensional structural one, and gives a reader no way to
say *why* two columns ended up together. This module asks the encoder a
smaller, answerable question instead: how close is this column to each of a
dozen named concepts — a date, a quantity, an identifier, a yes/no flag?

The output is a block of one dimension per concept, dimensionally comparable to
the structural block and readable column by column, plus the feature the whole
exercise is for:

    mismatch = how strongly the *name* suggests a concept that the *declared
               type* contradicts

`FECHA_INGRESO VARCHAR2(20)` scores high; `FECHA_CREACION DATE` scores zero.
That is the `wrong_data_types` anti-pattern expressed as a number, obtained
with no labels and no rules — zero-shot classification against prototypes.

Usage:
    from db_bad_clust.features.bert_embedder import BERTEmbedder
    from db_bad_clust.features.semantic_anchors import SemanticAnchors

    anchors = SemanticAnchors(BERTEmbedder())
    block = anchors.build_block(documents, columns)   # (N, n_concepts + 1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from db_bad_clust.data.schema_extractor import ColumnMetadata


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised row vectors."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


# The prototypes, in Spanish because the schema and its comments are. Each is a
# phrase rather than a word: a sentence encoder places "una fecha o un momento
# en el tiempo" far more usefully than the bare token "fecha".
CONCEPT_ANCHORS: dict[str, str] = {
    "fecha": "una fecha o un momento en el tiempo",
    "cantidad": "una cantidad numerica, un importe, un precio o un total",
    "identificador": "un identificador unico o codigo de un registro",
    "booleano": "un indicador de si o no, verdadero o falso, activo o inactivo",
    "texto_libre": "un texto libre, una descripcion o una observacion",
    "ubicacion": "una direccion postal, ciudad, provincia o pais",
    "persona": "el nombre de una persona, cliente, empleado o proveedor",
    "contacto": "un correo electronico, telefono o fax de contacto",
    "estado": "el estado o la situacion en que se encuentra un registro",
    "categoria": "una categoria, un tipo o una clasificacion",
    "binario": "un archivo, una imagen o datos binarios adjuntos",
}

# Which storage family each declared Oracle type belongs to.
TYPE_CATEGORY: dict[str, str] = {
    "DATE": "date",
    "TIMESTAMP": "date",
    "NUMBER": "number",
    "NUMERIC": "number",
    "DECIMAL": "number",
    "FLOAT": "number",
    "BINARY_DOUBLE": "number",
    "BINARY_FLOAT": "number",
    "VARCHAR2": "text",
    "VARCHAR": "text",
    "NVARCHAR2": "text",
    "CHAR": "text",
    "NCHAR": "text",
    "CLOB": "text",
    "NCLOB": "text",
    "LONG": "text",
    "BLOB": "binary",
    "RAW": "binary",
    "BFILE": "binary",
}

# Which storage families each concept can legitimately live in.
#
# This is a compatibility relation, not an identity one, and the difference is
# the whole feature. A first version asked "is the column's concept the same as
# the one its type implies?", which flags NOMBRE_CLIENTE VARCHAR2 — concept
# "persona", type "texto_libre" — even though storing a name as text is
# exactly right. Measured that way, wrong_data_types averaged 0.3355 against
# clean's 0.2863: no separation, and a top-12 that was almost all false
# positives. Most concepts are *supposed* to be text; only a few pairs are an
# anti-pattern, and those are the ones this table encodes.
CONCEPT_COMPATIBLE_CATEGORIES: dict[str, set[str]] = {
    "fecha": {"date"},
    "cantidad": {"number"},
    "identificador": {"number", "text"},
    "booleano": {"text", "number"},
    "texto_libre": {"text"},
    "ubicacion": {"text"},
    "persona": {"text"},
    "contacto": {"text"},
    "estado": {"text", "number"},
    "categoria": {"text", "number"},
    "binario": {"binary"},
}


def category_of_type(oracle_type: str | None) -> str | None:
    """The storage family of a declared type, or None when unrecognised."""
    if not oracle_type:
        return None
    normalized = oracle_type.upper().strip()
    if normalized.startswith("TIMESTAMP"):
        normalized = "TIMESTAMP"
    return TYPE_CATEGORY.get(normalized)


def concepts_allowed_in(category: str) -> list[str]:
    """Every concept that may legitimately be stored in this type family."""
    return [c for c, allowed in CONCEPT_COMPATIBLE_CATEGORIES.items() if category in allowed]


class SemanticAnchors:
    """Score columns against concept prototypes with a sentence encoder."""

    def __init__(self, embedder: Embedder, anchors: dict[str, str] | None = None) -> None:
        self.anchors = anchors or CONCEPT_ANCHORS
        self.concepts = list(self.anchors)
        self._embedder = embedder
        self._anchor_vectors: np.ndarray | None = None

    # ── Similarities ──────────────────────────────────────────────────

    @property
    def anchor_vectors(self) -> np.ndarray:
        """The prototype embeddings, computed once."""
        if self._anchor_vectors is None:
            self._anchor_vectors = _l2_normalise(
                np.asarray(self._embedder.encode(list(self.anchors.values())), dtype=np.float64)
            )
        return self._anchor_vectors

    def similarities(self, texts: list[str]) -> np.ndarray:
        """Cosine similarity of each text against each anchor. Shape (N, K)."""
        vectors = _l2_normalise(np.asarray(self._embedder.encode(texts), dtype=np.float64))
        return vectors @ self.anchor_vectors.T

    def dominant_concepts(self, texts: list[str]) -> list[str]:
        """The closest anchor for each text — the concept the name suggests."""
        return [self.concepts[i] for i in self.similarities(texts).argmax(axis=1)]

    # ── The feature this is all for ───────────────────────────────────

    def type_mismatch(
        self,
        texts: list[str],
        columns: list[ColumnMetadata],
    ) -> np.ndarray:
        """How much better the column reads as a concept its type cannot hold.

        The margin between the closest concept overall and the closest concept
        the declared type family *can* legitimately store. Zero when the best
        concept is already compatible — so a name stored as text scores zero —
        and zero when the type is unrecognised, because the feature never
        invents a mismatch it cannot justify. Shape (N, 1).
        """
        if len(texts) != len(columns):
            raise ValueError(
                f"texts and columns must have the same number of entries, "
                f"got {len(texts)} and {len(columns)}"
            )
        sims = self.similarities(texts)
        out = np.zeros((len(texts), 1), dtype=np.float64)
        for i, column in enumerate(columns):
            category = category_of_type(column.data_type)
            if category is None:
                continue
            allowed = [
                self.concepts.index(c)
                for c in concepts_allowed_in(category)
                if c in self.concepts
            ]
            if not allowed:
                continue
            out[i, 0] = max(0.0, float(sims[i].max()) - float(sims[i][allowed].max()))
        return out

    def build_block(
        self,
        texts: list[str],
        columns: list[ColumnMetadata],
    ) -> np.ndarray:
        """The anchor similarities with the mismatch appended. Shape (N, K+1)."""
        return np.hstack([self.similarities(texts), self.type_mismatch(texts, columns)])


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, so a dot product is a cosine."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
