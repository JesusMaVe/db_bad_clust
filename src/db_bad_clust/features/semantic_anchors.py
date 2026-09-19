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

from typing import TYPE_CHECKING, Literal, Protocol

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
    # Oracle 23ai's native type. Before it, Oracle had no boolean type and
    # flags were stored as NUMBER(1), CHAR(1) or VARCHAR2(1) — see
    # `declared_family`, which reads those as boolean too.
    "BOOLEAN": "boolean",
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


# ═══════════════════════════════════════════════════════════════════════
# The conflict block — what the name expects minus what the type declares
# ═══════════════════════════════════════════════════════════════════════
#
# Why this exists (measured on the 153 columns without the giant tables):
# clustering the raw document embedding gives ARI 0.593 against the *table*
# a column belongs to and 0.111 against its anti-pattern label — the clusters
# are the tables. Take the table out (no table comment, per-table centering,
# column-only sentences) and nothing is left: ARI -0.008 / 0.041 / 0.008. A
# sentence encoder *averages* its facets, and 186 of 243 columns share the
# facet "texto de longitud variable de hasta 255 caracteres, admite nulos", so
# the type phrase dominates and the one thing an anti-pattern IS — a conflict
# between what the name promises and what the type delivers — is diluted.
#
# This block asks the encoder two separate questions and subtracts the answers.
# `expectation` is a zero-shot distribution over six storage families given the
# column NAME alone; `declared_family` is the same six families read off the
# declared type. Their difference is zero for a column whose name and type
# agree, whatever its topic, so every clean column lands near the origin and
# every kind of conflict (date-as-text, amount-as-text, flag-as-CLOB, reference
# without a key) is a displacement in its own direction. That is the geometry
# a density clustering can use: the same representation scores ARI 0.214 with
# an ARI-against-table of 0.015, and its clusters cross ten or more tables.
#
# Hard, not soft (candidato #7). The first version read the name as a softmax
# over the families. Whenever that distribution flattened — a higher
# temperature, a different anchor wording — the declared family dominated the
# block and the silhouette/validity criteria chose k=2: a split by declared
# type, ARI -0.08. On the full corpus it collapsed even with these anchors
# once k could range over 2..30. The default is now the HARD expectation: a
# one-hot on the closest family, plus the margin between the best and the
# second-best cosine as the confidence column. It has no temperature, and none
# of the three wordings in TYPE_FAMILY_ANCHOR_VARIANTS collapses under it.
#
# What is left is the reader itself. The three wordings agree on a column's
# family 58-67% of the time, four encoders x three wordings only 42% on
# average — zero-shot reading of a bare column name is noisy. Voting among
# them does not help: vote shares behave like a soft expectation and bring the
# collapse back. So the wording is part of the method, and results are
# reported as the mean over the three wordings, not the best one.

TYPE_FAMILIES: tuple[str, ...] = ("date", "amount", "reference", "text", "boolean", "binary")

TYPE_FAMILY_ANCHORS: dict[str, str] = {
    "date": "este campo guarda una fecha, un momento o una marca de tiempo",
    "amount": (
        "este campo guarda una cantidad, un importe, un precio, un salario, un stock o un numero"
    ),
    "reference": (
        "este campo guarda el identificador o la clave de otro registro o entidad relacionada"
    ),
    "text": "este campo guarda un nombre, una descripcion, un texto, una direccion o un correo",
    "boolean": "este campo guarda un indicador booleano de si o no, activo o inactivo",
    "binary": "este campo guarda un archivo, una imagen o datos binarios",
}

EXPECTATION_TEMPERATURE = 0.05

# The three wordings measured in candidato #7. "orig" is TYPE_FAMILY_ANCHORS,
# chosen in candidato #6's prototype while looking at the labelled score —
# which is why the headline is the mean over all three, never "orig" alone.
# They exist to measure how much a result depends on the wording. Voting or
# averaging across them was measured and is worse (see the note above).
TYPE_FAMILY_ANCHOR_VARIANTS: dict[str, dict[str, str]] = {
    "orig": TYPE_FAMILY_ANCHORS,
    "W2": {
        family: "este campo guarda " + phrase
        for family, phrase in {
            "date": "la fecha o la hora en que ocurrio algo",
            "amount": "un valor numerico como un monto, un total o un conteo",
            "reference": "una referencia al id de otra tabla",
            "text": "un texto libre como un nombre o una observacion",
            "boolean": "una bandera de verdadero o falso",
            "binary": "un contenido binario adjunto",
        }.items()
    },
    "W3": {
        family: "este campo guarda " + phrase
        for family, phrase in {
            "date": "fecha u hora",
            "amount": "numero, cantidad o importe",
            "reference": "identificador de otro registro",
            "text": "texto libre o nombre",
            "boolean": "si o no",
            "binary": "datos binarios",
        }.items()
    },
}

ExpectationMode = Literal["hard", "soft"]

# Storage family of each declared-type category (see TYPE_CATEGORY). The two
# families with no type of their own — `reference` and `boolean` — are read off
# the constraints and the length instead, in `declared_family`.
_CATEGORY_FAMILY: dict[str, str] = {
    "date": "date",
    "number": "amount",
    "text": "text",
    "binary": "binary",
    "boolean": "boolean",
}

_TEXT_TYPES = frozenset({"CHAR", "NCHAR", "VARCHAR2", "VARCHAR", "NVARCHAR2"})
_REFERENCE = TYPE_FAMILIES.index("reference")
_AMOUNT = TYPE_FAMILIES.index("amount")
_TEXT = TYPE_FAMILIES.index("text")

# Layout of the raw conflict block `ConflictBlock.build` returns, in column
# order: (name, width, internal weight). The weights are applied by
# `FeatureBuilder.build` after scaling each sub-block to unit total variance —
# see `features.feature_builder.scale_conflict_subblocks` for why the block is
# not simply z-scored dimension by dimension.
CONFLICT_SUBBLOCKS: tuple[tuple[str, int, float], ...] = (
    ("diff", len(TYPE_FAMILIES), 1.0),
    ("declared", len(TYPE_FAMILIES), 0.7),
    # Margin (hard mode) or entropy (soft mode): how sure the reader was.
    ("confidence", 1, 0.5),
    # 1 where the name reads as a reference to another record but nothing
    # declares it one — see `ConflictBlock.build`. Weight 0.7, like the other
    # declaration indicators, fixed before it was measured.
    ("unbacked_reference", 1, 0.7),
)
CONFLICT_DIM = sum(width for _, width, _ in CONFLICT_SUBBLOCKS)


def declared_family(column: ColumnMetadata) -> np.ndarray:
    """The storage families a column's declaration puts it in. Shape (6,), multi-hot.

    The type gives one of date/amount/text/binary/boolean (none when
    unrecognised); a foreign key adds `reference`; the ways Oracle stored a
    flag before 23ai add `boolean`: NUMBER(1) and text of one character
    (CHAR(1), VARCHAR2(1), in characters, not bytes). So a NUMBER foreign key
    is amount+reference, an ACTIVO CHAR(1) is text+boolean and an ACTIVO
    NUMBER(1) is amount+boolean. Multi-hot on purpose: the conflict is
    measured against everything the declaration can legitimately mean, and a
    NUMBER(1) can hold a 0-9 code as well as a flag.
    """
    out = np.zeros(len(TYPE_FAMILIES), dtype=np.float64)
    family = _CATEGORY_FAMILY.get(category_of_type(column.data_type) or "")
    if family is not None:
        out[TYPE_FAMILIES.index(family)] = 1.0
    if column.is_foreign_key:
        out[TYPE_FAMILIES.index("reference")] = 1.0
    if _stores_a_flag(column):
        out[TYPE_FAMILIES.index("boolean")] = 1.0
    return out


def _stores_a_flag(column: ColumnMetadata) -> bool:
    """NUMBER(1), or text one character long — how pre-23ai Oracle stored a flag."""
    oracle_type = (column.data_type or "").upper().strip()
    if oracle_type == "NUMBER":
        return column.data_precision == 1 and not column.data_scale
    if oracle_type in _TEXT_TYPES:
        length = getattr(column, "char_length", None) or column.data_length
        return length is not None and 0 < length <= 1
    return False


class ConflictBlock:
    """Name-expectation minus declared-family, per column, with a sentence encoder.

    Reads the column *name* — never the document, which states the type and
    would hand the encoder the answer it is being asked for.
    """

    def __init__(
        self,
        embedder: Embedder,
        anchors: dict[str, str] | None = None,
        temperature: float = EXPECTATION_TEMPERATURE,
        mode: ExpectationMode = "hard",
    ) -> None:
        self.anchors = anchors or TYPE_FAMILY_ANCHORS
        if tuple(self.anchors) != TYPE_FAMILIES:
            raise ValueError(f"anchors must be keyed by {TYPE_FAMILIES}, got {tuple(self.anchors)}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        self.temperature = temperature
        self.mode = mode
        self._embedder = embedder
        self._anchor_vectors: np.ndarray | None = None

    @property
    def anchor_vectors(self) -> np.ndarray:
        """The family prototypes, embedded once."""
        if self._anchor_vectors is None:
            self._anchor_vectors = _l2_normalise(
                np.asarray(self._embedder.encode(list(self.anchors.values())), dtype=np.float64)
            )
        return self._anchor_vectors

    def cosines(self, names: list[str]) -> np.ndarray:
        """Cosine of each NAME against each family prototype. Shape (N, 6)."""
        vectors = _l2_normalise(np.asarray(self._embedder.encode(names), dtype=np.float64))
        return vectors @ self.anchor_vectors.T

    def expectation(self, names: list[str]) -> np.ndarray:
        """The families each NAME suggests. Shape (N, 6), rows sum to 1.

        Hard mode (default): one-hot on the closest family. Ties go to the
        first family in TYPE_FAMILIES, as `argmax` does.

        Soft mode: softmax of the cosines at `temperature`. Kept to reproduce
        candidato #6; it collapses the clustering when it flattens.
        """
        sims = self.cosines(names)
        if self.mode == "hard":
            return np.eye(len(TYPE_FAMILIES))[sims.argmax(axis=1)]
        logits = sims / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)

    def build(self, names: list[str], columns: list[ColumnMetadata]) -> np.ndarray:
        """The raw conflict block. Shape (N, 14):
        [expectation - declared | declared | confidence | unbacked_reference].

        confidence is the margin between the best and second-best cosine in
        hard mode, and the entropy of the expectation in soft mode.

        A name that reads as a reference (CLIENTE_ID) on a column with no
        foreign key is not a type error: a NUMBER or VARCHAR2 is exactly where
        a key belongs. Many real schemas declare no foreign keys at all, and
        counting each such column as a type conflict would bury the real ones.
        So, when the expected family is `reference`:

          - a primary key is the record's own key: no conflict, no finding;
          - no foreign key, declared as number or text: no type conflict, and
            `unbacked_reference` = 1, a finding of its own (on this corpus it
            catches 3 of the 4 impossible_data columns);
          - anything else (a reference stored as a date, say) stays a conflict.

        Raw and corpus-independent — no scaling happens here, so the block is
        portable between databases (health_report) and is normalised at fusion
        time like every other block (`FeatureBuilder.build`).
        """
        if len(names) != len(columns):
            raise ValueError(
                f"names and columns must have the same number of entries, "
                f"got {len(names)} and {len(columns)}"
            )
        sims = self.cosines(names)
        declared = np.array([declared_family(c) for c in columns], dtype=np.float64).reshape(
            len(columns), len(TYPE_FAMILIES)
        )
        if self.mode == "hard":
            expected = np.eye(len(TYPE_FAMILIES))[sims.argmax(axis=1)]
            ranked = np.sort(sims, axis=1)
            confidence = ranked[:, -1:] - ranked[:, -2:-1]
        else:
            logits = sims / self.temperature
            logits -= logits.max(axis=1, keepdims=True)
            expected = np.exp(logits)
            expected /= expected.sum(axis=1, keepdims=True)
            confidence = -(expected * np.log(expected + 1e-12)).sum(axis=1, keepdims=True)
        diff = expected - declared
        unbacked = np.zeros((len(columns), 1), dtype=np.float64)
        reads_as_reference = sims.argmax(axis=1) == _REFERENCE
        for i, column in enumerate(columns):
            if not reads_as_reference[i]:
                continue
            if column.is_primary_key:
                diff[i] = 0.0
            elif not column.is_foreign_key and (
                declared[i, _AMOUNT] or declared[i, _TEXT]
            ):
                diff[i] = 0.0
                unbacked[i, 0] = 1.0
        return np.hstack([diff, declared, confidence, unbacked])
