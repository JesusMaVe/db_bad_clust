"""Tests for the semantic anchors. No model download and no DB required."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from db_bad_clust.data.schema_extractor import ColumnMetadata
from db_bad_clust.features.semantic_anchors import (
    CONCEPT_ANCHORS,
    CONCEPT_COMPATIBLE_CATEGORIES,
    TYPE_CATEGORY,
    SemanticAnchors,
    category_of_type,
    concepts_allowed_in,
)


class StubEmbedder:
    """Places each text on a unit axis chosen by a keyword, so cosine is exact.

    The anchor phrases and the column texts are mapped onto the same small set
    of axes; a text sharing an axis with an anchor has cosine 1 against it and 0
    against every other. That makes the expected similarities literals rather
    than something recomputed the way the code computes them.
    """

    AXES: ClassVar[list[str]] = list(CONCEPT_ANCHORS)

    def __init__(self, routing: dict[str, str] | None = None) -> None:
        self.routing = routing or {}
        self.seen: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.seen.extend(texts)
        out = np.zeros((len(texts), len(self.AXES)), dtype=np.float64)
        for i, text in enumerate(texts):
            concept = self._concept_for(text)
            out[i, self.AXES.index(concept)] = 1.0
        return out

    def _concept_for(self, text: str) -> str:
        for needle, concept in self.routing.items():
            if needle in text:
                return concept
        for concept, phrase in CONCEPT_ANCHORS.items():
            if text == phrase:
                return concept
        return self.AXES[0]


def _column(**kwargs) -> ColumnMetadata:
    defaults = {"name": "FECHA_INGRESO", "data_type": "DATE", "nullable": True}
    return ColumnMetadata(**{**defaults, **kwargs})


class TestTypeCategories:
    def test_date_types_are_one_family(self):
        assert category_of_type("DATE") == "date"
        assert category_of_type("TIMESTAMP(6)") == "date"

    def test_every_text_type_lands_in_the_text_family(self):
        for oracle_type in ("VARCHAR2", "CHAR", "CLOB", "NVARCHAR2", "LONG"):
            assert category_of_type(oracle_type) == "text"

    def test_an_unknown_type_has_no_family(self):
        assert category_of_type("XMLTYPE") is None

    def test_every_concept_in_the_table_is_a_real_anchor(self):
        """A typo here would silently disable the mismatch feature."""
        assert set(CONCEPT_COMPATIBLE_CATEGORIES) <= set(CONCEPT_ANCHORS)

    def test_every_category_named_is_one_a_type_maps_to(self):
        named = {c for allowed in CONCEPT_COMPATIBLE_CATEGORIES.values() for c in allowed}
        assert named <= set(TYPE_CATEGORY.values())

    def test_text_can_hold_most_concepts_but_not_a_date(self):
        allowed = concepts_allowed_in("text")
        assert "persona" in allowed
        assert "texto_libre" in allowed
        assert "fecha" not in allowed


class TestSimilarities:
    def test_one_column_per_row_and_one_anchor_per_dimension(self):
        anchors = SemanticAnchors(StubEmbedder())
        sims = anchors.similarities(["cualquier texto", "otro"])
        assert sims.shape == (2, len(CONCEPT_ANCHORS))

    def test_a_text_on_an_anchor_axis_scores_one_there_and_zero_elsewhere(self):
        anchors = SemanticAnchors(StubEmbedder(routing={"nacimiento": "fecha"}))
        sims = anchors.similarities(["fecha de nacimiento"])
        assert sims[0, anchors.concepts.index("fecha")] == pytest.approx(1.0)
        assert sims[0, anchors.concepts.index("cantidad")] == pytest.approx(0.0)

    def test_the_anchors_are_embedded_once_and_reused(self):
        embedder = StubEmbedder()
        anchors = SemanticAnchors(embedder)
        anchors.similarities(["a"])
        anchors.similarities(["b"])
        assert embedder.seen.count(CONCEPT_ANCHORS["fecha"]) == 1

    def test_dominant_concept_names_the_best_matching_anchor(self):
        anchors = SemanticAnchors(StubEmbedder(routing={"precio": "cantidad"}))
        assert anchors.dominant_concepts(["precio unitario"]) == ["cantidad"]


class TestTypeMismatch:
    def test_a_date_named_column_stored_as_date_has_no_mismatch(self):
        anchors = SemanticAnchors(StubEmbedder(routing={"fecha": "fecha"}))
        mismatch = anchors.type_mismatch(
            ["fecha ingreso"], [_column(data_type="DATE")]
        )
        assert mismatch[0, 0] == pytest.approx(0.0)

    def test_a_person_name_stored_as_text_is_not_flagged(self):
        """Compatibility, not identity: "persona" is not "texto_libre", but text
        is exactly where a person's name belongs. The identity formulation
        flagged this and buried the real signal under such false positives.
        """
        anchors = SemanticAnchors(StubEmbedder(routing={"nombre": "persona"}))
        mismatch = anchors.type_mismatch(
            ["nombre cliente"], [_column(data_type="VARCHAR2", data_length=100)]
        )
        assert mismatch[0, 0] == pytest.approx(0.0)

    def test_a_contact_stored_as_a_date_is_flagged(self):
        anchors = SemanticAnchors(StubEmbedder(routing={"email": "contacto"}))
        mismatch = anchors.type_mismatch(["email"], [_column(data_type="DATE")])
        assert mismatch[0, 0] == pytest.approx(1.0)

    def test_a_date_named_column_stored_as_text_is_flagged(self):
        """The wrong_data_types signature: the name says date, the type says text."""
        anchors = SemanticAnchors(StubEmbedder(routing={"fecha": "fecha"}))
        mismatch = anchors.type_mismatch(
            ["fecha ingreso"], [_column(data_type="VARCHAR2", data_length=20)]
        )
        assert mismatch[0, 0] == pytest.approx(1.0)

    def test_an_unmappable_type_yields_no_claim_of_mismatch(self):
        anchors = SemanticAnchors(StubEmbedder(routing={"fecha": "fecha"}))
        mismatch = anchors.type_mismatch(["fecha ingreso"], [_column(data_type="XMLTYPE")])
        assert mismatch[0, 0] == pytest.approx(0.0)

    def test_shape_is_one_column_per_row(self):
        anchors = SemanticAnchors(StubEmbedder())
        mismatch = anchors.type_mismatch(["a", "b", "c"], [_column()] * 3)
        assert mismatch.shape == (3, 1)

    def test_rejects_mismatched_input_lengths(self):
        anchors = SemanticAnchors(StubEmbedder())
        with pytest.raises(ValueError, match="same number"):
            anchors.type_mismatch(["a", "b"], [_column()])


class TestBlock:
    def test_block_is_the_similarities_plus_the_mismatch(self):
        anchors = SemanticAnchors(StubEmbedder(routing={"fecha": "fecha"}))
        block = anchors.build_block(["fecha ingreso"], [_column(data_type="VARCHAR2")])
        assert block.shape == (1, len(CONCEPT_ANCHORS) + 1)
        assert block[0, -1] == pytest.approx(1.0)

    def test_the_block_is_small_enough_to_weigh_against_the_structural_one(self):
        """The point of anchors: a dozen interpretable dimensions, not 384."""
        assert len(CONCEPT_ANCHORS) <= 16
