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


# ═══════════════════════════════════════════════════════════════════════
# The conflict block
# ═══════════════════════════════════════════════════════════════════════

from db_bad_clust.features.semantic_anchors import (  # noqa: E402
    CONFLICT_DIM,
    CONFLICT_SUBBLOCKS,
    TYPE_FAMILIES,
    TYPE_FAMILY_ANCHORS,
    ConflictBlock,
    declared_family,
)


class FamilyStubEmbedder:
    """Each family anchor sits on its own unit axis; a name is routed by keyword.

    A name with no matching keyword gets an equal share of every axis, so its
    expectation comes out flat — the shape a meaningless name (C3) should have.
    """

    def __init__(self, routing: dict[str, str] | None = None) -> None:
        self.routing = routing or {}
        self.seen: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.seen.extend(texts)
        out = np.zeros((len(texts), len(TYPE_FAMILIES)), dtype=np.float64)
        for i, text in enumerate(texts):
            family = self._family_for(text)
            if family is None:
                out[i, :] = 1.0 / np.sqrt(len(TYPE_FAMILIES))
            else:
                out[i, TYPE_FAMILIES.index(family)] = 1.0
        return out

    def _family_for(self, text: str) -> str | None:
        for family, phrase in TYPE_FAMILY_ANCHORS.items():
            if text == phrase:
                return family
        for needle, family in self.routing.items():
            if needle in text:
                return family
        return None


def _idx(family: str) -> int:
    return TYPE_FAMILIES.index(family)


class TestDeclaredFamily:
    def test_a_date_is_the_date_family_only(self):
        d = declared_family(_column(data_type="DATE"))
        assert d[_idx("date")] == 1.0
        assert d.sum() == 1.0

    def test_a_number_foreign_key_is_amount_and_reference(self):
        d = declared_family(_column(data_type="NUMBER", is_foreign_key=True))
        assert d[_idx("amount")] == 1.0
        assert d[_idx("reference")] == 1.0
        assert d.sum() == 2.0

    def test_a_char_of_length_one_is_text_and_boolean(self):
        d = declared_family(_column(name="ACTIVO", data_type="CHAR", data_length=1))
        assert d[_idx("text")] == 1.0
        assert d[_idx("boolean")] == 1.0

    def test_a_wide_char_is_text_only(self):
        d = declared_family(_column(data_type="CHAR", data_length=10))
        assert d[_idx("boolean")] == 0.0
        assert d[_idx("text")] == 1.0

    def test_an_unknown_type_declares_nothing(self):
        assert declared_family(_column(data_type="XMLTYPE")).sum() == 0.0

    def test_layout_constants_agree(self):
        assert CONFLICT_DIM == 2 * len(TYPE_FAMILIES) + 1
        assert [name for name, _, _ in CONFLICT_SUBBLOCKS] == ["diff", "declared", "confidence"]
        assert tuple(TYPE_FAMILY_ANCHORS) == TYPE_FAMILIES


class TestExpectation:
    def test_a_routed_name_is_almost_all_its_family(self):
        block = ConflictBlock(FamilyStubEmbedder(routing={"fecha": "date"}))
        p = block.expectation(["fecha nacimiento"])
        assert p.shape == (1, len(TYPE_FAMILIES))
        assert p[0, _idx("date")] > 0.99
        assert p.sum() == pytest.approx(1.0)

    def test_a_meaningless_name_is_flat_in_soft_mode(self):
        block = ConflictBlock(FamilyStubEmbedder(), mode="soft")
        p = block.expectation(["c3"])
        assert np.allclose(p, 1.0 / len(TYPE_FAMILIES))

    def test_hard_mode_is_one_hot_even_for_a_meaningless_name(self):
        """No temperature, no flattening: the property that stops the collapse."""
        block = ConflictBlock(FamilyStubEmbedder())
        p = block.expectation(["c3", "fecha"])
        assert set(np.unique(p)) <= {0.0, 1.0}
        assert np.array_equal(p.sum(axis=1), [1.0, 1.0])

    def test_hard_is_the_default(self):
        assert ConflictBlock(FamilyStubEmbedder()).mode == "hard"

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError):
            ConflictBlock(FamilyStubEmbedder(), mode="medium")

    def test_the_anchors_are_embedded_once(self):
        embedder = FamilyStubEmbedder()
        block = ConflictBlock(embedder)
        block.expectation(["a"])
        block.expectation(["b"])
        assert embedder.seen.count(TYPE_FAMILY_ANCHORS["date"]) == 1

    def test_rejects_a_non_positive_temperature(self):
        with pytest.raises(ValueError):
            ConflictBlock(FamilyStubEmbedder(), temperature=0.0)

    def test_rejects_anchors_not_keyed_by_the_families(self):
        with pytest.raises(ValueError):
            ConflictBlock(FamilyStubEmbedder(), anchors={"fecha": "x"})


class TestConflictBuild:
    def test_shape_is_n_by_thirteen(self):
        block = ConflictBlock(FamilyStubEmbedder(routing={"fecha": "date"}))
        out = block.build(["fecha alta", "fecha baja"], [_column(), _column()])
        assert out.shape == (2, CONFLICT_DIM)

    def test_a_date_name_declared_date_has_zero_diff(self):
        block = ConflictBlock(FamilyStubEmbedder(routing={"fecha": "date"}))
        out = block.build(["fecha ingreso"], [_column(data_type="DATE")])
        diff = out[0, : len(TYPE_FAMILIES)]
        assert np.allclose(diff, 0.0, atol=1e-6)

    def test_a_date_name_declared_text_conflicts_in_two_directions(self):
        """The wrong_data_types signature: +1 where the name expects, -1 where the type is."""
        block = ConflictBlock(FamilyStubEmbedder(routing={"fecha": "date"}))
        out = block.build(["fecha ingreso"], [_column(data_type="VARCHAR2", data_length=20)])
        diff = out[0, : len(TYPE_FAMILIES)]
        assert diff[_idx("date")] == pytest.approx(1.0, abs=1e-6)
        assert diff[_idx("text")] == pytest.approx(-1.0, abs=1e-6)
        assert np.allclose(np.delete(diff, [_idx("date"), _idx("text")]), 0.0, atol=1e-6)

    def test_declared_block_is_copied_through(self):
        block = ConflictBlock(FamilyStubEmbedder())
        col = _column(data_type="NUMBER", is_foreign_key=True)
        out = block.build(["x"], [col])
        declared = out[0, len(TYPE_FAMILIES) : 2 * len(TYPE_FAMILIES)]
        assert np.array_equal(declared, declared_family(col))

    def test_soft_entropy_is_zero_for_a_sure_name_and_maximal_for_a_flat_one(self):
        block = ConflictBlock(FamilyStubEmbedder(routing={"fecha": "date"}), mode="soft")
        out = block.build(["fecha", "c3"], [_column(), _column()])
        sure, flat = out[0, -1], out[1, -1]
        assert sure == pytest.approx(0.0, abs=1e-6)
        assert flat == pytest.approx(np.log(len(TYPE_FAMILIES)), abs=1e-6)

    def test_rejects_mismatched_lengths(self):
        block = ConflictBlock(FamilyStubEmbedder())
        with pytest.raises(ValueError):
            block.build(["a", "b"], [_column()])

    def test_reads_the_name_it_is_given_not_a_document(self):
        embedder = FamilyStubEmbedder()
        ConflictBlock(embedder).build(["fecha ingreso"], [_column()])
        assert "fecha ingreso" in embedder.seen
        assert not any("tipo" in t for t in embedder.seen if t not in TYPE_FAMILY_ANCHORS.values())


class TestHardConfidence:
    def test_margin_is_one_for_a_name_on_an_anchor_axis(self):
        """Cosine 1 against its family and 0 against the rest: margin 1."""
        block = ConflictBlock(FamilyStubEmbedder(routing={"fecha": "date"}))
        out = block.build(["fecha alta"], [_column()])
        assert out[0, -1] == pytest.approx(1.0)

    def test_margin_is_zero_for_a_name_equidistant_from_every_anchor(self):
        block = ConflictBlock(FamilyStubEmbedder())
        out = block.build(["c3"], [_column()])
        assert out[0, -1] == pytest.approx(0.0)

    def test_hard_diff_for_a_date_stored_as_text(self):
        block = ConflictBlock(FamilyStubEmbedder(routing={"fecha": "date"}))
        out = block.build(["fecha ingreso"], [_column(data_type="VARCHAR2", data_length=20)])
        assert np.array_equal(out[0, : len(TYPE_FAMILIES)], [1.0, 0.0, 0.0, -1.0, 0.0, 0.0])


class TestAnchorVariants:
    def test_every_variant_is_keyed_by_the_families_in_order(self):
        from db_bad_clust.features.semantic_anchors import TYPE_FAMILY_ANCHOR_VARIANTS

        assert list(TYPE_FAMILY_ANCHOR_VARIANTS) == ["orig", "W2", "W3"]
        for anchors in TYPE_FAMILY_ANCHOR_VARIANTS.values():
            assert tuple(anchors) == TYPE_FAMILIES

    def test_orig_is_the_default_wording(self):
        from db_bad_clust.features.semantic_anchors import TYPE_FAMILY_ANCHOR_VARIANTS

        assert TYPE_FAMILY_ANCHOR_VARIANTS["orig"] is TYPE_FAMILY_ANCHORS

    def test_every_variant_builds_a_block(self):
        from db_bad_clust.features.semantic_anchors import TYPE_FAMILY_ANCHOR_VARIANTS

        for anchors in TYPE_FAMILY_ANCHOR_VARIANTS.values():
            out = ConflictBlock(FamilyStubEmbedder(), anchors=anchors).build(["x"], [_column()])
            assert out.shape == (1, CONFLICT_DIM)
