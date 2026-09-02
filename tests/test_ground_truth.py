"""
test_ground_truth.py — Tests for loading the hand-labelled ground truth.

The rule-engine-derived ground truth is gone (it made accuracy circular); what
is left is the CSV loader and the label vocabulary that guards it.
"""

from __future__ import annotations

import pytest

from db_bad_clust.generation.ground_truth import (
    LABEL_CLEAN,
    MANUAL_LABEL_VOCABULARY,
    load_manual_ground_truth,
)


class TestVocabulary:
    def test_clean_is_part_of_the_vocabulary(self) -> None:
        assert LABEL_CLEAN in MANUAL_LABEL_VOCABULARY

    def test_covers_the_ten_labelled_classes(self) -> None:
        assert len(MANUAL_LABEL_VOCABULARY) == 10


class TestLoadManualGroundTruth:
    """Manual CSV label loading."""

    def test_loads_filled_csv(self, tmp_path) -> None:
        p = tmp_path / "labels.csv"
        p.write_text(
            "table,column,data_type,label\n"
            "EMPLEADOS,SALARIO,VARCHAR2,wrong_data_types\n"
            "PROVEEDORES,ID,NUMBER,clean\n"
        )
        gt = load_manual_ground_truth(p)
        assert gt == {"EMPLEADOS.SALARIO": "wrong_data_types", "PROVEEDORES.ID": "clean"}

    def test_keys_are_upper_cased(self, tmp_path) -> None:
        p = tmp_path / "labels.csv"
        p.write_text("table,column,data_type,label\nempleados,salario,VARCHAR2,clean\n")
        assert "EMPLEADOS.SALARIO" in load_manual_ground_truth(p)

    def test_rejects_empty_label(self, tmp_path) -> None:
        p = tmp_path / "labels.csv"
        p.write_text("table,column,data_type,label\nEMPLEADOS,SALARIO,VARCHAR2,\n")
        with pytest.raises(ValueError, match="Empty label"):
            load_manual_ground_truth(p)

    def test_rejects_invalid_label(self, tmp_path) -> None:
        p = tmp_path / "labels.csv"
        p.write_text("table,column,data_type,label\nEMPLEADOS,SALARIO,VARCHAR2,nope\n")
        with pytest.raises(ValueError, match="Invalid label"):
            load_manual_ground_truth(p)

    def test_rejects_an_empty_file(self, tmp_path) -> None:
        p = tmp_path / "labels.csv"
        p.write_text("table,column,data_type,label\n")
        with pytest.raises(ValueError, match="No rows"):
            load_manual_ground_truth(p)


class TestRealLabels:
    """The versioned ground truth must stay loadable and complete."""

    def test_the_repo_csv_has_243_valid_labels(self) -> None:
        gt = load_manual_ground_truth("output/manual_labels.csv")
        assert len(gt) == 243
        assert set(gt.values()) <= MANUAL_LABEL_VOCABULARY
