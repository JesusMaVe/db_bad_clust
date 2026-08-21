"""
test_ground_truth.py — Tests for the ground_truth module.

Verifies:
  - build_ground_truth_map returns correct number of entries
  - All tables are mapped
  - Column-level anti-pattern labels are correct
  - get_ground_truth returns correct labels for column_names
  - Unknown columns return None
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anti_patterns import generate_poorly_designed_tables
from ground_truth import (
    LABEL_CLEAN,
    build_ground_truth_map,
    get_ground_truth,
    load_manual_ground_truth,
)

# ── build_ground_truth_map ────────────────────────────────────────────────


class TestBuildGroundTruthMap:
    """Ground truth map construction."""

    def test_returns_dict(self) -> None:
        gt = build_ground_truth_map()
        assert isinstance(gt, dict)

    def test_all_tables_mapped(self) -> None:
        gt = build_ground_truth_map()
        tables = generate_poorly_designed_tables()
        total_cols = sum(len(t.columns) for t in tables)
        assert len(gt) == total_cols

    def test_no_unknown_labels(self) -> None:
        gt = build_ground_truth_map()
        values = set(gt.values())
        assert LABEL_CLEAN in values
        valid_labels = {
            "wrong_data_types", "reserved_words", "self_contradictory",
            "impossible_data", "self_referencing", "polymorphic",
            "giant_table", "inconsistent_naming", "eav", LABEL_CLEAN,
            "bad_boolean", "number_as_text",
            # raw rule label surviving on non-flagged tables with CLOB
            # keyword columns (e.g. METADATA.FECHA_CREACION) — issue CLOB fix
            "date_as_text",
        }
        assert values.issubset(valid_labels)

    def test_empleados_has_correct_types(self) -> None:
        """Column-level: only columns with wrong types are flagged."""
        gt = build_ground_truth_map()
        # FECHA_NACIMIENTO is VARCHAR2 but name suggests DATE -> wrong_data_types
        assert gt["EMPLEADOS.FECHA_NACIMIENTO"] == "wrong_data_types"
        # SALARIO is VARCHAR2 but name suggests NUMBER -> wrong_data_types
        assert gt["EMPLEADOS.SALARIO"] == "wrong_data_types"
        # NOMBRE_EMPLEADO is VARCHAR2 and name suggests TEXT -> clean
        assert gt["EMPLEADOS.NOMBRE_EMPLEADO"] == "clean"

    def test_reserved_words_detected(self) -> None:
        """Reserved word columns are detected."""
        gt = build_ground_truth_map()
        assert gt["REPORTES.NULL"] == "reserved_words"
        assert gt["REPORTES.SELECT"] == "reserved_words"

    def test_polymorphic_detected(self) -> None:
        """Polymorphic columns are detected."""
        gt = build_ground_truth_map()
        assert gt["TRANSACCIONES.TIPO"] == "polymorphic"
        assert gt["METADATA.TIPO_DATO"] == "polymorphic"

    def test_eav_columns(self) -> None:
        """EAV table columns are flagged as eav."""
        gt = build_ground_truth_map()
        assert gt["CONFIGURACION.CLAVE"] == "eav"
        assert gt["CONFIGURACION.VALOR"] == "eav"

    def test_self_referencing(self) -> None:
        """Self-referencing columns are detected."""
        gt = build_ground_truth_map()
        assert gt["CATEGORIAS.CAT_PADRE_ID"] == "self_referencing"

    def test_giant_table_columns(self) -> None:
        """Giant table columns are flagged."""
        gt = build_ground_truth_map()
        assert gt["TABLA_BASE_DATOS.A"] == "giant_table"
        assert gt["BACKUP_DATOS.COL_001"] == "giant_table"

    def test_inconsistent_naming(self) -> None:
        """Inconsistent naming columns are flagged."""
        gt = build_ground_truth_map()
        assert gt["CLIENTES_DIRECCIONES.ID_CLIENTE"] == "inconsistent_naming"
        assert gt["TBL_DATOS.C1"] == "inconsistent_naming"

    def test_clean_columns(self) -> None:
        """Clean columns are labeled clean."""
        gt = build_ground_truth_map()
        assert gt["PROVEEDORES.ID"] == "clean"
        assert gt["PRODUCTOS.SKU"] == "clean"

    def test_impossible_data(self) -> None:
        """Impossible data columns are detected (FK without reference)."""
        gt = build_ground_truth_map()
        assert gt["AUDITORIA_LOG.REGISTRO_ID"] == "impossible_data"
        assert gt["EMPLEADOS.DEPARTAMENTO"] == "impossible_data"


# ── load_manual_ground_truth ─────────────────────────────────────────────


class TestLoadManualGroundTruth:
    """Manual CSV label loading."""

    def test_loads_filled_csv(self, tmp_path) -> None:
        p = tmp_path / "labels.csv"
        p.write_text("table,column,data_type,label\nEMPLEADOS,SALARIO,VARCHAR2,wrong_data_types\nPROVEEDORES,ID,NUMBER,clean\n")
        gt = load_manual_ground_truth(p)
        assert gt == {"EMPLEADOS.SALARIO": "wrong_data_types", "PROVEEDORES.ID": "clean"}

    def test_rejects_empty_label(self, tmp_path) -> None:
        p = tmp_path / "labels.csv"
        p.write_text("table,column,data_type,label\nEMPLEADOS,SALARIO,VARCHAR2,\n")
        try:
            load_manual_ground_truth(p)
        except ValueError as e:
            assert "Empty label" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def test_rejects_invalid_label(self, tmp_path) -> None:
        p = tmp_path / "labels.csv"
        p.write_text("table,column,data_type,label\nEMPLEADOS,SALARIO,VARCHAR2,nope\n")
        try:
            load_manual_ground_truth(p)
        except ValueError as e:
            assert "Invalid label" in str(e)
        else:
            raise AssertionError("expected ValueError")


# ── get_ground_truth ─────────────────────────────────────────────────────


class TestGetGroundTruth:
    """Label lookup for pipeline columns."""

    def test_with_column_names(self) -> None:
        gt_map = build_ground_truth_map()
        column_names = ["EMPLEADOS.FECHA_NACIMIENTO", "EMPLEADOS.NOMBRE_EMPLEADO", "PROVEEDORES.ID"]
        result = get_ground_truth([], column_names=column_names, gt_map=gt_map)
        assert result == ["wrong_data_types", "clean", "clean"]

    def test_without_column_names_returns_unknown(self) -> None:
        gt_map = build_ground_truth_map()
        result = get_ground_truth(["EMPLEADOS", "PROVEEDORES"], gt_map=gt_map)
        assert result == [None, None]

    def test_unknown_column_returns_none(self) -> None:
        gt_map = build_ground_truth_map()
        result = get_ground_truth([], column_names=["NONEXISTENT.GARBAGE"], gt_map=gt_map)
        assert result == [None]

    def test_no_gt_map_rebuilds(self) -> None:
        column_names = ["EMPLEADOS.FECHA_NACIMIENTO", "TABLA_BASE_DATOS.a"]
        result = get_ground_truth([], column_names=column_names)
        assert result[0] == "wrong_data_types"
        assert result[1] == "giant_table"
