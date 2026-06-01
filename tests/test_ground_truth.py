"""
test_ground_truth.py — Tests for the ground_truth module.

Verifies:
  - build_ground_truth_map returns correct number of entries
  - All 23 tables are mapped
  - Primary anti-pattern labels are correct for known tables
  - get_ground_truth returns correct labels for column_names
  - Unknown columns return None
  - get_primary_anti_pattern handles empty/no-flag tables
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from anti_patterns import AntiPatternTable, generate_poorly_designed_tables
from ground_truth import (
    ANTI_PATTERN_PRIORITY,
    LABEL_CLEAN,
    build_ground_truth_map,
    get_ground_truth,
    get_primary_anti_pattern,
)


# ── get_primary_anti_pattern ─────────────────────────────────────────────


class TestGetPrimaryAntiPattern:
    """Anti-pattern priority resolution."""

    def test_wrong_data_types(self) -> None:
        table = AntiPatternTable(name="T", columns={"C": "NUMBER"}, wrong_data_types=True)
        assert get_primary_anti_pattern(table) == "wrong_data_types"

    def test_polymorphic_over_wrong_types(self) -> None:
        table = AntiPatternTable(
            name="T",
            columns={"C": "NUMBER"},
            polymorphic=True,
            wrong_data_types=True,
        )
        assert get_primary_anti_pattern(table) == "polymorphic"

    def test_eav_over_others(self) -> None:
        table = AntiPatternTable(
            name="T",
            columns={"C": "NUMBER"},
            eav_antipattern=True,
            wrong_data_types=True,
            giant_table=True,
        )
        assert get_primary_anti_pattern(table) == "eav"

    def test_reserved_words(self) -> None:
        table = AntiPatternTable(
            name="T",
            columns={"C": "NUMBER"},
            reserved_word_columns=True,
        )
        assert get_primary_anti_pattern(table) == "reserved_words"

    def test_self_referencing(self) -> None:
        table = AntiPatternTable(
            name="T",
            columns={"C": "NUMBER"},
            self_referencing=True,
        )
        assert get_primary_anti_pattern(table) == "self_referencing"

    def test_giant_table(self) -> None:
        table = AntiPatternTable(
            name="T",
            columns={"C": "NUMBER"},
            giant_table=True,
        )
        assert get_primary_anti_pattern(table) == "giant_table"

    def test_inconsistent_naming(self) -> None:
        table = AntiPatternTable(
            name="T",
            columns={"C": "NUMBER"},
            inconsistent_naming=True,
        )
        assert get_primary_anti_pattern(table) == "inconsistent_naming"

    def test_self_contradictory(self) -> None:
        table = AntiPatternTable(
            name="T",
            columns={"C": "NUMBER"},
            self_contradictory=True,
        )
        assert get_primary_anti_pattern(table) == "self_contradictory"

    def test_impossible_data(self) -> None:
        table = AntiPatternTable(
            name="T",
            columns={"C": "NUMBER"},
            impossible_data=True,
        )
        assert get_primary_anti_pattern(table) == "impossible_data"

    def test_clean_no_flags(self) -> None:
        table = AntiPatternTable(name="T", columns={"C": "NUMBER"})
        assert get_primary_anti_pattern(table) == "clean"

    def test_only_no_primary_key_is_clean(self) -> None:
        table = AntiPatternTable(name="T", columns={"C": "NUMBER"}, no_primary_key=True)
        assert get_primary_anti_pattern(table) == "clean"


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
        defined_labels = {label for _, label in ANTI_PATTERN_PRIORITY} | {LABEL_CLEAN}
        assert values.issubset(defined_labels)

    def test_empleados_is_wrong_data_types(self) -> None:
        gt = build_ground_truth_map()
        assert gt["EMPLEADOS.ID"] == "wrong_data_types"
        assert gt["EMPLEADOS.NOMBRE_EMPLEADO"] == "wrong_data_types"

    def test_empleados_historial_is_self_contradictory(self) -> None:
        gt = build_ground_truth_map()
        assert gt["EMPLEADOS_HISTORIAL.ID"] == "self_contradictory"
        assert gt["EMPLEADOS_HISTORIAL.SALARIO_ANTERIOR"] == "self_contradictory"

    def test_todo_en_uno_is_polymorphic(self) -> None:
        gt = build_ground_truth_map()
        assert gt["TODO_EN_UNO.ID"] == "polymorphic"
        assert gt["TODO_EN_UNO.TIPO_REGISTRO"] == "polymorphic"

    def test_configuracion_is_eav(self) -> None:
        gt = build_ground_truth_map()
        assert gt["CONFIGURACION.ID"] == "eav"
        assert gt["CONFIGURACION.CLAVE"] == "eav"
        assert gt["CONFIGURACION.VALOR"] == "eav"

    def test_reportes_is_reserved_words(self) -> None:
        gt = build_ground_truth_map()
        assert gt["REPORTES.NULL"] == "reserved_words"
        assert gt["REPORTES.SELECT"] == "reserved_words"

    def test_categorias_is_self_referencing(self) -> None:
        gt = build_ground_truth_map()
        assert gt["CATEGORIAS.ID"] == "self_referencing"
        assert gt["CATEGORIAS.CAT_PADRE_ID"] == "self_referencing"

    def test_clean_tables(self) -> None:
        gt = build_ground_truth_map()
        assert gt["PROVEEDORES.ID"] == "clean"
        assert gt["PRODUCTOS.SKU"] == "clean"
        assert gt["AUDITORIA_LOG.ID"] == "clean"
        assert gt["INVENTARIO.PRODUCTO_ID"] == "clean"
        assert gt["TABLA_VACIA.ID"] == "clean"

    def test_departamentos_is_wrong_data_types(self) -> None:
        gt = build_ground_truth_map()
        assert gt["DEPARTAMENTOS.ID"] == "wrong_data_types"
        assert gt["DEPARTAMENTOS.NOMBRE_DEPARTAMENTO"] == "wrong_data_types"

    def test_transacciones_is_impossible_data(self) -> None:
        gt = build_ground_truth_map()
        assert gt["TRANSACCIONES.ID"] == "impossible_data"
        assert gt["TRANSACCIONES.MONTO"] == "impossible_data"

    def test_ordenes_compra_is_wrong_data_types(self) -> None:
        gt = build_ground_truth_map()
        assert gt["ORDENES_COMPRA.ORDER_ID"] == "wrong_data_types"

    def test_clientes_direcciones_is_inconsistent_naming(self) -> None:
        gt = build_ground_truth_map()
        assert gt["CLIENTES_DIRECCIONES.ID_CLIENTE"] == "inconsistent_naming"

    def test_tbl_datos_is_inconsistent_naming(self) -> None:
        gt = build_ground_truth_map()
        assert gt["TBL_DATOS.C1"] == "inconsistent_naming"

    def test_tabla_base_datos_is_giant_table(self) -> None:
        gt = build_ground_truth_map()
        assert gt["TABLA_BASE_DATOS.A"] == "giant_table"

    def test_backup_datos_is_giant_table(self) -> None:
        gt = build_ground_truth_map()
        assert gt["BACKUP_DATOS.COL_001"] == "giant_table"


# ── get_ground_truth ─────────────────────────────────────────────────────


class TestGetGroundTruth:
    """Label lookup for pipeline columns."""

    def test_with_column_names(self) -> None:
        gt_map = build_ground_truth_map()
        column_names = ["EMPLEADOS.ID", "EMPLEADOS.NOMBRE_EMPLEADO", "PROVEEDORES.ID"]
        result = get_ground_truth([], column_names=column_names, gt_map=gt_map)
        assert result == ["wrong_data_types", "wrong_data_types", "clean"]

    def test_without_column_names_returns_unknown(self) -> None:
        gt_map = build_ground_truth_map()
        result = get_ground_truth(["EMPLEADOS", "PROVEEDORES"], gt_map=gt_map)
        assert result == [None, None]

    def test_unknown_column_returns_none(self) -> None:
        gt_map = build_ground_truth_map()
        result = get_ground_truth([], column_names=["NONEXISTENT.GARBAGE"], gt_map=gt_map)
        assert result == [None]

    def test_no_gt_map_rebuilds(self) -> None:
        column_names = ["EMPLEADOS.ID", "TABLA_BASE_DATOS.a"]
        result = get_ground_truth([], column_names=column_names)
        assert result[0] == "wrong_data_types"
        assert result[1] == "giant_table"
