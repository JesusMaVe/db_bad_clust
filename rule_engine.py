"""
rule_engine.py — Core rule engine for anti-pattern detection

Purpose:
  Heuristic rules that detect common database design anti-patterns
  by analyzing column metadata and schema structure.

Architecture:
  - ColumnRuleEngine: Detects anti-patterns at COLUMN level
    (wrong_data_types, reserved_words, bad_boolean, etc.)
  - TableRuleEngine: Detects anti-patterns at TABLE level
    (giant_table, eav_pattern, inconsistent_naming)

Usage:
  col_engine = ColumnRuleEngine()
  col_results = col_engine.classify(columns, schema)

  table_engine = TableRuleEngine()
  table_results = table_engine.classify(schema)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata
from structural_encoder import ORACLE_TYPE_MAP

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RuleMatch:
    """Result of a single anti-pattern rule check."""

    rule_name: str
    anti_pattern: str
    confidence: float
    severity: Severity
    explanation: str
    fix: str


@dataclass
class ClassificationResult:
    """Classification of a single column or table-level issue."""

    column_name: str
    table_name: str
    predicted_label: str
    confidence: float
    method: str
    explanation: str
    fix: str
    needs_review: bool
    severity: Severity = Severity.MEDIUM


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQL_RESERVED_WORDS = {
    "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "DROP",
    "CREATE", "ALTER", "TABLE", "INDEX", "VIEW", "TRIGGER", "PROCEDURE",
    "FUNCTION", "PACKAGE", "BODY", "BEGIN", "END", "IF", "THEN", "ELSE",
    "LOOP", "FOR", "WHILE", "RETURN", "DECLARE", "EXCEPTION", "CURSOR",
    "OPEN", "CLOSE", "FETCH", "COMMIT", "ROLLBACK", "SAVEPOINT",
    "GRANT", "REVOKE", "EXECUTE", "DESCRIBE", "EXPLAIN", "PLAN",
    "NULL", "NOT", "AND", "OR", "IN", "EXISTS", "BETWEEN", "LIKE",
    "IS", "TRUE", "FALSE", "DEFAULT", "CHECK", "CONSTRAINT",
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "INDEX",
    "ASC", "DESC", "ORDER", "GROUP", "HAVING", "DISTINCT", "UNION",
    "ALL", "INTERSECT", "MINUS", "CONNECT", "BY", "START", "WITH",
    "LEVEL", "ROWNUM", "ROWID", "SYSDATE", "UID", "USER",
    "VARCHAR", "VARCHAR2", "NUMBER", "DATE", "CHAR", "CLOB", "BLOB",
    "INTEGER", "INT", "FLOAT", "TIMESTAMP",
}

# Expanded date keywords (lowercase) - matches anywhere in column name
DATE_KEYWORDS_HIGH = {
    "fecha", "fec", "fch", "fh",
    "date", "time", "timestamp",
    "dia", "mes", "anio", "ano",
    "hor", "hora",
    "created", "updated", "modified",
    "start", "end", "begin",
    "inicio", "fin",
    "alta", "baja", "apertura", "cierre",
    "f_alta", "f_baja", "f_mod", "f_cre",
    "creado", "modificado",
}

# Medium severity: date keywords that can legitimately be VARCHAR2
DATE_KEYWORDS_MEDIUM = set()

DATE_KEYWORDS = DATE_KEYWORDS_HIGH | DATE_KEYWORDS_MEDIUM

# Expanded number keywords (lowercase) - matches anywhere in column name
NUMBER_KEYWORDS_HIGH = {
    "precio", "costo", "salario", "sueldo",
    "importe", "monto", "cantidad",
    "presupuesto",
    "intento", "intentos",
}

# Medium severity: number keywords that can legitimately be VARCHAR2
NUMBER_KEYWORDS_MEDIUM = {
    "id", "fk", "pk",
    "num", "number", "amt", "amount",
    "count", "qty",
    "total", "sum", "suma",
    "price", "cost", "salary",
    "peso", "kg", "lt", "mts",
    "porcentaje", "tasa", "rate",
    "fee", "tarifa", "comision",
    "saldo", "balance", "deuda",
    "credito", "cargo", "abono",
    "valor", "monto_total", "importe_total",
    "presup",
    "nivel", "level",
    "orden", "order",
    "stock", "existencia",
}

NUMBER_KEYWORDS = NUMBER_KEYWORDS_HIGH | NUMBER_KEYWORDS_MEDIUM

# Exceptions: columns where number keywords are legitimate as VARCHAR2
NUMBER_KEYWORDS_EXCEPTIONS = {
    "codigo_postal", "codigo_serie", "numero_serie",
    "numero_documento", "codigo_documento",
}

# Expanded boolean keywords (lowercase) - use word boundary matching
BOOLEAN_KEYWORDS_WORD = {
    "flag", "ind", "active", "enabled",
    "activo", "habilitado",
    "valid", "valido",
    "aprobado", "approved",
    "bloqueado", "blocked",
    "eliminado", "deleted",
    "existente", "existing",
}

# Prefix patterns for boolean (must start with these)
BOOLEAN_PREFIX_PATTERNS = {
    "es_", "has_", "is_",
}

# Keep full set for backward compatibility
BOOLEAN_KEYWORDS = BOOLEAN_KEYWORDS_WORD | BOOLEAN_PREFIX_PATTERNS

POLYMORPHIC_TYPE_NAMES = {
    "TYPE", "TIPO", "DISCRIMINATOR", "KIND", "OBJECTION_TYPE",
    "PAYMENT_TYPE", "DOCUMENT_TYPE", "RECORD_TYPE", "ENTITY_TYPE",
    "TIPO_REGISTRO", "TIPO_DATO", "TIPO_DOCUMENTO",
}

SELF_REF_NAMES = {
    "parent_id", "padre_id", "superior_id", "jefe_id", "manager_id",
    "father_id", "boss_id", "supervisor_id", "referencia_id",
    "cat_padre_id",
}

GIANT_TABLE_THRESHOLD = 15


def _has_boolean_keyword(name_lower: str) -> bool:
    """Check if column name contains a boolean keyword using word boundaries.
    
    Avoids false positives like 'proveedores_id' matching 'es_'.
    """
    # Check word-boundary keywords (exact match in underscore-separated parts)
    words = name_lower.replace("-", "_").split("_")
    for word in words:
        if word in BOOLEAN_KEYWORDS_WORD:
            return True

    # Check prefix patterns
    for prefix in BOOLEAN_PREFIX_PATTERNS:
        if name_lower.startswith(prefix) and len(name_lower) > len(prefix):
            return True

    return False


def _kw_match(name_lower: str, keyword: str) -> bool:
    """Token-boundary keyword match.

    Prevents substring false positives like 'fec' matching inside
    'afectada'. A keyword matches when a token (name split on _ or -)
    equals it or has it as prefix/suffix (camelCase without separators).
    """
    tokens = name_lower.replace("-", "_").split("_")
    return any(
        t == keyword or t.startswith(keyword) or t.endswith(keyword)
        for t in tokens
        if t
    )


def _has_number_keyword(name_lower: str) -> bool:
    """Check if column name contains a number keyword, with exceptions."""
    # Check exceptions first
    if name_lower in NUMBER_KEYWORDS_EXCEPTIONS:
        return False

    # Check high severity (token-boundary match)
    for kw in NUMBER_KEYWORDS_HIGH:
        if _kw_match(name_lower, kw):
            return True

    # Check medium severity (word boundary match)
    words = name_lower.replace("-", "_").split("_")
    for word in words:
        if word in NUMBER_KEYWORDS_MEDIUM:
            return True

    return False


# ---------------------------------------------------------------------------
# Column-Level Rule Engine
# ---------------------------------------------------------------------------


class ColumnRuleEngine:
    """
    Detects anti-patterns at the COLUMN level.

    Rules:
      1. wrong_date_as_text — DATE stored as VARCHAR/CHAR
      2. wrong_number_as_text — NUMBER stored as VARCHAR/CHAR
      3. bad_boolean — BOOLEAN stored as VARCHAR/CHAR
      4. reserved_words — Column names that are SQL reserved words
      5. self_referencing — FK referencing own table
      6. impossible_data — Primary key that allows NULL
      7. polymorphic — Type indicator column detected
      8. wrong_data_types — VARCHAR2 with number/date keywords in table with wrong_data_types flag
    """

    def __init__(self) -> None:
        self._col_id_to_table: dict[int, str] = {}
        self._rules = [
            self._rule_wrong_date_as_text,
            self._rule_wrong_number_as_text,
            self._rule_bad_boolean,
            self._rule_reserved_words,
            self._rule_self_referencing,
            self._rule_impossible_data,
            self._rule_polymorphic,
        ]

    def classify(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None = None
    ) -> list[ClassificationResult]:
        """Classify columns using column-level rules."""
        results: list[ClassificationResult] = []

        # Build a list of (table_name, column) pairs from schema for proper mapping
        col_table_pairs: list[tuple[str, ColumnMetadata]] = []
        if schema:
            for table in schema.tables:
                for col in table.columns:
                    col_table_pairs.append((table.name, col))

        # Create a lookup from column id to table name (handles duplicates)
        col_id_to_table: dict[int, str] = {}
        for table_name, col in col_table_pairs:
            col_id_to_table[id(col)] = table_name

        # Also build a name-based lookup (last wins, for results without identity)
        col_name_to_table: dict[str, str] = {}
        for table_name, col in col_table_pairs:
            col_name_to_table[col.name] = table_name

        # Identity-based lookup: same column name may exist in several
        # tables (e.g. ACTIVO); name lookups would collapse them.
        self._col_id_to_table: dict[int, str] = {}
        if schema:
            for table_name, col in col_table_pairs:
                self._col_id_to_table[id(col)] = table_name

        for rule_fn in self._rules:
            rule_results = rule_fn(columns, schema)
            if schema:
                for r in rule_results:
                    if not r.table_name:
                        match = next(
                            (t.name for t in schema.tables
                             for c in t.columns if c.name == r.column_name),
                            None,
                        )
                        if match:
                            r.table_name = match
            results.extend(rule_results)

        results.sort(key=lambda r: (-r.confidence, r.column_name))
        return results


    def _table_for(self, col: ColumnMetadata) -> str:
        """Return the table name for a column via identity lookup."""
        return self._col_id_to_table.get(id(col), "")

    # ── Rule 1: wrong_date_as_text ─────────────────────────────────────

    def _rule_wrong_date_as_text(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect DATE columns stored as VARCHAR/CHAR."""
        results: list[ClassificationResult] = []
        canonical_types = set(ORACLE_TYPE_MAP.get(t, t) for t in ["VARCHAR", "CHAR"])

        for col in columns:
            col_type = col.data_type.upper().split("(")[0].split(" ")[0]
            mapped_type = ORACLE_TYPE_MAP.get(col_type, col_type)

            if mapped_type in canonical_types:
                name_lower = col.name.lower()
                if any(_kw_match(name_lower, kw) for kw in DATE_KEYWORDS_HIGH):
                    results.append(
                        ClassificationResult(
                            column_name=col.name,
                            table_name=self._table_for(col),
                            predicted_label="date_as_text",
                            confidence=0.9,
                            severity=Severity.HIGH,
                            method="rule_engine",
                            explanation=(
                                f"Column '{col.name}' is {col.data_type} but name "
                                f"contains date keyword."
                            ),
                            fix="Change type to DATE or TIMESTAMP.",
                            needs_review=True,
                        )
                    )

        return results

    # ── Rule 2: wrong_number_as_text ───────────────────────────────────

    def _rule_wrong_number_as_text(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect NUMBER columns stored as VARCHAR/CHAR."""
        results: list[ClassificationResult] = []
        canonical_numeric = set(ORACLE_TYPE_MAP.get(t, t) for t in ["VARCHAR", "CHAR"])

        for col in columns:
            col_type = col.data_type.upper().split("(")[0].split(" ")[0]
            mapped_type = ORACLE_TYPE_MAP.get(col_type, col_type)

            if mapped_type in canonical_numeric:
                name_lower = col.name.lower()
                if _has_number_keyword(name_lower):
                    results.append(
                        ClassificationResult(
                            column_name=col.name,
                            table_name=self._table_for(col),
                            predicted_label="number_as_text",
                            confidence=0.85,
                            severity=Severity.HIGH,
                            method="rule_engine",
                            explanation=(
                                f"Column '{col.name}' is {col.data_type} but name "
                                f"suggests numeric content."
                            ),
                            fix="Change type to NUMBER.",
                            needs_review=True,
                        )
                    )

        return results

    # ── Rule 3: bad_boolean ────────────────────────────────────────────

    def _rule_bad_boolean(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect BOOLEAN stored as VARCHAR/CHAR."""
        results: list[ClassificationResult] = []
        text_types = set(ORACLE_TYPE_MAP.get(t, t) for t in ["VARCHAR", "CHAR"])

        for col in columns:
            col_type = col.data_type.upper().split("(")[0].split(" ")[0]
            mapped_type = ORACLE_TYPE_MAP.get(col_type, col_type)

            if mapped_type in text_types:
                name_lower = col.name.lower()
                if _has_boolean_keyword(name_lower):
                    results.append(
                        ClassificationResult(
                            column_name=col.name,
                            table_name=self._table_for(col),
                            predicted_label="bad_boolean",
                            confidence=0.8,
                            severity=Severity.MEDIUM,
                            method="rule_engine",
                            explanation=(
                                f"Column '{col.name}' is {col.data_type} but name "
                                f"indicates boolean value."
                            ),
                            fix="Use NUMBER(1) with CHECK constraint or BOOLEAN (Oracle 23c+).",
                            needs_review=True,
                        )
                    )

        return results

    # ── Rule 4: reserved_words ─────────────────────────────────────────

    def _rule_reserved_words(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect column names that are SQL reserved words."""
        results: list[ClassificationResult] = []

        for col in columns:
            if col.name.upper() in SQL_RESERVED_WORDS:
                results.append(
                    ClassificationResult(
                        column_name=col.name,
                        table_name=self._table_for(col),
                        predicted_label="reserved_words",
                        confidence=0.95,
                        severity=Severity.HIGH,
                        method="rule_engine",
                        explanation=(
                            f"Column name '{col.name}' is a SQL reserved word."
                        ),
                        fix=f"Rename to '{col.name}_COL' or similar.",
                        needs_review=False,
                    )
                )

        return results

    # ── Rule 5: self_referencing ───────────────────────────────────────

    def _rule_self_referencing(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect FK that references the same table or names suggesting self-ref."""
        results: list[ClassificationResult] = []

        if not schema:
            return results

        for table in schema.tables:
            for col in table.columns:
                # Case 1: Explicit FK to same table
                if col.is_foreign_key and col.fk_references_table:
                    if col.fk_references_table == table.name:
                        results.append(
                            ClassificationResult(
                                column_name=col.name,
                                table_name=table.name,
                                predicted_label="self_referencing",
                                confidence=0.95,
                                severity=Severity.MEDIUM,
                                method="rule_engine",
                                explanation=(
                                    f"FK '{col.name}' references its own table "
                                    f"'{table.name}'."
                                ),
                                fix="Valid for hierarchies, but verify intent.",
                                needs_review=True,
                            )
                        )
                        continue

                # Case 2: Heuristic fallback - name suggests self-reference
                if col.name.lower() in SELF_REF_NAMES:
                    results.append(
                        ClassificationResult(
                            column_name=col.name,
                            table_name=table.name,
                            predicted_label="self_referencing",
                            confidence=0.7,
                            severity=Severity.MEDIUM,
                            method="rule_engine",
                            explanation=(
                                f"Column '{col.name}' name suggests self-referencing FK."
                            ),
                            fix="Verify if this is a self-referencing hierarchy.",
                            needs_review=True,
                        )
                    )

        return results

    # ── Rule 6: impossible_data ─────────────────────────────────────────

    def _rule_impossible_data(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect primary key that allows NULL or FK without reference."""
        results: list[ClassificationResult] = []

        for col in columns:
            # PK + NULL = impossible
            if col.is_primary_key and col.nullable:
                table_name = ""
                if schema:
                    for table in schema.tables:
                        if col in table.columns:
                            table_name = table.name
                            break
                results.append(
                    ClassificationResult(
                        column_name=col.name,
                        table_name=table_name,
                        predicted_label="impossible_data",
                        confidence=0.95,
                        severity=Severity.HIGH,
                        method="rule_engine",
                        explanation=(
                            f"Primary key column '{col.name}' allows NULL values, "
                            f"which is logically impossible."
                        ),
                        fix="Set the primary key column to NOT NULL.",
                        needs_review=False,
                    )
                )

            # FK without reference column
            if col.is_foreign_key and not col.fk_references_column:
                table_name = ""
                if schema:
                    for table in schema.tables:
                        if col in table.columns:
                            table_name = table.name
                            break
                results.append(
                    ClassificationResult(
                        column_name=col.name,
                        table_name=table_name,
                        predicted_label="impossible_data",
                        confidence=0.8,
                        severity=Severity.MEDIUM,
                        method="rule_engine",
                        explanation=(
                            f"Foreign key column '{col.name}' has no reference column defined."
                        ),
                        fix="Define fk_references_column for this foreign key.",
                        needs_review=True,
                    )
                )

        return results

    # ── Rule 7: polymorphic ─────────────────────────────────────────────

    def _rule_polymorphic(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect type indicator columns that suggest polymorphic associations."""
        results: list[ClassificationResult] = []

        if not schema:
            return results

        for table in schema.tables:
            for col in table.columns:
                name_upper = col.name.upper()
                if name_upper in POLYMORPHIC_TYPE_NAMES:
                    results.append(
                        ClassificationResult(
                            column_name=col.name,
                            table_name=table.name,
                            predicted_label="polymorphic",
                            confidence=0.75,
                            severity=Severity.MEDIUM,
                            method="rule_engine",
                            explanation=(
                                f"Column '{col.name}' in table '{table.name}' appears to be a "
                                f"type discriminator, suggesting a polymorphic association."
                            ),
                            fix=(
                                "Use separate tables per type or add a foreign key constraint "
                                "to the referenced table per type."
                            ),
                            needs_review=True,
                        )
                    )

        return results

    # ── Rule 8: implicit foreign key (SchemaSpy signal) ─────────────────

    def detect_implicit_fks(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect implicit FKs: a column named like a reference (e.g.
        PRODUCTO_ID) with no FK constraint declared (SchemaSpy signal).

        Report-level finding: not part of the classify() rules because the
        label has no ground-truth counterpart.

        Only flags when the referenced-looking name matches a real column
        of another table (e.g. PRODUCTOS.ID) — avoids flagging plain IDs.
        """
        results: list[ClassificationResult] = []

        if not schema:
            return results

        # All referenced candidates: {column_name -> [table names]}
        ref_targets: dict[str, list[str]] = {}
        for table in schema.tables:
            for col in table.columns:
                if col.is_primary_key or col.name.upper() == "ID":
                    ref_targets.setdefault(col.name.upper(), []).append(table.name)

        for table in schema.tables:
            for col in table.columns:
                if col.is_foreign_key:
                    continue
                name = col.name.upper()
                # pattern: <TABLE>_ID / <TABLE>_FK or TABLE_BASE.ID
                base = None
                if name.endswith("_ID") and len(name) > 3:
                    base = name[:-3]
                elif name.endswith("_FK") and len(name) > 3:
                    base = name[:-3]
                if not base:
                    continue
                # does the base name correspond to a table with a matching PK/ID?
                target_tables = ref_targets.get("ID", [])
                plausible = any(
                    t.startswith(base) or base.startswith(t.rstrip("S"))
                    for t in target_tables
                ) or base in {t.rstrip("S") for t in ref_targets if t != "ID"}
                if plausible:
                    results.append(
                        ClassificationResult(
                            column_name=col.name,
                            table_name=table.name,
                            predicted_label="implicit_fk",
                            confidence=0.7,
                            severity=Severity.MEDIUM,
                            method="rule_engine",
                            explanation=(
                                f"Column '{col.name}' looks like a reference to "
                                f"'{base}' but no FK constraint is declared."
                            ),
                            fix="Add a FOREIGN KEY constraint to the referenced table.",
                            needs_review=True,
                        )
                    )

        return results

    # ── Rule 9: wrong_data_types ────────────────────────────────────────

    def _rule_wrong_data_types(
        self, columns: list[ColumnMetadata], schema: DatabaseSchema | None
    ) -> list[ClassificationResult]:
        """Detect VARCHAR2 columns with number/date keywords in tables flagged as wrong_data_types."""
        results: list[ClassificationResult] = []
        text_types = set(ORACLE_TYPE_MAP.get(t, t) for t in ["VARCHAR", "CHAR"])

        if not schema:
            return results

        # Build table wrong_data_types flags using the table engine
        table_engine = TableRuleEngine()
        table_results = table_engine.classify(schema=schema)
        table_flags: dict[str, bool] = {}
        for r in table_results:
            if r.predicted_label == "wrong_data_types":
                table_flags[r.table_name] = True

        for table in schema.tables:
            if not table_flags.get(table.name):
                continue
            for col in table.columns:
                col_type = col.data_type.upper().split("(")[0].split(" ")[0]
                mapped_type = ORACLE_TYPE_MAP.get(col_type, col_type)
                if mapped_type not in text_types:
                    continue
                name_lower = col.name.lower()
                is_date = any(_kw_match(name_lower, kw) for kw in DATE_KEYWORDS_HIGH)
                is_number = _has_number_keyword(name_lower)
                if is_date or is_number:
                    results.append(
                        ClassificationResult(
                            column_name=col.name,
                            table_name=table.name,
                            predicted_label="wrong_data_types",
                            confidence=0.85,
                            severity=Severity.HIGH,
                            method="rule_engine",
                            explanation=(
                                f"Column '{col.name}' is {col.data_type} but name "
                                f"suggests {'date' if is_date else 'numeric'} content "
                                f"in table with wrong_data_types pattern."
                            ),
                            fix="Change to proper DATE/NUMBER type.",
                            needs_review=True,
                        )
                    )

        return results


# ---------------------------------------------------------------------------
# Table-Level Rule Engine
# ---------------------------------------------------------------------------


class TableRuleEngine:
    """
    Detects anti-patterns at the TABLE level.

    Rules:
      1. giant_table — Tables with >20 columns
      2. eav_pattern — Entity-Attribute-Value anti-pattern
      3. inconsistent_naming — Mixed naming conventions
    """

    EAV_ENTITY_PATTERNS = [
        r"^\w*ENTITY\w*_?ID$",
        r"^\w*_ID$",
        r"^ID$",
    ]
    EAV_ATTRIBUTE_PATTERNS = [
        r"^\w*ATTRIBUTE\w*$",
        r"^\w*FIELD\w*$",
        r"^\w*PROPERTY\w*$",
        r"^\w*COLUMN\w*$",
        r"^\w*ATTR\w*$",
        r"^\w*KEY\w*$",
        r"^\w*CLAVE\w*$",
    ]
    EAV_VALUE_PATTERNS = [
        r"^\w*VALUE\w*$",
        r"^\w*VALOR\w*$",
        r"^\w*DATA\w*$",
        r"^\w*CONTENT\w*$",
        r"^\w*CONTENIDO\w*$",
    ]

    def __init__(self) -> None:
        self._rules = [
            self._rule_giant_table,
            self._rule_eav_pattern,
            self._rule_inconsistent_naming,
            self._rule_wrong_data_types,
        ]

    def classify(self, schema: DatabaseSchema) -> list[ClassificationResult]:
        """Classify tables using table-level rules.

        Priority: giant_table > eav > inconsistent_naming > wrong_data_types.
        Only the highest-priority detection per table is returned.

        Note: missing_pk and redundant_index (SchemaSpy signals) are NOT
        in the priority chain — they are report-level findings, not
        column labels. Call detect_missing_pk() / detect_redundant_indexes()
        explicitly.
        """
        results: list[ClassificationResult] = []
        flagged_tables: set[str] = set()

        for rule_fn in self._rules:
            rule_results = rule_fn(schema)
            for r in rule_results:
                if r.table_name not in flagged_tables:
                    flagged_tables.add(r.table_name)
                    results.append(r)

        results.sort(key=lambda r: (-r.confidence, r.table_name))
        return results

    # ── Rule 1: giant_table ────────────────────────────────────────────

    def _rule_giant_table(
        self, schema: DatabaseSchema
    ) -> list[ClassificationResult]:
        """Detect tables with too many columns."""
        results: list[ClassificationResult] = []

        for table in schema.tables:
            if len(table.columns) > GIANT_TABLE_THRESHOLD:
                results.append(
                    ClassificationResult(
                        column_name="*",
                        table_name=table.name,
                        predicted_label="giant_table",
                        confidence=min(1.0, len(table.columns) / 50.0),
                        severity=Severity.HIGH if len(table.columns) > 30 else Severity.MEDIUM,
                        method="rule_engine",
                        explanation=(
                            f"Table '{table.name}' has {len(table.columns)} columns "
                            f"(threshold: {GIANT_TABLE_THRESHOLD})."
                        ),
                        fix="Consider splitting into smaller, focused tables.",
                        needs_review=True,
                    )
                )

        return results

    # ── Rule 2: eav_pattern ─────────────────────────────────────────────

    def _rule_eav_pattern(
        self, schema: DatabaseSchema
    ) -> list[ClassificationResult]:
        """Detect Entity-Attribute-Value anti-pattern in tables."""
        results: list[ClassificationResult] = []

        for table in schema.tables:
            col_names = [c.name.upper() for c in table.columns]

            has_entity = any(
                re.match(pat, name) for name in col_names for pat in self.EAV_ENTITY_PATTERNS
            )
            has_attribute = any(
                re.match(pat, name) for name in col_names for pat in self.EAV_ATTRIBUTE_PATTERNS
            )
            has_value = any(
                re.match(pat, name) for name in col_names for pat in self.EAV_VALUE_PATTERNS
            )

            if has_entity and has_attribute and has_value:
                results.append(
                    ClassificationResult(
                        column_name="*",
                        table_name=table.name,
                        predicted_label="eav",
                        confidence=0.85,
                        severity=Severity.HIGH,
                        method="rule_engine",
                        explanation=(
                            f"Table '{table.name}' appears to follow the Entity-Attribute-Value "
                            f"pattern (entity/ID, attribute, value columns detected)."
                        ),
                        fix=(
                            "Normalize to a proper relational design with dedicated columns "
                            "per attribute, or use JSON/XML column types."
                        ),
                        needs_review=True,
                    )
                )

        return results

    # ── Rule 3: inconsistent_naming ────────────────────────────────────

    SPANISH_COLUMN_NAMES = {
        "NOMBRE", "APELLIDO", "DIRECCION", "TELEFONO", "FECHA", "NUMERO",
        "DESCRIPCION", "CORREO", "EDAD", "ESTADO", "CIUDAD", "PAIS",
        "EMPRESA", "SUCURSAL", "REGISTRO", "CODIGO", "CLIENTE", "PROVEEDOR",
        "PRODUCTO", "ORDEN", "VENTA", "COMPRA", "IMPORTE", "PRECIO",
        "CANTIDAD", "STOCK", "SALDO", "MONTO", "SUBTOTAL", "TOTAL",
        "USUARIO", "CONTRASENA", "ACCESO", "ROL", "MODULO", "OPCION",
        "CONSULTA", "REPORTE", "OBSERVACION", "NOTA", "COMENTARIO",
        "DIRECCION_ALTERNATIVA", "DIRECCION_COMPLETA", "PROVINCIA_CIUDAD",
        "CODIGO_POSTAL_PAIS", "OBSERVACIONES", "TELEFONOS", "EMAILS",
        "FAX", "NOMBRE_DEPARTAMENTO", "NOMBRE_PROVEEDOR", "NOMBRE_CLIENTE",
        "PRESUPUESTO", "UBICACION_ALMACEN", "STOCK_SEGURIDAD",
        "TALLAS_DISPONIBLES", "CATEGORIAS", "PRECIOS", "CLIENTE_ID",
        "EMPLEADO_ID", "ORDEN_ID", "PRODUCTO_ID", "USUARIO_ID",
        "FECHA_CREACION", "FECHA_MODIFICACION", "FECHA_ELIMINACION",
        "NUMERO_DOCUMENTO", "NUMERO_TELEFONO", "NUMERO_ORDEN",
        "IMPORTE_TOTAL", "PRECIO_UNITARIO", "CANTIDAD_TOTAL",
        "SALDO_DISPONIBLE", "MONTO_TOTAL", "SUBTOTAL_IMPUESTO",
    }

    ENGLISH_COLUMN_NAMES = {
        "FIRST_NAME", "LAST_NAME", "PHONE_NUMBER", "EMAIL_ADDRESS",
        "DATE_OF_BIRTH", "STATUS_CODE", "CITY_NAME", "COUNTRY_NAME",
        "COMPANY_NAME", "BRANCH_NAME", "RECORD_ID", "CODE_VALUE",
        "CLIENT_NAME", "SUPPLIER_NAME", "PRODUCT_NAME", "ORDER_NUMBER",
        "SALE_AMOUNT", "PURCHASE_AMOUNT", "AMOUNT_TOTAL", "PRICE_VALUE",
        "QUANTITY_TOTAL", "STOCK_LEVEL", "BALANCE_AMOUNT", "TOTAL_SUM",
        "USER_NAME", "PASSWORD_HASH", "ACCESS_TOKEN", "ROLE_NAME",
        "MODULE_NAME", "OPTION_VALUE", "QUERY_TEXT", "REPORT_NAME",
        "COMMENT_TEXT", "CLIENT_ID", "PRODUCT_ID",
        "SUPPLIER_ID", "EMPLOYEE_ID", "BRANCH_ID",
        "CREATED_DATE", "MODIFIED_DATE", "DELETED_DATE",
        "ORDER_ID", "USER_ID", "EMAILS",
    }

    GENERIC_NAMES = {
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "COL1", "COL2", "COL3", "COL4", "COL5", "COL_EXTRA", "TEMP_DATA",
    }

    def _rule_inconsistent_naming(
        self, schema: DatabaseSchema
    ) -> list[ClassificationResult]:
        """Detect mixed Spanish/English naming or generic names within a table."""
        results: list[ClassificationResult] = []

        for table in schema.tables:
            if len(table.columns) < 3:
                continue

            has_spanish = False
            has_english = False
            has_generic = False

            for col in table.columns:
                name = col.name.upper()
                if name in self.SPANISH_COLUMN_NAMES:
                    has_spanish = True
                if name in self.ENGLISH_COLUMN_NAMES:
                    has_english = True
                if name in self.GENERIC_NAMES:
                    has_generic = True

            language_mix = has_spanish and has_english
            generic_only = has_generic and not has_spanish and not has_english
            generic_mix = has_generic and (has_spanish or has_english)

            if language_mix or generic_mix or generic_only:
                if language_mix:
                    label_detail = "mixes Spanish and English names"
                elif generic_only:
                    label_detail = "contains only generic/non-meaningful names"
                else:
                    label_detail = "contains generic/non-meaningful names"
                results.append(
                    ClassificationResult(
                        column_name="*",
                        table_name=table.name,
                        predicted_label="inconsistent_naming",
                        confidence=0.8,
                        severity=Severity.LOW,
                        method="rule_engine",
                        explanation=(
                            f"Table '{table.name}' {label_detail}."
                        ),
                        fix="Standardize column names to a single language and convention.",
                        needs_review=True,
                    )
                )

        return results

    # ── Rule 4: missing primary key (SchemaSpy signal) ──────────────────

    def detect_missing_pk(self, schema: DatabaseSchema) -> list[ClassificationResult]:
        """Detect tables with no primary key (SchemaSpy 'orphan table').

        Report-level finding: not part of the classify() priority chain
        because it would mask more specific table-level detections.
        """
        results: list[ClassificationResult] = []

        for table in schema.tables:
            if not any(col.is_primary_key for col in table.columns):
                results.append(
                    ClassificationResult(
                        column_name="*",
                        table_name=table.name,
                        predicted_label="missing_pk",
                        confidence=0.9,
                        severity=Severity.HIGH,
                        method="rule_engine",
                        explanation=(
                            f"Table '{table.name}' has no primary key."
                        ),
                        fix="Add a PRIMARY KEY constraint on a unique identifier column.",
                        needs_review=False,
                    )
                )

        return results

    # ── Rule 5: redundant index (SchemaSpy signal) ──────────────────────

    def detect_redundant_indexes(self, schema: DatabaseSchema) -> list[ClassificationResult]:
        """Detect composite indexes whose leading column duplicates an
        existing single-column index.

        Report-level finding (see detect_missing_pk).

        Consumes redundant index pairs attached to the schema by
        SchemaExtractor.get_redundant_indexes() (schema.redundant_indexes).
        """
        results: list[ClassificationResult] = []
        pairs = getattr(schema, "redundant_indexes", None) or []

        for table_name, column_name, single_idx, composite_idx in pairs:
            results.append(
                ClassificationResult(
                    column_name="*",
                    table_name=table_name,
                    predicted_label="redundant_index",
                    confidence=0.85,
                    severity=Severity.MEDIUM,
                    method="rule_engine",
                    explanation=(
                        f"Index {composite_idx} on '{table_name}' starts with "
                        f"'{column_name}', already indexed by {single_idx}."
                    ),
                    fix=f"Drop the redundant index {single_idx} or extend it.",
                    needs_review=True,
                )
            )

        return results

    # ── Rule 6: wrong_data_types ────────────────────────────────────────

    WRONG_DATA_TYPES_THRESHOLD = 20

    def _rule_wrong_data_types(
        self, schema: DatabaseSchema
    ) -> list[ClassificationResult]:
        """Detect tables where most columns are VARCHAR2 but names suggest numeric/date."""
        results: list[ClassificationResult] = []
        text_types = set(ORACLE_TYPE_MAP.get(t, t) for t in ["VARCHAR", "CHAR"])

        for table in schema.tables:
            if len(table.columns) < 3:
                continue

            # Count columns by type
            varchar_cols = []
            other_cols = []
            for col in table.columns:
                col_type = col.data_type.upper().split("(")[0].split(" ")[0]
                mapped_type = ORACLE_TYPE_MAP.get(col_type, col_type)
                if mapped_type in text_types:
                    varchar_cols.append(col)
                else:
                    other_cols.append(col)

            # Need at least 1 VARCHAR2 column to be suspicious
            if len(varchar_cols) < 1:
                continue

            # Check if any VARCHAR2 columns have names suggesting wrong type
            suspect_cols = []
            for col in varchar_cols:
                name_lower = col.name.lower()
                is_date = any(_kw_match(name_lower, kw) for kw in DATE_KEYWORDS_HIGH)
                is_number = _has_number_keyword(name_lower)
                is_boolean = _has_boolean_keyword(name_lower)
                if is_date or is_number or is_boolean:
                    suspect_cols.append(col)

            # Flag table if enough suspects (threshold: 20%)
            if len(suspect_cols) >= 1:
                pct_suspect = len(suspect_cols) / len(varchar_cols) * 100
                if pct_suspect < self.WRONG_DATA_TYPES_THRESHOLD:
                    continue

                confidence = min(0.9, 0.5 + pct_suspect / 200)
                severity = Severity.HIGH if pct_suspect > 50 else Severity.MEDIUM

                suspect_names = [c.name for c in suspect_cols[:5]]
                results.append(
                    ClassificationResult(
                        column_name=", ".join(suspect_names),
                        table_name=table.name,
                        predicted_label="wrong_data_types",
                        confidence=confidence,
                        severity=severity,
                        method="rule_engine",
                        explanation=(
                            f"Table '{table.name}' has {len(suspect_cols)} VARCHAR2 columns "
                            f"with names suggesting numeric/date content "
                            f"({len(suspect_cols)}/{len(varchar_cols)} = {pct_suspect:.0f}%)."
                        ),
                        fix="Migrate VARCHAR2 columns to proper DATE/NUMBER types.",
                        needs_review=True,
                    )
                )

        return results


# ---------------------------------------------------------------------------
# Unified Pipeline
# ---------------------------------------------------------------------------

# Tables where ALL columns get the table-level label
_TABLE_LABEL_TABLES = {"giant_table", "eav", "inconsistent_naming"}


def classify(
    columns: list[ColumnMetadata] | None = None,
    schema: DatabaseSchema | None = None,
) -> list[ClassificationResult]:
    """Run column-level + table-level engines and merge results.

    Strategy:
    1. Table-level labels (giant_table, eav, inconsistent_naming) propagate
       to ALL columns in the table.
    2. Column-level detections override for individual columns.
    3. Table-level wrong_data_types + column-level wrong_data_type keywords
       produce wrong_data_types label.
    4. Unmatched columns default to 'clean'.
    """
    if schema is None:
        return []

    col_engine = ColumnRuleEngine()
    table_engine = TableRuleEngine()

    # Run column rules per table: the same column name may exist in several
    # tables (e.g. ACTIVO), so a single flat run collapses detections.
    col_results: list[ClassificationResult] = []
    for table in schema.tables:
        col_results.extend(col_engine.classify(columns=table.columns, schema=schema))
    table_results = table_engine.classify(schema=schema)

    # Build table-level label map: table_name → label
    table_labels: dict[str, str] = {}
    for r in table_results:
        table_labels[r.table_name] = r.predicted_label

    # Build column-level map: "TABLE.COLUMN" → label
    col_labels: dict[str, str] = {}
    for r in col_results:
        col_labels[f"{r.table_name}.{r.column_name}"] = r.predicted_label

    # Add wrong_data_types detections: tables with wrong_data_types flag
    # Override column-level number_as_text/date_as_text with wrong_data_types
    # but preserve bad_boolean and other non-type detections
    # Only for columns where column engine detects number_as_text/date_as_text
    _OVERRIDABLE_LABELS = {"number_as_text", "date_as_text"}
    text_types = set(ORACLE_TYPE_MAP.get(t, t) for t in ["VARCHAR", "CHAR"])
    for table in schema.tables:
        if table_labels.get(table.name) != "wrong_data_types":
            continue
        for col in table.columns:
            key = f"{table.name}.{col.name}"
            existing = col_labels.get(key)
            if existing not in _OVERRIDABLE_LABELS:
                continue  # only override number_as_text/date_as_text
            col_labels[key] = "wrong_data_types"

    # Merge: for each column, decide final label
    results: list[ClassificationResult] = []

    for table in schema.tables:
        tl = table_labels.get(table.name)

        for col in table.columns:
            key = f"{table.name}.{col.name}"
            cl = col_labels.get(key)

            # Case 1: Table-level label applies to ALL columns.
            # Exception: a definitive column-level wrong-type detection
            # (date/number stored as text) is more specific than the
            # table-level naming heuristic, so it wins.
            if tl and tl in _TABLE_LABEL_TABLES:
                if tl == "inconsistent_naming" and cl in (
                    "date_as_text",
                    "number_as_text",
                ):
                    results.append(ClassificationResult(
                        column_name=col.name,
                        table_name=table.name,
                        predicted_label=cl,
                        confidence=0.85,
                        severity=Severity.MEDIUM,
                        method="rule_engine",
                        explanation=f"Column '{col.name}' detected as {cl}.",
                        fix="",
                        needs_review=True,
                    ))
                    continue
                results.append(ClassificationResult(
                    column_name=col.name,
                    table_name=table.name,
                    predicted_label=tl,
                    confidence=0.9,
                    severity=Severity.HIGH,
                    method="rule_engine",
                    explanation=f"Table '{table.name}' detected as {tl}.",
                    fix="",
                    needs_review=False,
                ))
                continue

            # Case 2: Column-level detection
            if cl:
                results.append(ClassificationResult(
                    column_name=col.name,
                    table_name=table.name,
                    predicted_label=cl,
                    confidence=0.85,
                    severity=Severity.MEDIUM,
                    method="rule_engine",
                    explanation=f"Column '{col.name}' detected as {cl}.",
                    fix="",
                    needs_review=True,
                ))
                continue

            # Case 3: default clean
            results.append(ClassificationResult(
                column_name=col.name,
                table_name=table.name,
                predicted_label="clean",
                confidence=0.9,
                severity=Severity.LOW,
                method="rule_engine",
                explanation="No anti-pattern detected.",
                fix="",
                needs_review=False,
            ))

    return results
