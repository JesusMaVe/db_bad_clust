"""
schema_extractor.py — Extracción de metadatos desde Oracle Database

Propósito:
  Extraer el esquema completo de una base de datos Oracle: tablas, columnas,
  tipos de dato, restricciones (PK, FK, Unique), índices, y nulabilidad.

  Este módulo es el punto de entrada de la Fase 2 del pipeline
  (ML sobre esquemas de BD). Los datos extraídos alimentan los módulos
  de preprocesamiento textual, embeddings BERT y clustering.

Flujo:
  1. Conectar a Oracle (reusa OracleConnector)
  2. Query a vistas del diccionario de datos:
     - user_tables
     - user_tab_columns
     - user_constraints / user_cons_columns
     - user_ind_columns
  3. Poblar dataclasses anidadas: DatabaseSchema → TableMetadata → ColumnMetadata
  4. Serializar a dict/JSON para consumo por otros módulos

Uso:
  extractor = SchemaExtractor(connection)
  schema = extractor.extract_all()
  for table in schema.tables:
      print(table.name, len(table.columns))
"""

import oracledb
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    nullable: bool
    data_length: Optional[int] = None
    data_precision: Optional[int] = None
    data_scale: Optional[int] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False
    is_indexed: bool = False
    fk_references_table: Optional[str] = None
    fk_references_column: Optional[str] = None
    fk_name: Optional[str] = None
    default_value: Optional[str] = None
    comments: Optional[str] = None


@dataclass
class TableMetadata:
    name: str
    columns: List[ColumnMetadata] = field(default_factory=list)
    table_comment: Optional[str] = None
    row_count_approx: Optional[int] = None


@dataclass
class DatabaseSchema:
    tables: List[TableMetadata] = field(default_factory=list)

    def table_names(self) -> List[str]:
        return [t.name for t in self.tables]

    def total_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    def to_dict(self) -> dict:
        return {"tables": [asdict(t) for t in self.tables]}


# ---------------------------------------------------------------------------
# Schema Extractor
# ---------------------------------------------------------------------------

class SchemaExtractor:
    """
    Extrae metadatos del esquema de una base de datos Oracle.

    Consulta las vistas del diccionario de datos del usuario actual
    para construir un DatabaseSchema completo con información de
    columnas, tipos, restricciones e índices.
    """

    def __init__(self, connection: oracledb.Connection):
        self.connection = connection
        self.cursor = connection.cursor()

    # ── Método principal ──────────────────────────────────────────────

    def extract_all(self) -> DatabaseSchema:
        """
        Extrae el esquema completo de la base de datos.

        Returns:
            DatabaseSchema con todas las tablas, columnas y metadatos.
        """
        table_names = self._get_table_names()
        tables = []

        for tname in table_names:
            columns = self._get_columns(tname)
            pk_cols = self._get_primary_key_columns(tname)
            fk_info = self._get_foreign_keys(tname)
            unique_cols = self._get_unique_columns(tname)
            indexed_cols = self._get_indexed_columns(tname)

            for col in columns:
                col.is_primary_key = col.name in pk_cols
                col.is_foreign_key = col.name in fk_info
                if col.is_foreign_key:
                    ref = fk_info[col.name]
                    col.fk_references_table = ref["ref_table"]
                    col.fk_references_column = ref["ref_column"]
                    col.fk_name = ref["fk_name"]
                col.is_unique = col.name in unique_cols
                col.is_indexed = col.name in indexed_cols

            tables.append(TableMetadata(name=tname, columns=columns))

        logger.info(
            "Esquema extraído: %d tablas, %d columnas",
            len(tables),
            sum(len(t.columns) for t in tables),
        )
        return DatabaseSchema(tables=tables)

    # ── Consultas al diccionario de datos ─────────────────────────────

    def _get_table_names(self) -> List[str]:
        """Obtiene nombres de tablas del usuario actual."""
        self.cursor.execute(
            "SELECT table_name FROM user_tables ORDER BY table_name"
        )
        return [row[0] for row in self.cursor.fetchall()]

    def _get_columns(self, table_name: str) -> List[ColumnMetadata]:
        """
        Obtiene columnas de una tabla con tipo, nulabilidad y longitud.
        """
        self.cursor.execute(
            """
            SELECT
                column_name,
                data_type,
                nullable,
                data_length,
                data_precision,
                data_scale,
                data_default
            FROM user_tab_columns
            WHERE table_name = :table_name
            ORDER BY column_id
            """,
            table_name=table_name,
        )
        columns = []
        for row in self.cursor.fetchall():
            col = ColumnMetadata(
                name=row[0],
                data_type=row[1],
                nullable=(row[2] == "Y"),
                data_length=row[3],
                data_precision=row[4],
                data_scale=row[5],
                default_value=row[6],
            )
            columns.append(col)
        return columns

    def _get_primary_key_columns(self, table_name: str) -> set:
        """
        Obtiene conjunto de nombres de columna que forman la PK.
        """
        self.cursor.execute(
            """
            SELECT cc.column_name
            FROM user_constraints c
            JOIN user_cons_columns cc
              ON c.constraint_name = cc.constraint_name
            WHERE c.table_name = :table_name
              AND c.constraint_type = 'P'
            """,
            table_name=table_name,
        )
        return {row[0] for row in self.cursor.fetchall()}

    def _get_foreign_keys(self, table_name: str) -> dict:
        """
        Obtiene diccionario {col_name -> {ref_table, ref_column, fk_name}}
        para columnas que son llave foránea.
        """
        self.cursor.execute(
            """
            SELECT
                cc.column_name,
                c2.table_name  AS ref_table,
                cc2.column_name AS ref_column,
                c.constraint_name AS fk_name
            FROM user_constraints c
            JOIN user_cons_columns cc
              ON c.constraint_name = cc.constraint_name
            JOIN user_constraints c2
              ON c.r_constraint_name = c2.constraint_name
            JOIN user_cons_columns cc2
              ON c2.constraint_name = cc2.constraint_name
             AND cc2.position = cc.position
            WHERE c.table_name = :table_name
              AND c.constraint_type = 'R'
            """,
            table_name=table_name,
        )
        fk_map = {}
        for row in self.cursor.fetchall():
            fk_map[row[0]] = {
                "ref_table": row[1],
                "ref_column": row[2],
                "fk_name": row[3],
            }
        return fk_map

    def _get_unique_columns(self, table_name: str) -> set:
        """
        Obtiene conjunto de columnas con constraint UNIQUE.
        """
        self.cursor.execute(
            """
            SELECT cc.column_name
            FROM user_constraints c
            JOIN user_cons_columns cc
              ON c.constraint_name = cc.constraint_name
            WHERE c.table_name = :table_name
              AND c.constraint_type = 'U'
            """,
            table_name=table_name,
        )
        return {row[0] for row in self.cursor.fetchall()}

    def _get_indexed_columns(self, table_name: str) -> set:
        """
        Obtiene conjunto de columnas que tienen algún índice.
        """
        self.cursor.execute(
            """
            SELECT column_name
            FROM user_ind_columns
            WHERE table_name = :table_name
            """,
            table_name=table_name,
        )
        return {row[0] for row in self.cursor.fetchall()}

    # ── Utilidad ──────────────────────────────────────────────────────

    def close(self):
        """Cierra el cursor interno."""
        self.cursor.close()
