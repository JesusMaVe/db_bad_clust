"""
text_preprocessor.py — Textual preprocessing of column names

Purpose:
  Transform technical column names (camelCase, snake_case, UPPER_CASE)
  into clean, contextualized text for BERT input.

  Pipeline:
    1. Split camelCase / snake_case / UPPER_CASE
    2. Expand common abbreviations (ID → identifier, FK → foreign key, etc.)
    3. Remove redundant prefixes (tbl_, col_, fld_)
    4. Lowercase only ALL-CAPS tokens (Oracle convention); keep mixed-case
       tokens as-is because the embedding model is case-sensitive
    5. Contextualize with table name → "customers: customer name"
    6. Append the column comment verbatim (natural language, if any)

Usage:
  preprocessor = TextPreprocessor()
  text = preprocessor.process("FECHA_NACIMIENTO", table_name="EMPLEADOS")
  # → "empleados: fecha nacimiento"
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db_bad_clust.data.schema_extractor import ColumnMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abbreviation dictionary
# ---------------------------------------------------------------------------

ABBREVIATIONS = {
    # General
    "id": "identifier",
    "pk": "primary key",
    "fk": "foreign key",
    "uq": "unique",
    "idx": "index",
    "num": "number",
    "no": "number",
    "qty": "quantity",
    "amt": "amount",
    "desc": "description",
    "descr": "description",
    "info": "information",
    "config": "configuration",
    "temp": "temporary",
    "ref": "reference",
    "orig": "original",
    "src": "source",
    "dest": "destination",
    "dst": "destination",
    "prev": "previous",
    "next": "next",
    "cur": "current",
    "curr": "current",
    "avg": "average",
    "min": "minimum",
    "minute": "minute",
    "max": "maximum",
    "cnt": "count",
    "total": "total",
    "sum": "total",
    "calc": "calculation",
    "del": "delete",
    "rem": "remove",
    "addr": "address",
    "dept": "department",
    "org": "organization",
    "emp": "employee",
    "empl": "employee",
    "prod": "product",
    "cat": "category",
    "subcat": "subcategory",
    "tbl": "table",
    "col": "column",
    "fld": "field",
    "msg": "message",
    "txt": "text",
    "img": "image",
    "pic": "picture",
    "doc": "document",
    "dir": "directory",
    "path": "route",
    "url": "web address",
    "uri": "resource identifier",
    "uuid": "unique identifier",
    "guid": "unique identifier",
    "seq": "sequence",
    "perm": "permission",
    "priv": "privilege",
    "sess": "session",
    "attr": "attribute",
    "prop": "property",
    "param": "parameter",
    "args": "arguments",
    "expr": "expression",
    "stmt": "statement",
    "regex": "regular expression",
    "dt": "date",
    "hr": "hour",
    "hrs": "hours",
    "sec": "second",
    "ms": "milliseconds",
    "ts": "timestamp",
}


# Spanish overrides, for the sentences `build_document` writes.
#
# The dictionary above expands into English, which is right for `process` (it
# feeds a bare noun phrase to a multilingual encoder) and wrong inside a Spanish
# sentence: "columna registro identifier" is a sentence in neither language.
# Only seven of this corpus's tokens hit the English dictionary at all, but two
# of them are the most frequent in the schema — `col` (53 columns) and `id`
# (24) — so the mismatch is not marginal. Applied *over* ABBREVIATIONS, so
# entries with no Spanish equivalent (qty, amt, ...) still expand.
ABBREVIATIONS_ES = {
    "id": "identificador",
    "pk": "clave primaria",
    "fk": "clave foranea",
    "uq": "unico",
    "idx": "indice",
    "col": "columna",
    "tbl": "tabla",
    "fld": "campo",
    "num": "numero",
    "no": "numero",
    "cant": "cantidad",
    "qty": "cantidad",
    "amt": "importe",
    "desc": "descripcion",
    "descr": "descripcion",
    "info": "informacion",
    "config": "configuracion",
    "temp": "temporal",
    "fec": "fecha",
    "dt": "fecha",
    "ts": "marca de tiempo",
    "cod": "codigo",
    "dir": "direccion",
    "addr": "direccion",
    "tel": "telefono",
    "obs": "observaciones",
    "ref": "referencia",
    "reg": "registro",
    "val": "valor",
    "usr": "usuario",
    "emp": "empleado",
    "empl": "empleado",
    "cli": "cliente",
    "prod": "producto",
    "prov": "proveedor",
    "dept": "departamento",
    "cat": "categoria",
    "subcat": "subcategoria",
    "org": "organizacion",
    "msg": "mensaje",
    "txt": "texto",
    "img": "imagen",
    "doc": "documento",
    "seq": "secuencia",
    "attr": "atributo",
    "param": "parametro",
    "min": "minimo",
    "max": "maximo",
    "avg": "promedio",
    "cnt": "conteo",
    "sum": "total",
    "prev": "anterior",
    "curr": "actual",
    "cur": "actual",
    "orig": "original",
    "src": "origen",
    "dest": "destino",
    "dst": "destino",
}


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class TextPreprocessor:
    """
    Preprocesses column/table names to generate clean text.

    The resulting text is lowercase, without redundant prefixes,
    with expanded abbreviations and contextualized with the table name.
    """

    def __init__(self, abbreviations: dict[str, str] | None = None) -> None:
        self.abbreviations = abbreviations or ABBREVIATIONS
        # Compile dictionary for efficient lookup
        self._abbrev_lower = {k.lower(): v for k, v in self.abbreviations.items()}

    @classmethod
    def for_documents(cls) -> TextPreprocessor:
        """A preprocessor whose expansions match the language of the documents.

        `build_document` writes Spanish; the default dictionary expands into
        English. Use this wherever documents are built, so the names inside a
        sentence are in the sentence's own language.
        """
        return cls(abbreviations={**ABBREVIATIONS, **ABBREVIATIONS_ES})

    # ── Full pipeline ─────────────────────────────────────────────────

    def process(
        self,
        column_name: str,
        table_name: str | None = None,
        comment: str | None = None,
    ) -> str:
        """
        Apply the full preprocessing pipeline.

        Args:
            column_name: Column name (e.g. "FECHA_NACIMIENTO").
            table_name: Container table name (e.g. "EMPLEADOS").
            comment: Column comment from the data dictionary, appended
                verbatim (already natural language).

        Returns:
            Clean, contextualized text (e.g. "empleados: fecha nacimiento").
        """
        text = column_name

        # 1. Split camelCase
        text = self._split_camel_case(text)

        # 2. Split snake_case / UPPER_CASE
        text = self._split_on_underscores(text)

        # 3. Remove redundant prefixes
        text = self._remove_redundant_prefixes(text)

        # 4. Tokenize and expand abbreviations
        tokens = text.split()
        tokens = [self._expand_token(t) for t in tokens]

        # 5. Collapse whitespace; lowercase only ALL-CAPS tokens.
        #    Oracle stores unquoted identifiers as uppercase — normalizing
        #    them avoids WordPiece char-fragmentation — while mixed-case
        #    tokens (camelCase remnants) carry signal for cased models.
        text = " ".join(t.lower() if t.isupper() else t for t in tokens).strip()

        # 6. Contextualize with table name
        if table_name:
            table_clean = self.process(table_name)
            text = f"{table_clean}: {text}"

        # 7. Append column comment verbatim
        if comment and comment.strip():
            text = f"{text} | {comment.strip()}"

        return text

    # ── Column documents ──────────────────────────────────────────────

    def build_document(
        self,
        column: ColumnMetadata,
        table_name: str,
        table_comment: str | None = None,
        include_table_comment: bool = True,
    ) -> str:
        """Describe one column as a Spanish sentence, type and keys included.

        `process` returns a bare noun phrase — "empleados: fecha nacimiento" —
        which says what the column is *about* and nothing about how it is
        stored. An encoder reading that cannot tell FECHA_INGRESO DATE from
        FECHA_INGRESO VARCHAR2(20), and telling those apart is exactly the
        `wrong_data_types` anti-pattern. Writing the type out in words puts
        both halves in the same sentence, so the discordance between what a
        column is called and how it is declared becomes something the
        embedding can represent.

        The sentence is Spanish because the schema and its comments are, and
        the encoder is multilingual.

        Returns:
            e.g. "tabla empleados, columna fecha ingreso, tipo texto de
            longitud variable de hasta 20 caracteres, admite nulos"
        """
        parts = [
            f"tabla {self.process(table_name)}",
            f"columna {self.process(column.name)}",
            f"tipo {self._describe_type(column)}",
        ]
        parts.extend(self._describe_constraints(column))

        sentences = [", ".join(parts)]
        if include_table_comment and table_comment and table_comment.strip():
            sentences.append(f"La tabla contiene: {table_comment.strip()}")
        if column.comments and column.comments.strip():
            sentences.append(column.comments.strip())
        return ". ".join(sentence.rstrip(". ") for sentence in sentences) + "."

    def build_documents(
        self,
        schema: object,
        include_table_comment: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Build a document per column across a whole schema.

        Returns:
            (documents, keys) where keys are "TABLE.COLUMN", in the same order,
            so the embeddings can be aligned with any other per-column table.
        """
        documents: list[str] = []
        keys: list[str] = []
        for table in schema.tables:  # type: ignore[attr-defined]
            for column in table.columns:
                documents.append(
                    self.build_document(
                        column,
                        table.name,
                        table.table_comment,
                        include_table_comment=include_table_comment,
                    )
                )
                keys.append(f"{table.name}.{column.name}")
        return documents, keys

    # ── Describing a column in words ──────────────────────────────────

    def _describe_type(self, column: ColumnMetadata) -> str:
        """Render the Oracle type as Spanish prose."""
        oracle_type = (column.data_type or "").upper().strip()
        length = column.data_length
        scale = column.data_scale
        precision = column.data_precision

        if oracle_type in {"VARCHAR2", "VARCHAR", "NVARCHAR2"}:
            return (
                f"texto de longitud variable de hasta {length} caracteres"
                if length
                else "texto de longitud variable"
            )
        if oracle_type in {"CHAR", "NCHAR"}:
            return (
                f"texto de longitud fija de {length} caracteres"
                if length
                else "texto de longitud fija"
            )
        if oracle_type in {"NUMBER", "NUMERIC", "DECIMAL", "FLOAT", "BINARY_DOUBLE", "BINARY_FLOAT"}:
            if scale:
                return f"numero decimal con {precision or 'varios'} digitos y {scale} decimales"
            return "numero entero"
        if oracle_type == "DATE":
            return "fecha"
        if oracle_type.startswith("TIMESTAMP"):
            return "marca de tiempo"
        if oracle_type.startswith("INTERVAL"):
            return "intervalo de tiempo"
        if oracle_type in {"CLOB", "NCLOB"}:
            return "texto largo sin limite de longitud"
        if oracle_type == "LONG":
            return "texto largo en el tipo heredado LONG"
        if oracle_type in {"BLOB", "RAW", "LONG RAW", "BFILE"}:
            return "datos binarios"
        # An unrecognised type is still information — never silently drop it.
        return oracle_type.lower() if oracle_type else "tipo desconocido"

    @staticmethod
    def _describe_constraints(column: ColumnMetadata) -> list[str]:
        """Render nullability and keys as Spanish prose."""
        parts = ["admite nulos" if column.nullable else "obligatorio"]
        if column.is_primary_key:
            parts.append("es clave primaria")
        if column.is_foreign_key:
            target = column.fk_references_table
            parts.append(
                f"es clave foranea hacia {target.lower()}" if target else "es clave foranea"
            )
        if column.is_unique:
            parts.append("con valores unicos")
        if column.is_indexed:
            parts.append("indexada")
        if column.default_value:
            parts.append(f"con valor por defecto {str(column.default_value).strip()}")
        return parts

    def process_batch(self, columns: list[str], table_name: str | None = None) -> list[str]:
        """
        Process a list of column names.

        Args:
            columns: List of column names.
            table_name: Table name (optional, same for all).

        Returns:
            List of preprocessed texts.
        """
        return [self.process(c, table_name) for c in columns]

    # ── Individual steps ──────────────────────────────────────────────

    @staticmethod
    def _split_camel_case(text: str) -> str:
        """
        Split camelCase and PascalCase by inserting spaces before
        uppercase letters, handling consecutive sequences (UUIDValue → uuid value).
        """
        # Insert space before uppercase that follows lowercase
        text = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", text)
        # Insert space before uppercase that follows multiple uppercase letters
        # (e.g. "XMLParser" → "XML Parser")
        text = re.sub(r"([A-ZÁÉÍÓÚÑ]+)([A-ZÁÉÍÓÚÑ][a-záéíóúñ])", r"\1 \2", text)
        return text

    @staticmethod
    def _split_on_underscores(text: str) -> str:
        """Replace underscores with spaces."""
        return text.replace("_", " ")

    @staticmethod
    def _remove_redundant_prefixes(text: str) -> str:
        """Remove prefixes like tbl_, col_, fld_ at the start of text."""
        prefixes = ["tbl ", "col ", "fld ", "tab "]
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        return text

    def _expand_token(self, token: str) -> str:
        """Expand an abbreviation if found in the dictionary."""
        # Look up in abbreviation dictionary (case-insensitive)
        lower = token.lower().strip(".,;:!?()[]{}")
        expansion = self._abbrev_lower.get(lower)
        if expansion:
            return expansion
        return token


# ---------------------------------------------------------------------------
# Quick utility
# ---------------------------------------------------------------------------


def preprocess(column_name: str, table_name: str | None = None) -> str:
    """
    Quick-access function (no class instantiation needed).
    """
    return TextPreprocessor().process(column_name, table_name)
