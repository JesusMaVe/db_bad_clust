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
