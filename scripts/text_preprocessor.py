"""
text_preprocessor.py — Preprocesamiento textual de nombres de columnas

Propósito:
  Transformar nombres técnicos de columnas (camelCase, snake_case, UPPER_CASE)
  en texto limpio y contextualizado para alimentar BERT.

  Pipeline:
    1. Separar camelCase / snake_case / UPPER_CASE
    2. Expandir abreviaturas comunes (ID → identifier, FK → foreign key, etc.)
    3. Eliminar prefijos redundantes (tbl_, col_, fld_)
    4. Poner en minúsculas y colapsar espacios
    5. Contextualizar con nombre de tabla → "clientes: nombre cliente"

Uso:
  preprocessor = TextPreprocessor()
  text = preprocessor.process("FECHA_NACIMIENTO", table_name="EMPLEADOS")
  # → "empleados: fecha nacimiento"
"""

import re
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diccionario de abreviaturas
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
    "min": "minute",
    "sec": "second",
    "ms": "milliseconds",
    "ts": "timestamp",
}


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Preprocesa nombres de columna/tabla para generar texto limpio.

    El texto resultante está en minúsculas, sin prefijos redundantes,
    con abreviaturas expandidas y contextualizado con el nombre de la tabla.
    """

    def __init__(self, abbreviations: dict = None):
        self.abbreviations = abbreviations or ABBREVIATIONS
        # Compilar diccionario para búsqueda eficiente
        self._abbrev_lower = {k.lower(): v for k, v in self.abbreviations.items()}

    # ── Pipeline completo ─────────────────────────────────────────────

    def process(self, column_name: str, table_name: str = None) -> str:
        """
        Aplica el pipeline completo de preprocesamiento.

        Args:
            column_name: Nombre de la columna (ej. "FECHA_NACIMIENTO").
            table_name: Nombre de la tabla contenedora (ej. "EMPLEADOS").

        Returns:
            Texto limpio y contextualizado (ej. "empleados: fecha nacimiento").
        """
        text = column_name

        # 1. Separar camelCase
        text = self._split_camel_case(text)

        # 2. Separar snake_case / UPPER_CASE
        text = self._split_on_underscores(text)

        # 3. Remover prefijos redundantes
        text = self._remove_redundant_prefixes(text)

        # 4. Tokenizar y expandir abreviaturas
        tokens = text.split()
        tokens = [self._expand_token(t) for t in tokens]

        # 5. Colapsar whitespace y lowercase
        text = " ".join(tokens).lower().strip()

        # 6. Contextualizar con nombre de tabla
        if table_name:
            table_clean = self.process(table_name) if table_name else ""
            text = f"{table_clean}: {text}"

        return text

    def process_batch(self, columns: list, table_name: str = None) -> list:
        """
        Procesa una lista de nombres de columna.

        Args:
            columns: Lista de nombres de columna.
            table_name: Nombre de la tabla (opcional, mismo para todas).

        Returns:
            Lista de textos preprocesados.
        """
        return [self.process(c, table_name) for c in columns]

    # ── Pasos individuales ────────────────────────────────────────────

    @staticmethod
    def _split_camel_case(text: str) -> str:
        """
        Separa camelCase y PascalCase insertando espacios antes de
        mayúsculas, manejando secuencias consecutivas (UUIDValue → uuid value).
        """
        # Insertar espacio antes de mayúscula que sigue a minúscula
        text = re.sub(r'([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])', r'\1 \2', text)
        # Insertar espacio antes de mayúscula que sigue a múltiples mayúsculas
        # (ej. "XMLParser" → "XML Parser")
        text = re.sub(r'([A-ZÁÉÍÓÚÑ]+)([A-ZÁÉÍÓÚÑ][a-záéíóúñ])', r'\1 \2', text)
        return text

    @staticmethod
    def _split_on_underscores(text: str) -> str:
        """Reemplaza guiones bajos con espacios."""
        return text.replace("_", " ")

    @staticmethod
    def _remove_redundant_prefixes(text: str) -> str:
        """Elimina prefijos como tbl_, col_, fld_ al inicio del texto."""
        prefixes = ["tbl ", "col ", "fld ", "tab "]
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break
        return text

    def _expand_token(self, token: str) -> str:
        """Expande una abreviatura si está en el diccionario."""
        # Buscar en diccionario de abreviaturas (case-insensitive)
        lower = token.lower().strip(".,;:!?()[]{}")
        expansion = self._abbrev_lower.get(lower)
        if expansion:
            return expansion
        return token


# ---------------------------------------------------------------------------
# Utilidad rápida
# ---------------------------------------------------------------------------

def preprocess(column_name: str, table_name: str = None) -> str:
    """
    Función de acceso directo (sin instanciar clase).
    """
    return TextPreprocessor().process(column_name, table_name)
