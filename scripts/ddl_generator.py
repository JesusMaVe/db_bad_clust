import oracledb
import logging
from anti_patterns import AntiPatternTable

logger = logging.getLogger(__name__)

class DDLGenerator:
    """Genera y ejecuta sentencias DDL con malas prácticas."""
    
    def __init__(self, connection: oracledb.Connection):
        self.connection = connection
        self.cursor = connection.cursor()
    
    def _clean_column_name(self, name: str) -> str:
        cleaned = name.strip()
        if ' ' in cleaned:
            cleaned = cleaned.replace(' ', '_')
        return cleaned.upper()

    def create_table(self, table: AntiPatternTable):
        self._drop_table_if_exists(table.name)
        cleaned_columns = self._build_cleaned_columns(table)
        sql = self._build_create_sql(table.name, cleaned_columns, table.check_constraints)

        try:
            self.cursor.execute(sql)
            logger.info(f"Tabla '{table.name}' creada con {len(cleaned_columns)} columnas")
        except oracledb.Error as e:
            logger.error(f"Error creando tabla {table.name}: {e}")
            self._create_table_raw(table)

        self._create_fk_constraints(table, cleaned_columns)
        self._create_indexes(table, cleaned_columns)

    def _build_cleaned_columns(self, table: AntiPatternTable) -> Dict[str, str]:
        cleaned = {}
        for col_name, col_type in table.columns.items():
            clean_name = self._clean_column_name(col_name)
            cleaned[clean_name] = col_type
        return cleaned

    def _build_create_sql(self, table_name: str, columns: Dict[str, str], check_constraints: List[str]) -> str:
        columns_def = []
        for col_name, col_type in columns.items():
            columns_def.append(f"{col_name} {col_type}")

        for i, expr in enumerate(check_constraints):
            constraint_name = f"CHK_{table_name}_{i}"
            columns_def.append(f'CONSTRAINT "{constraint_name}" CHECK ({expr}) DISABLE')

        columns_sql = ",\n    ".join(columns_def)
        return f"CREATE TABLE {table_name} (\n    {columns_sql}\n)"

    def _create_fk_constraints(self, table: AntiPatternTable, cleaned_columns: Dict[str, str]):
        for col_name, ref_spec in table.fk_constraints.items():
            clean_col = self._clean_column_name(col_name)
            if clean_col not in cleaned_columns:
                logger.warning(f"Columna FK '{clean_col}' no encontrada en {table.name}")
                continue

            parts = ref_spec.split()
            ref_table_col = parts[0]
            directives = parts[1:] if len(parts) > 1 else []

            if "(" in ref_table_col and ")" in ref_table_col:
                ref_table = ref_table_col[:ref_table_col.index("(")]
                ref_col = ref_table_col[ref_table_col.index("(")+1:ref_table_col.index(")")]
            else:
                ref_table = ref_table_col
                ref_col = "ID"

            constraint_name = f"FK_{table.name}_{clean_col}"
            disable = " DISABLE" if any(d.upper() == "DISABLE" for d in directives) else ""
            cascade = " ON DELETE CASCADE" if any(d.upper() == "CASCADE" for d in directives) else ""

            sql = f'ALTER TABLE {table.name} ADD CONSTRAINT "{constraint_name}" FOREIGN KEY ("{clean_col}") REFERENCES "{ref_table}" ("{ref_col}"){cascade}{disable}'

            try:
                self.cursor.execute(sql)
                logger.info(f"FK creada: {constraint_name} -> {ref_table}({ref_col}){disable}")
            except oracledb.Error as e:
                logger.warning(f"Error creando FK {constraint_name}: {e}")

    def _create_indexes(self, table: AntiPatternTable, cleaned_columns: Dict[str, str]):
        for i, cols in enumerate(table.indexes):
            clean_cols = [self._clean_column_name(c) for c in cols]
            invalid = [c for c in clean_cols if c not in cleaned_columns]
            if invalid:
                logger.warning(f"Columnas de índice no encontradas: {invalid}")
                continue

            index_name = f"IDX_{table.name}_{i}"
            cols_str = ", ".join(clean_cols)
            sql = f'CREATE INDEX "{index_name}" ON {table.name} ({cols_str})'

            try:
                self.cursor.execute(sql)
                logger.info(f"Índice creado: {index_name} en ({cols_str})")
            except oracledb.Error as e:
                logger.warning(f"Error creando índice {index_name}: {e}")

    def _create_table_raw(self, table: AntiPatternTable):
        try:
            columns_def = []
            for col_name, col_type in table.columns.items():
                columns_def.append(f'"{col_name.strip()}" {col_type}')

            columns_sql = ",\n    ".join(columns_def)
            sql = f'CREATE TABLE "{table.name}" (\n    {columns_sql}\n)'

            self.cursor.execute(sql)
            logger.info(f"Tabla '{table.name}' creada con nombres originales")
        except oracledb.Error as e:
            logger.error(f"Error fatal creando tabla {table.name}: {e}")

    def _drop_table_if_exists(self, table_name: str):
        try:
            self.cursor.execute(f'DROP TABLE "{table_name}" CASCADE CONSTRAINTS PURGE')
            logger.debug(f"Tabla existente '{table_name}' eliminada")
        except oracledb.Error:
            pass
