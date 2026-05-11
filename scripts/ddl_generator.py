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
        """Limpia nombre de columna pero mantiene problemas intencionales."""
        # Quitamos espacios al inicio y final
        cleaned = name.strip()
        # Si tiene espacios internos, los reemplazamos por guión bajo
        if ' ' in cleaned:
            cleaned = cleaned.replace(' ', '_')
        # Si es palabra reservada, la mantenemos pero en mayúsculas
        return cleaned.upper()
    
    def create_table(self, table: AntiPatternTable):
        """Crea una tabla con los anti-patrones definidos."""
        # Primero eliminamos la tabla si existe
        self._drop_table_if_exists(table.name)
        
        # Limpiamos los nombres de columna para que Oracle los acepte
        cleaned_columns = {}
        for col_name, col_type in table.columns.items():
            clean_name = self._clean_column_name(col_name)
            cleaned_columns[clean_name] = col_type
        
        # Construimos CREATE TABLE
        columns_def = []
        for col_name, col_type in cleaned_columns.items():
            # Si el nombre original tenía problemas, usamos el limpio
            # pero mantenemos el tipo de dato malo como anti-patrón
            columns_def.append(f"{col_name} {col_type}")
        
        columns_sql = ",\n    ".join(columns_def)
        sql = f"CREATE TABLE {table.name} (\n    {columns_sql}\n)"
        
        try:
            self.cursor.execute(sql)
            logger.info(f"Tabla '{table.name}' creada con {len(columns_def)} columnas")
            logger.debug(f"SQL ejecutado: {sql}")
            
        except oracledb.Error as e:
            logger.error(f"Error creando tabla {table.name}: {e}")
            # Intento alternativo sin limpieza
            self._create_table_raw(table)
    
    def _create_table_raw(self, table: AntiPatternTable):
        """Intenta crear la tabla exactamente como se definió."""
        try:
            columns_def = []
            for col_name, col_type in table.columns.items():
                # Envolvemos en comillas para permitir cualquier nombre
                columns_def.append(f'"{col_name.strip()}" {col_type}')
            
            columns_sql = ",\n    ".join(columns_def)
            sql = f'CREATE TABLE "{table.name}" (\n    {columns_sql}\n)'
            
            self.cursor.execute(sql)
            logger.info(f"Tabla '{table.name}' creada con nombres originales")
            
        except oracledb.Error as e:
            logger.error(f"Error fatal creando tabla {table.name}: {e}")
    
    def _drop_table_if_exists(self, table_name: str):
        """Elimina la tabla si existe para desarrollo limpio."""
        try:
            self.cursor.execute(f'DROP TABLE "{table_name}" CASCADE CONSTRAINTS PURGE')
            logger.debug(f"Tabla existente '{table_name}' eliminada")
        except oracledb.Error:
            # La tabla no existe, continuamos
            pass
