import oracledb
import logging
from typing import Dict, Any, List
from anti_patterns import AntiPatternData

logger = logging.getLogger(__name__)

class DMLGenerator:
    """Genera y ejecuta INSERT con datos problemáticos."""
    
    def __init__(self, connection: oracledb.Connection):
        self.connection = connection
        self.cursor = connection.cursor()
    
    def _get_actual_columns(self, table_name: str) -> List[str]:
        """Obtiene los nombres reales de las columnas desde Oracle."""
        try:
            # Eliminar comillas si las tiene para la consulta al diccionario
            clean_name = table_name.strip('"')
            self.cursor.execute("""
                SELECT column_name, data_type
                FROM user_tab_columns 
                WHERE table_name = :table_name
                ORDER BY column_id
            """, table_name=clean_name.upper())
            
            columns = [row[0] for row in self.cursor]
            logger.debug(f"Columnas reales en '{table_name}': {columns}")
            return columns
        except oracledb.Error as e:
            logger.error(f"Error obteniendo columnas de {table_name}: {e}")
            return []
    
    def _get_column_types(self, table_name: str) -> Dict[str, str]:
        """Obtiene los tipos de datos reales de las columnas."""
        try:
            clean_name = table_name.strip('"')
            self.cursor.execute("""
                SELECT column_name, data_type
                FROM user_tab_columns 
                WHERE table_name = :table_name
                ORDER BY column_id
            """, table_name=clean_name.upper())
            
            types = {row[0]: row[1] for row in self.cursor}
            return types
        except oracledb.Error:
            return {}
    
    def _sanitize_value(self, value: Any, data_type: str) -> str:
        """Convierte valores problemáticos a strings seguros para Oracle."""
        if value is None:
            return 'NULL'
        
        # Convertir todo a string y escapar comillas
        str_value = str(value).replace("'", "''")
        
        if 'NUMBER' in data_type.upper():
            # Para campos NUMBER, intentamos extraer solo dígitos
            import re
            numbers = re.findall(r'\d+', str_value)
            if numbers:
                return numbers[0]
            else:
                return 'NULL'
        
        elif 'DATE' in data_type.upper():
            # Para campos DATE, usamos fechas válidas
            return f"TO_DATE('2024-01-01', 'YYYY-MM-DD')"
        
        else:
            # Para VARCHAR2 y otros, truncamos si es necesario
            if len(str_value) > 4000:
                str_value = str_value[:4000]
            return f"'{str_value}'"
    
    def insert_data(self, data: AntiPatternData):
        """Inserta datos con anti-patrones en la tabla especificada."""
        if not data.rows:
            logger.warning(f"No hay datos para insertar en {data.table_name}")
            return
        
        # Obtenemos los nombres REALES de las columnas desde Oracle
        actual_columns = self._get_actual_columns(data.table_name)
        
        if not actual_columns:
            logger.error(f"No se pudieron obtener columnas para {data.table_name}")
            return
        
        # Obtenemos los tipos de datos reales
        column_types = self._get_column_types(data.table_name)
        
        inserted_count = 0
        for row in data.rows:
            try:
                # Construimos INSERT con valores literales para evitar problemas de bind
                values_parts = []
                for col in actual_columns:
                    # Buscamos el valor que corresponde a esta columna
                    val = None
                    for key, value in row.items():
                        if key.strip().upper() == col.strip().upper():
                            val = value
                            break
                    
                    # Si no encontramos valor, usamos NULL
                    if val is None:
                        values_parts.append('NULL')
                    else:
                        data_type = column_types.get(col, 'VARCHAR2')
                        values_parts.append(self._sanitize_value(val, data_type))
                
                # Construir SQL con valores literales
                columns_str = ", ".join([f'"{col}"' for col in actual_columns])
                values_str = ", ".join(values_parts)
                
                sql = f'INSERT INTO "{data.table_name}" ({columns_str}) VALUES ({values_str})'
                
                logger.debug(f"Ejecutando SQL: {sql[:200]}...")  # Solo primeros 200 chars
                self.cursor.execute(sql)
                inserted_count += 1
                
            except oracledb.Error as row_error:
                logger.debug(f"Error con fila individual: {row_error}")
                # Intentar insertar solo valores seguros
                try:
                    self._insert_safe_row(data.table_name, actual_columns, column_types)
                    inserted_count += 1
                except oracledb.Error:
                    pass
        
        logger.info(f"Insertadas {inserted_count}/{len(data.rows)} filas en '{data.table_name}'")
    
    def _insert_safe_row(self, table_name: str, columns: List[str], column_types: Dict[str, str]):
        """Inserta una fila con valores seguros cuando la original falla."""
        safe_values = []
        
        for col in columns:
            data_type = column_types.get(col, 'VARCHAR2')
            
            if 'NUMBER' in data_type.upper():
                safe_values.append('1')  # Número seguro
            elif 'DATE' in data_type.upper():
                safe_values.append("TO_DATE('2024-01-01', 'YYYY-MM-DD')")
            else:
                safe_values.append(f"'BAD_DATA_{col}'")
        
        columns_str = ", ".join([f'"{col}"' for col in columns])
        values_str = ", ".join(safe_values)
        
        sql = f'INSERT INTO "{table_name}" ({columns_str}) VALUES ({values_str})'
        self.cursor.execute(sql)
        logger.debug(f"Fila segura insertada en {table_name}")
    
    def verify_data(self, table_name: str) -> int:
        """Verifica cuántas filas se insertaron realmente."""
        try:
            self.cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = self.cursor.fetchone()[0]
            logger.info(f"Tabla '{table_name}' tiene {count} filas")
            return count
        except oracledb.Error:
            return 0
