import sys
from pathlib import Path

# Agregamos el directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from db_connector import OracleConnector
from ddl_generator import DDLGenerator
from dml_generator import DMLGenerator
from anti_patterns import generate_poorly_designed_tables, generate_bad_data_for_table
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Iniciando generación de datos de entrenamiento incorrectos...")
    
    # 1. Conectar a Oracle
    db = OracleConnector()
    connection = db.connect()
    
    try:
        # 2. Generar catálogo de tablas malas
        logger.info("Generando catálogo de anti-patrones...")
        bad_tables = generate_poorly_designed_tables()
        logger.info(f"Catálogo creado: {len(bad_tables)} tablas problemáticas")
        
        # 3. Crear las tablas
        ddl_gen = DDLGenerator(connection)
        for table in bad_tables:
            logger.info(f"Creando tabla: {table.name}")
            ddl_gen.create_table(table)
        
        # 4. Hacer commit después de crear tablas
        connection.commit()
        logger.info("Tablas creadas y confirmadas")
        
        # 5. Poblar con datos incorrectos
        dml_gen = DMLGenerator(connection)
        for table in bad_tables:
            logger.info(f"Generando datos para: {table.name}")
            bad_data = generate_bad_data_for_table(table)
            dml_gen.insert_data(bad_data)
            # Commit después de cada tabla para no perder datos si algo falla
            connection.commit()
        
        # 6. Commit final
        connection.commit()
        logger.info("Base de datos poblada exitosamente con datos incorrectos")
        
        # 7. Verificar datos insertados
        logger.info("\n" + "="*50)
        logger.info("VERIFICACIÓN FINAL:")
        logger.info("="*50)
        
        cursor = connection.cursor()
        
        # Solo verificamos las tablas que creamos esta vez
        current_tables = [table.name for table in bad_tables]
        
        for table_name in current_tables:
            try:
                row_count = dml_gen.verify_data(table_name)
                if row_count > 0:
                    # Mostrar ejemplos de datos malos
                    cursor.execute(f'SELECT * FROM "{table_name}" WHERE ROWNUM <= 3')
                    logger.info(f"\nTabla: {table_name} ({row_count} filas)")
                    logger.info("   Ejemplos de datos incorrectos:")
                    for row in cursor:
                        logger.info(f"   {row}")
            except oracledb.Error as e:
                logger.error(f"Error verificando {table_name}: {e}")
        
        logger.info("\n¡Listo! La base de datos está lista para entrenar tu modelo ML")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        connection.rollback()
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
