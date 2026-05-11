import oracledb
import yaml
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class OracleConnector:
    """Manejador de conexión a Oracle Database."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        self.config = self._load_config(config_path)
        self.connection: Optional[oracledb.Connection] = None
        
    def _load_config(self, config_path: str) -> dict:
        """Carga configuración desde archivo YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def connect(self):
        """Establece conexión con Oracle Database."""
        try:
            self.connection = oracledb.connect(
                user=self.config['database']['user'],
                password=self.config['database']['password'],
                dsn=self.config['database']['dsn']
            )
            logger.info(f"Conexión exitosa a Oracle Database")
            return self.connection
        except oracledb.Error as e:
            logger.error(f"Error conectando a Oracle: {e}")
            raise
    
    def close(self):
        """Cierra la conexión a la base de datos."""
        if self.connection:
            self.connection.close()
            logger.info("Conexión cerrada")
