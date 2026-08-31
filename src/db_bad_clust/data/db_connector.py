"""
db_connector.py — Oracle database connection manager.

Provides OracleConnector which reads config from YAML and manages
a connection lifecycle (connect / close). Raises typed exceptions
from exceptions.py on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import oracledb
import yaml

from db_bad_clust.exceptions import ConfigError
from db_bad_clust.exceptions import ConnectionError as DBConnectionError

logger = logging.getLogger(__name__)


class OracleConnector:
    """Oracle database connection manager.

    Reads connection parameters from a YAML config file and provides
    connect() / close() lifecycle.

    Args:
        config_path: Path to the YAML configuration file (relative or absolute).
    """

    def __init__(self, config_path: str = "../config.yaml") -> None:
        self.config: dict[str, Any] = self._load_config(config_path)
        self.connection: oracledb.Connection | None = None

    # ── Config loading ────────────────────────────────────────────────

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load database configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            ConfigError: If the file is missing, unreadable, or contains invalid YAML.
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        try:
            with path.open("r") as f:
                cfg: dict[str, Any] = yaml.safe_load(f)
            if cfg is None:
                raise ConfigError(f"Empty configuration file: {config_path}")
            return cfg
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_path}: {e}") from e
        except OSError as e:
            raise ConfigError(f"Cannot read {config_path}: {e}") from e

    # ── Connection ────────────────────────────────────────────────────

    def connect(self) -> oracledb.Connection:
        """Establish a connection to Oracle Database.

        Returns:
            An active oracledb.Connection instance.

        Raises:
            ConnectionError: If connection parameters are missing or the
                database is unreachable.
            ConfigError: If required config keys are missing.
        """
        db_cfg = self.config.get("database", {})
        user = db_cfg.get("user")
        password = db_cfg.get("password")
        dsn = db_cfg.get("dsn")

        if not all([user, password, dsn]):
            raise ConfigError(
                "Missing database configuration keys. "
                "Expected 'database.user', 'database.password', 'database.dsn'"
            )

        try:
            self.connection = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
            )
            logger.info("Connected to Oracle Database")
            return self.connection
        except oracledb.Error as e:
            raise DBConnectionError(f"Failed to connect to Oracle: {e}") from e

    def close(self) -> None:
        """Close the database connection if it is open."""
        if self.connection is not None:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except oracledb.Error as e:
                logger.warning("Error while closing connection: %s", e)
            finally:
                self.connection = None
