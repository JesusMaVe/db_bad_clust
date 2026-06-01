"""
orchestrator.py — Phase 1 entry point.

Generates poorly-designed database schemas (anti-patterns), creates them
in Oracle 23c, populates with dirty data, and verifies the results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import oracledb

sys.path.insert(0, str(Path(__file__).parent))

from anti_patterns import generate_bad_data_for_table, generate_poorly_designed_tables
from db_connector import OracleConnector
from ddl_generator import DDLGenerator
from dml_generator import DMLGenerator
from exceptions import BadDBError, DatabaseError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full Phase 1 pipeline: connect → create tables → insert data → verify."""
    logger.info("Starting Phase 1: generating bad training data...")

    db = OracleConnector()
    try:
        connection = db.connect()

        # 2. Generate anti-pattern table catalog
        logger.info("Generating anti-pattern catalog...")
        bad_tables = generate_poorly_designed_tables()
        logger.info("Catalog created: %d problematic tables", len(bad_tables))

        # 3. Create tables
        ddl_gen = DDLGenerator(connection)
        for table in bad_tables:
            logger.info("Creating table: %s", table.name)
            ddl_gen.create_table(table)

        connection.commit()
        logger.info("All tables created and committed")

        # 4. Populate with dirty data
        dml_gen = DMLGenerator(connection)
        for table in bad_tables:
            logger.info("Generating data for: %s", table.name)
            bad_data = generate_bad_data_for_table(table)
            dml_gen.insert_data(bad_data)
            connection.commit()

        connection.commit()
        logger.info("Database populated with dirty data")

        # 5. Verify inserted data
        logger.info("\n" + "=" * 50)
        logger.info("FINAL VERIFICATION:")
        logger.info("=" * 50)

        cursor = connection.cursor()
        current_tables = [table.name for table in bad_tables]

        for table_name in current_tables:
            try:
                row_count = dml_gen.verify_data(table_name)
                if row_count > 0:
                    cursor.execute(f'SELECT * FROM "{table_name}" WHERE ROWNUM <= 3')
                    logger.info("\nTable: %s (%d rows)", table_name, row_count)
                    logger.info("   Bad data examples:")
                    for row in cursor:
                        logger.info("   %s", row)
            except oracledb.Error as e:
                logger.error("Error verifying %s: %s", table_name, e)

        logger.info("\nDone! Database is ready for ML model training")

    except DatabaseError as e:
        logger.error("Database error: %s", e)
        try:
            db.connection.rollback()
        except Exception:
            pass
    except BadDBError as e:
        logger.error("Pipeline error: %s", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
