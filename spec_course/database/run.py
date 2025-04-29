import argparse
import os
from pathlib import Path
from typing import Type

from spec_course.database.db import create_database
from spec_course.database.etl.accuracy import Accuracy
from spec_course.database.etl.base import ETLBase
from spec_course.scripts.utils import setup_logger

logger = setup_logger(log_name="etl_process")


def get_etl_class(etl_name: str) -> Type[ETLBase]:
    """Map ETL class name to actual class"""
    etl_classes = {
        "accuracy": Accuracy,
    }

    if etl_name not in etl_classes:
        raise ValueError(
            f"Unknown ETL class: {etl_name}. Available classes: {list(etl_classes.keys())}"
        )

    return etl_classes[etl_name]


def process_files(etl_class: Type[ETLBase], data_dir: Path, db_name: str) -> None:
    """Process all files in directory using specified ETL class"""
    etl = etl_class(db_name)
    print(list(data_dir.glob("results_*.json")))
    print(data_dir)
    for file_path in data_dir.glob("*/results_*.json"):
        try:
            etl.run(file_path)
            logger.info(f"Successfully processed: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Run ETL process for database")
    parser.add_argument(
        "--etl_class",
        type=str,
        required=True,
        choices=["accuracy"],
        help="ETL class to use (e.g., accuracy)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing data files to process",
    )
    parser.add_argument(
        "--db_name", type=str, required=True, help="SQLite database name"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    if not os.path.exists(args.db_name):
        logger.info(f"Creating new database: {args.db_name}")
        create_database(args.db_name)

    try:
        etl_class = get_etl_class(args.etl_class)
        process_files(etl_class, data_dir, args.db_name)
    except ValueError as e:
        logger.error(e)


if __name__ == "__main__":
    main()
