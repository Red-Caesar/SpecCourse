import sqlite3
from pathlib import Path

from spec_course.scripts.utils import load_config


def create_database(db_name: str) -> None:
    """Create SQLite database and initialize tables from YAML definitions"""
    conn = sqlite3.connect(db_name)
    conn.execute("PRAGMA foreign_keys = ON")

    current_dir = Path(__file__).parent
    tables = load_config(current_dir / "tables.yaml").get("database_tables", {})
    try:
        for table_name, values in tables.items():
            columns = values["columns"]
            dependent_columns = values.get("dependent_columns", {})

            column_defs = [
                f"{col_name} {col_type}" for col_name, col_type in columns.items()
            ]
            dependent_column_defs = [
                f" FOREIGN KEY ({col_name}) REFERENCES {col_base_table}"
                for col_name, col_base_table in dependent_columns.items()
            ]

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {", ".join(column_defs + dependent_column_defs)}
            )
            """

            conn.execute(create_table_sql)

        conn.commit()
    except sqlite3.Error as e:
        print(f"Error creating tables: {e}")
    finally:
        conn.close()


def insert_model(db_name: str, model_name: str) -> int:
    """Insert a row into models table"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "INSERT INTO models (model_name) VALUES (?)", (model_name,)
        )
        model_id = cursor.lastrowid
        conn.commit()
        return model_id
    finally:
        conn.close()


def insert_quantization(db_name: str, quantization_type: str) -> int:
    """Insert a row into quantizations table"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "INSERT INTO quantizations (quantization_type) VALUES (?)",
            (quantization_type,),
        )
        quantization_id = cursor.lastrowid
        conn.commit()
        return quantization_id
    finally:
        conn.close()


def insert_sd_setup(
    db_name: str,
    target_model_id: int,
    target_quantization_id: int,
    draft_model_id: int,
    draft_quantization_id: int,
) -> int:
    """Insert a row into sd_setups table"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            """INSERT INTO sd_setups
            (target_model_id, target_quantization_id, draft_model_id, draft_quantization_id)
            VALUES (?, ?, ?, ?)""",
            (
                target_model_id,
                target_quantization_id,
                draft_model_id,
                draft_quantization_id,
            ),
        )
        sd_setup_id = cursor.lastrowid
        conn.commit()
        return sd_setup_id
    finally:
        conn.close()


def insert_accuracy(
    db_name: str, model_id: int, quantization_id: int, gsm8k_score: float, date: str
) -> None:
    """Insert a row into accuracy table"""
    conn = sqlite3.connect(db_name)
    try:
        conn.execute(
            "INSERT INTO accuracy (model_id, quantization_id, gsm8k_score, date) VALUES (?, ?, ?, ?)",
            (model_id, quantization_id, gsm8k_score, date),
        )
        conn.commit()
    finally:
        conn.close()


def insert_load_test_performance(
    db_name: str, sd_setup_id: int, rps: int, latency: float, date: str
) -> None:
    """Insert a row into ld_performances table"""
    conn = sqlite3.connect(db_name)
    try:
        conn.execute(
            "INSERT INTO ld_performances (sd_setup_id, rps, end_to_end_latencty, date) VALUES (?, ?, ?, ?)",
            (sd_setup_id, rps, latency, date),
        )
        conn.commit()
    finally:
        conn.close()


def insert_sd_performance(
    db_name: str, sd_setup_id: int, acceptance_rate: float, date: str
) -> None:
    """Insert a row into sd_performances table"""
    conn = sqlite3.connect(db_name)
    try:
        conn.execute(
            "INSERT INTO sd_performances (sd_setup_id, acceptance_rate, date) VALUES (?, ?, ?)",
            (sd_setup_id, acceptance_rate, date),
        )
        conn.commit()
    finally:
        conn.close()


def get_model_id(db_name: str, model_name: str) -> int | None:
    """Search for model by name and return its ID if exists"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", (model_name,)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()


def get_quantization_id(db_name: str, quantization_type: str) -> int | None:
    """Search for quantization by type and return its ID if exists"""
    conn = sqlite3.connect(db_name)
    try:
        cursor = conn.execute(
            "SELECT quantization_id FROM quantizations WHERE quantization_type = ?",
            (quantization_type,),
        )
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()
