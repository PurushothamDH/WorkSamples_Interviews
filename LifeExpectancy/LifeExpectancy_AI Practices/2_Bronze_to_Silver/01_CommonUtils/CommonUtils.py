# Databricks notebook source
# DBTITLE 1,CommonUtils - Shared Utilities
# MAGIC %md
# MAGIC # CommonUtils — Shared Utilities
# MAGIC
# MAGIC Shared helper class used by all agents in the Bronze-to-Silver pipeline.
# MAGIC
# MAGIC **Provides:**
# MAGIC - Spark session management
# MAGIC - Table read/write helpers
# MAGIC - Column type detection (numeric / categorical)
# MAGIC - Logging utility
# MAGIC - Schema summary

# COMMAND ----------

# DBTITLE 1,CommonUtils Class
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnull, approx_count_distinct
import logging


class CommonUtils:
    """Shared utilities for all agents in the Bronze-to-Silver pipeline."""

    def __init__(self, spark=None):
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("BronzeToSilver")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
        return logger

    # ── Table I/O ──────────────────────────────────────────────
    def read_table(self, table_name):
        """Read a Unity Catalog table into a DataFrame."""
        self.logger.info(f"Reading table: {table_name}")
        return self.spark.table(table_name)

    def write_delta_table(self, df, table_name, mode="overwrite"):
        """Write a DataFrame to a Delta table."""
        self.logger.info(f"Writing to Delta table: {table_name} (mode={mode})")
        df.write.mode(mode).saveAsTable(table_name)

    # ── Column Helpers ────────────────────────────────────────
    def get_numeric_columns(self, df):
        """Return list of numeric column names."""
        return [
            c for c, d in df.dtypes
            if d in ("int", "bigint", "double", "float", "decimal")
        ]

    def get_categorical_columns(self, df):
        """Return list of categorical (string) column names."""
        return [c for c, d in df.dtypes if d == "string"]

    def get_schema_summary(self, df):
        """Return a summary of the DataFrame schema."""
        return {c: d for c, d in df.dtypes}

    # ── Display Helpers ────────────────────────────────────────
    def print_section(self, title):
        """Print a section separator."""
        sep = "=" * 60
        print(f"\n{sep}\n{title}\n{sep}")

# COMMAND ----------

