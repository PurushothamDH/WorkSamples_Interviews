# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,MasterAgent Overview
# MAGIC %md
# MAGIC # MasterAgent — Multi-Agent Orchestrator
# MAGIC
# MAGIC Orchestrates the full Bronze-to-Gold analytics pipeline:
# MAGIC
# MAGIC ```
# MAGIC MasterAgent
# MAGIC      ↓
# MAGIC EDAAgent
# MAGIC      ├── Data Quality
# MAGIC      ├── Null Analysis
# MAGIC      ├── Outlier Analysis
# MAGIC      └── Anomaly Analysis
# MAGIC      ↓
# MAGIC DiagnosticAgent
# MAGIC      ├── Correlation
# MAGIC      ├── Root Cause
# MAGIC      ├── Driver Analysis
# MAGIC      └── Insights
# MAGIC      ↓
# MAGIC MLAgent
# MAGIC      ├── Feature Engineering
# MAGIC      ├── Train Models
# MAGIC      ├── Select Best Model
# MAGIC      └── Prediction Table
# MAGIC      ↓
# MAGIC Save Delta Tables
# MAGIC ```
# MAGIC
# MAGIC **Usage:** Set the table names and target column in the execution cell, then run all cells.

# COMMAND ----------

# DBTITLE 1,Import Agent Notebooks
# Import all agent notebooks via %run so their classes are available
%run ../01_CommonUtils/CommonUtils
%run ../02_EDAAgent/EDAAgent
%run ../03_DiagnosticAgent/DiagnosticAgent
%run ../04_MLAgent/MLAgent

# COMMAND ----------

# DBTITLE 1,MasterAgent Class
from pyspark.sql import SparkSession


class MasterAgent:
    """Master Agent: Orchestrates EDA → Diagnostic → ML → Save Delta Tables."""

    def __init__(self, spark=None):
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.eda_results = None
        self.diagnostic_results = None
        self.ml_results = None

    def run(self, bronze_table, target_column, silver_table, gold_table, categorical_cols=None):
        """Execute the full Bronze-to-Silver-to-Gold pipeline.

        Args:
            bronze_table:   Fully qualified Bronze table name (EDA input)
            target_column:  Target variable for correlation & ML
            silver_table:   Fully qualified Silver table name (Diagnostic + ML input)
            gold_table:     Fully qualified Gold table name (prediction output)
            categorical_cols: List of categorical columns for ML encoding
        """
        print("╔" + "═" * 58 + "╗")
        print("║        MULTI-AGENT ANALYTICS SYSTEM - BRONZE TO GOLD      ║")
        print("╚" + "═" * 58 + "╝")

        # ── Step 1: EDA Agent ─────────────────────────────────
        print("\n[1/4] Launching EDA Agent...")
        eda_agent = EDAAgent(self.spark)
        self.eda_results = eda_agent.run(bronze_table)

        # ── Step 2: Diagnostic Agent ──────────────────────────
        print("\n[2/4] Launching Diagnostic Agent...")
        diag_agent = DiagnosticAgent(self.spark)
        self.diagnostic_results = diag_agent.run(silver_table, target_column)

        # ── Step 3: ML Agent ──────────────────────────────────
        print("\n[3/4] Launching ML Agent...")
        ml_agent = MLAgent(self.spark)
        self.ml_results = ml_agent.run(
            silver_table, target_column, gold_table, categorical_cols
        )

        # ── Step 4: Summary ───────────────────────────────────
        print("\n[4/4] Pipeline Summary")
        print("╔" + "═" * 58 + "╗")
        print("║                  PIPELINE COMPLETE                       ║")
        print("╚" + "═" * 58 + "╝")

        print(f"\nEDA: {self.eda_results['data_quality']['total_records']} records analyzed")
        print(f"Diagnostic: {len(self.diagnostic_results['correlations'])} correlations found")
        if self.diagnostic_results['correlations']:
            print(f"  Top driver: {self.diagnostic_results['correlations'][0]['column']}")
        print(f"ML: Best model = {self.ml_results['best_model']['best_model']}")
        print(f"  R2 = {self.ml_results['best_model']['r2']}, RMSE = {self.ml_results['best_model']['rmse']}")
        print(f"Predictions saved to: {self.ml_results['prediction_table']}")

        return {
            "eda": self.eda_results,
            "diagnostic": self.diagnostic_results,
            "ml": self.ml_results,
        }

# COMMAND ----------

# DBTITLE 1,Execute Pipeline (commented)
# ── Execute Pipeline ──────────────────────────────────────
master = MasterAgent(spark)
results = master.run(
    bronze_table="lifeexpectancy.brone_lifeexpectancy.life_expectancy",
    target_column="Life_expectancy",
    silver_table="lifeexpectancy.silver.life_expectancy",
    gold_table="lifeexpectancy.silver.life_expectancy_ml_predictions",
    categorical_cols=None,
)

# COMMAND ----------

# DBTITLE 1,Persist Agent Results as Delta Tables
# ── Persist EDA & Diagnostic Results as Delta Tables for PBI ──
# Converts the in-memory `results` dict into queryable Delta tables
# Uses CREATE OR REPLACE TABLE to safely refresh results each run

from pyspark.sql.types import *

# Helper: save a list of dicts as a Delta table via CREATE OR REPLACE
def save_as_table(data_rows, table_name):
    """Create or replace a Delta table from a list of Python dicts."""
    if not data_rows:
        print(f"Skipped (no data): {table_name}")
        return
    df = spark.createDataFrame(data_rows)
    view_name = f"_tmp_persist_{table_name.replace('.', '_')}"
    df.createOrReplaceTempView(view_name)
    spark.sql(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}")
    spark.sql(f"DROP VIEW IF EXISTS {view_name}")
    print(f"Saved: {table_name}")

# ── EDA Tables ──────────────────────────────────────────────

# 1. Data Quality
dq = results["eda"]["data_quality"]
save_as_table([dq], "lifeexpectancy.silver.eda_data_quality")

# 2. Null Analysis
nulls_data = [
    {"column": k, "null_count": v["null_count"], "null_percentage": v["null_percentage"]}
    for k, v in results["eda"]["nulls"].items()
]
save_as_table(nulls_data, "lifeexpectancy.silver.eda_nulls")

# 3. Outlier Analysis
outliers_data = [
    {"column": k, **v} for k, v in results["eda"]["outliers"].items()
]
save_as_table(outliers_data, "lifeexpectancy.silver.eda_outliers")

# 4. Anomaly Analysis
anomalies_data = [
    {"column": k, **v} for k, v in results["eda"]["anomalies"].items()
]
save_as_table(anomalies_data, "lifeexpectancy.silver.eda_anomalies")

# ── Diagnostic Tables ───────────────────────────────────────

# 5. Correlations
save_as_table(results["diagnostic"]["correlations"], "lifeexpectancy.silver.diag_correlations")

# 6. Root Causes
save_as_table(results["diagnostic"]["root_causes"], "lifeexpectancy.silver.diag_root_causes")

# 7. Driver Analysis (cast all numerics to float to avoid type mismatch)
drivers_data = [
    {
        "feature": d["feature"],
        "min": float(d["min"]) if d["min"] is not None else None,
        "max": float(d["max"]) if d["max"] is not None else None,
        "mean": float(d["mean"]) if d["mean"] is not None else None,
        "stddev": float(d["stddev"]) if d["stddev"] is not None else None,
        "correlation_with_target": float(d["correlation_with_target"]) if d["correlation_with_target"] is not None else None,
    }
    for d in results["diagnostic"]["drivers"]
]
save_as_table(drivers_data, "lifeexpectancy.silver.diag_drivers")

# 8. Insights
insights_data = [
    {"id": i + 1, "insight": text}
    for i, text in enumerate(results["diagnostic"]["insights"])
]
save_as_table(insights_data, "lifeexpectancy.silver.diag_insights")

# ── ML Model Results ────────────────────────────────────────

# 9. Model Comparison
model_data = [
    {"model": name, "r2": metrics["r2"], "rmse": metrics["rmse"]}
    for name, metrics in results["ml"]["model_results"].items()
]
save_as_table(model_data, "lifeexpectancy.silver.ml_model_comparison")

print("\n" + "=" * 60)
print("All EDA & Diagnostic results persisted to Delta tables!")
print("=" * 60)
print("\nTables for Power BI:")
print("  EDA:")
print("    • lifeexpectancy.silver.eda_data_quality")
print("    • lifeexpectancy.silver.eda_nulls")
print("    • lifeexpectancy.silver.eda_outliers")
print("    • lifeexpectancy.silver.eda_anomalies")
print("  Diagnostic:")
print("    • lifeexpectancy.silver.diag_correlations")
print("    • lifeexpectancy.silver.diag_root_causes")
print("    • lifeexpectancy.silver.diag_drivers")
print("    • lifeexpectancy.silver.diag_insights")
print("  ML:")
print("    • lifeexpectancy.silver.ml_model_comparison")
print("    • lifeexpectancy.silver.life_expectancy_ml_predictions (predictions)")