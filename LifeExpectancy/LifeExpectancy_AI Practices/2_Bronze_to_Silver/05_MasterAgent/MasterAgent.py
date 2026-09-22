# Databricks notebook source
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
# Uncomment and update the table names to run the full pipeline.
#
# master = MasterAgent(spark)
# results = master.run(
#     bronze_table="catalog.schema.bronze_lifeexpectancy",
#     target_column="life_expectancy",
#     silver_table="catalog.schema.silver_lifeexpectancy",
#     gold_table="catalog.schema.gold_lifeexpectancy_predictions",
#     categorical_cols=["country", "status"],
# )

# COMMAND ----------

