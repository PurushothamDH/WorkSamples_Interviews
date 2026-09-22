# Databricks notebook source
# DBTITLE 1,EDAAgent Overview
# MAGIC %md
# MAGIC # EDAAgent — Exploratory Data Analysis
# MAGIC
# MAGIC Agent 1 in the pipeline. Runs four sub-analyses on the Bronze dataset:
# MAGIC
# MAGIC | Sub-Analysis | Method | Output |
# MAGIC |---|---|---|
# MAGIC | **Data Quality** | Record / column / duplicate counts | Summary dict |
# MAGIC | **Null Analysis** | Per-column null count & % | Per-column breakdown |
# MAGIC | **Outlier Analysis** | IQR method (Q1-Q3 ± 1.5×IQR) | Bounds + outlier count |
# MAGIC | **Anomaly Analysis** | Z-score method (|z| > threshold) | Mean, std, anomaly count |
# MAGIC
# MAGIC **Flow:** `MasterAgent → EDAAgent → DiagnosticAgent`

# COMMAND ----------

# DBTITLE 1,EDAAgent Class
from pyspark.sql.functions import col, mean as spark_mean, stddev as spark_stddev, abs as spark_abs


class EDAAgent:
    """Enhanced EDA Agent: Data Quality, Null Analysis, Outlier Analysis, Anomaly Analysis."""

    def __init__(self, spark):
        self.spark = spark

    # ── Data Quality ───────────────────────────────────────────
    def data_quality_checks(self, df):
        """Run comprehensive data quality checks."""
        record_count = df.count()
        col_count = len(df.columns)
        duplicate_count = record_count - df.distinct().count()

        return {
            "total_records": record_count,
            "total_columns": col_count,
            "duplicate_records": duplicate_count,
            "duplicate_percentage": round((duplicate_count / record_count) * 100, 2) if record_count > 0 else 0,
        }

    # ── Null Analysis ─────────────────────────────────────────
    def null_analysis(self, df):
        """Analyze null values across all columns."""
        total = df.count()
        null_summary = {}

        for c in df.columns:
            null_count = df.filter(col(c).isNull()).count()
            null_summary[c] = {
                "null_count": null_count,
                "null_percentage": round((null_count / total) * 100, 2) if total > 0 else 0,
            }

        return null_summary

    # ── Outlier Analysis (IQR) ─────────────────────────────────
    def outlier_analysis(self, df, numeric_cols=None):
        """Detect outliers using the IQR method for numeric columns."""
        if numeric_cols is None:
            numeric_cols = [
                c for c, d in df.dtypes
                if d in ("int", "bigint", "double", "float", "decimal")
            ]

        outlier_results = {}
        for c in numeric_cols:
            stats = df.approxQuantile(c, [0.25, 0.75], 0.05)
            q1, q3 = stats[0], stats[1]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = df.filter((col(c) < lower_bound) | (col(c) > upper_bound)).count()

            outlier_results[c] = {
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outlier_count": outlier_count,
            }

        return outlier_results

    # ── Anomaly Analysis (Z-score) ────────────────────────────
    def anomaly_analysis(self, df, numeric_cols=None, threshold=3.0):
        """Detect anomalies using the Z-score method for numeric columns."""
        if numeric_cols is None:
            numeric_cols = [
                c for c, d in df.dtypes
                if d in ("int", "bigint", "double", "float", "decimal")
            ]

        anomaly_results = {}
        for c in numeric_cols:
            stats = df.select(spark_mean(c).alias("mean"), spark_stddev(c).alias("std")).collect()[0]
            mean_val = stats["mean"] or 0
            std_val = stats["std"] or 1

            if std_val == 0:
                anomaly_results[c] = {"anomaly_count": 0, "method": "z-score", "threshold": threshold}
                continue

            anomaly_count = df.filter(spark_abs((col(c) - mean_val) / std_val) > threshold).count()

            anomaly_results[c] = {
                "mean": round(mean_val, 4),
                "stddev": round(std_val, 4),
                "anomaly_count": anomaly_count,
                "threshold": threshold,
                "method": "z-score",
            }

        return anomaly_results

    # ── Profile Dataset (full report) ──────────────────────────
    def profile_dataset(self, table_name):
        """Run full EDA profile on a table."""
        df = self.spark.table(table_name)

        print("=" * 60)
        print(f"EDA Report for: {table_name}")
        print("=" * 60)

        print("\n--- Data Quality ---")
        dq = self.data_quality_checks(df)
        print(f"Total Records: {dq['total_records']}")
        print(f"Total Columns: {dq['total_columns']}")
        print(f"Duplicate Records: {dq['duplicate_records']} ({dq['duplicate_percentage']}%)")

        print("\n--- Null Analysis ---")
        nulls = self.null_analysis(df)
        for col_name, info in nulls.items():
            if info["null_count"] > 0:
                print(f"  {col_name}: {info['null_count']} nulls ({info['null_percentage']}%)")

        print("\n--- Outlier Analysis (IQR) ---")
        outliers = self.outlier_analysis(df)
        for col_name, info in outliers.items():
            if info["outlier_count"] > 0:
                print(f"  {col_name}: {info['outlier_count']} outliers (bounds: {info['lower_bound']} to {info['upper_bound']})")

        print("\n--- Anomaly Analysis (Z-score) ---")
        anomalies = self.anomaly_analysis(df)
        for col_name, info in anomalies.items():
            if info["anomaly_count"] > 0:
                print(f"  {col_name}: {info['anomaly_count']} anomalies (mean={info['mean']}, std={info['stddev']})")

        print("\n" + "=" * 60)
        print("EDA Agent Completed")
        print("=" * 60)

        return {"data_quality": dq, "nulls": nulls, "outliers": outliers, "anomalies": anomalies}

    def run(self, table_name):
        """Main entry point for EDA Agent."""
        return self.profile_dataset(table_name)

# COMMAND ----------

