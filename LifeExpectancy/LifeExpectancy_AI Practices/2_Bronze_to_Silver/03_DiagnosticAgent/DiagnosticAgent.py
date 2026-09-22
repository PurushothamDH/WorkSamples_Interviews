# Databricks notebook source
# DBTITLE 1,DiagnosticAgent Overview
# MAGIC %md
# MAGIC # DiagnosticAgent — Correlation & Root Cause
# MAGIC
# MAGIC Agent 2 in the pipeline. Performs diagnostic analysis on the Silver dataset:
# MAGIC
# MAGIC | Sub-Analysis | Description | Output |
# MAGIC |---|---|---|
# MAGIC | **Correlation** | Pearson correlation of numeric features vs. target | Sorted list with strength labels |
# MAGIC | **Root Cause** | Top features driving target variance | Ranked causes with direction |
# MAGIC | **Driver Analysis** | Min/max/mean/std + correlation per feature | Feature stats dict |
# MAGIC | **Insights** | Human-readable summary of findings | List of insight strings |
# MAGIC
# MAGIC **Flow:** `EDAAgent → DiagnosticAgent → MLAgent`

# COMMAND ----------

# DBTITLE 1,DiagnosticAgent Class
from pyspark.sql.functions import col, min as spark_min, max as spark_max, mean as spark_mean, stddev as spark_stddev


class DiagnosticAgent:
    """Diagnostic Agent: Correlation, Root Cause, Driver Analysis, Insights."""

    def __init__(self, spark):
        self.spark = spark

    # ── Correlation Analysis ──────────────────────────────────
    def perform_correlation_analysis(self, df, target_column):
        """Compute Pearson correlation between numeric features and target."""
        numeric_cols = [
            c for c, d in df.dtypes
            if d in ("int", "bigint", "double", "float", "decimal")
        ]
        results = []

        for c in numeric_cols:
            if c != target_column:
                try:
                    corr_val = df.stat.corr(c, target_column)
                    results.append({
                        "column": c,
                        "correlation": round(corr_val, 4),
                        "strength": self._corr_strength(corr_val),
                    })
                except Exception:
                    results.append({"column": c, "correlation": None, "strength": "N/A"})

        results.sort(key=lambda x: abs(x["correlation"]) if x["correlation"] is not None else 0, reverse=True)
        return results

    def _corr_strength(self, val):
        if val is None:
            return "N/A"
        av = abs(val)
        if av >= 0.7:
            return "Strong"
        elif av >= 0.4:
            return "Moderate"
        elif av >= 0.2:
            return "Weak"
        else:
            return "Negligible"

    # ── Root Cause Analysis ───────────────────────────────────
    def root_cause_analysis(self, df, target_column, top_n=5):
        """Identify top features contributing to target variance."""
        correlations = self.perform_correlation_analysis(df, target_column)
        top_drivers = correlations[:top_n]

        root_causes = []
        for item in top_drivers:
            if item["correlation"] is not None and abs(item["correlation"]) >= 0.2:
                root_causes.append({
                    "feature": item["column"],
                    "correlation": item["correlation"],
                    "direction": "Positive" if item["correlation"] > 0 else "Negative",
                    "impact": item["strength"],
                })

        return root_causes

    # ── Driver Analysis ───────────────────────────────────────
    def driver_analysis(self, df, target_column):
        """Analyze key drivers affecting the target variable."""
        numeric_cols = [
            c for c, d in df.dtypes
            if d in ("int", "bigint", "double", "float", "decimal") and c != target_column
        ]

        drivers = []
        for c in numeric_cols:
            stats = df.select(
                spark_min(c).alias("min"),
                spark_max(c).alias("max"),
                spark_mean(c).alias("mean"),
                spark_stddev(c).alias("std"),
            ).collect()[0]

            corr_val = df.stat.corr(c, target_column)

            drivers.append({
                "feature": c,
                "min": round(stats["min"], 4) if stats["min"] else None,
                "max": round(stats["max"], 4) if stats["max"] else None,
                "mean": round(stats["mean"], 4) if stats["mean"] else None,
                "stddev": round(stats["std"], 4) if stats["std"] else None,
                "correlation_with_target": round(corr_val, 4) if corr_val else None,
            })

        return drivers

    # ── Insights ─────────────────────────────────────────────
    def generate_insights(self, correlations, root_causes, drivers):
        """Generate human-readable insights from analysis."""
        insights = []

        if correlations:
            top = correlations[0]
            insights.append(f"Strongest correlation: {top['column']} (r={top['correlation']}, {top['strength']})")

        if root_causes:
            cause_list = ", ".join([f"{rc['feature']} ({rc['direction']})" for rc in root_causes[:3]])
            insights.append(f"Top root causes: {cause_list}")

        high_var_drivers = [d for d in drivers if d["stddev"] and d["stddev"] > 0]
        if high_var_drivers:
            insights.append(f"High-variance drivers: {', '.join([d['feature'] for d in high_var_drivers[:3]])}")

        return insights

    # ── Full Diagnostic Run ───────────────────────────────────
    def run(self, table_name, target_column):
        """Main entry point for Diagnostic Agent."""
        df = self.spark.table(table_name)

        print("=" * 60)
        print(f"Diagnostic Report for: {table_name}")
        print(f"Target Variable: {target_column}")
        print("=" * 60)

        print("\n--- Correlation Analysis ---")
        correlations = self.perform_correlation_analysis(df, target_column)
        for item in correlations:
            print(f"  {item['column']}: r={item['correlation']} ({item['strength']})")

        print("\n--- Root Cause Analysis ---")
        root_causes = self.root_cause_analysis(df, target_column)
        for rc in root_causes:
            print(f"  {rc['feature']}: {rc['correlation']} ({rc['direction']}, {rc['impact']})")

        print("\n--- Driver Analysis ---")
        drivers = self.driver_analysis(df, target_column)
        for d in drivers:
            print(f"  {d['feature']}: mean={d['mean']}, std={d['stddev']}, corr={d['correlation_with_target']}")

        print("\n--- Insights ---")
        insights = self.generate_insights(correlations, root_causes, drivers)
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")

        print("\n" + "=" * 60)
        print("Diagnostic Agent Completed")
        print("=" * 60)

        return {
            "correlations": correlations,
            "root_causes": root_causes,
            "drivers": drivers,
            "insights": insights,
        }

# COMMAND ----------

