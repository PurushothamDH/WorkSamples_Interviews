# Databricks notebook source
# DBTITLE 1,MLAgent Overview
# MAGIC %md
# MAGIC # MLAgent — Feature Engineering, Training & Prediction
# MAGIC
# MAGIC Agent 3 in the pipeline. Builds and trains ML models on the Silver dataset:
# MAGIC
# MAGIC | Step | Description | Output |
# MAGIC |---|---|---|
# MAGIC | **Feature Engineering** | StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler | Pipeline stages + feature list |
# MAGIC | **Train Models** | LinearRegression, RandomForest, DecisionTree, GBT | R² & RMSE per model |
# MAGIC | **Select Best Model** | Pick highest R² | Best model name + metrics |
# MAGIC | **Prediction Table** | Generate predictions & save as Delta table | Saved prediction table |
# MAGIC
# MAGIC **Flow:** `DiagnosticAgent → MLAgent → Save Delta Tables`

# COMMAND ----------

# DBTITLE 1,MLAgent Class
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline


class MLAgent:
    """ML Agent: Feature Engineering, Train Models, Select Best Model, Prediction Table."""

    def __init__(self, spark):
        self.spark = spark
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None

    # ── Feature Engineering ───────────────────────────────────
    def feature_engineering(self, df, target_column, categorical_cols=None):
        """Build feature pipeline with encoding and scaling."""
        stages = []
        processed_cols = []

        # Handle categorical columns
        if categorical_cols:
            for cat_col in categorical_cols:
                indexer = StringIndexer(
                    inputCol=cat_col,
                    outputCol=f"{cat_col}_indexed",
                    handleInvalid="keep",
                )
                encoder = OneHotEncoder(
                    inputCol=f"{cat_col}_indexed",
                    outputCol=f"{cat_col}_encoded",
                )
                stages.extend([indexer, encoder])
                processed_cols.append(f"{cat_col}_encoded")

        # Numeric columns (excluding target)
        numeric_cols = [
            c for c, d in df.dtypes
            if d in ("int", "bigint", "double", "float", "decimal") and c != target_column
        ]
        processed_cols.extend(numeric_cols)

        # Assemble & scale features
        assembler = VectorAssembler(
            inputCols=processed_cols,
            outputCol="features_raw",
            handleInvalid="skip",
        )
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=True,
            withStd=True,
        )
        stages.extend([assembler, scaler])

        self.feature_columns = processed_cols
        print(f"Feature Engineering: {len(processed_cols)} features prepared")
        print(f"  Numeric: {numeric_cols}")
        if categorical_cols:
            print(f"  Categorical (encoded): {categorical_cols}")

        return stages, processed_cols

    # ── Train Models ──────────────────────────────────────────
    def train_models(self, df, target_column, feature_stages, train_ratio=0.8):
        """Train multiple models and select the best one."""
        train_data, test_data = df.randomSplit([train_ratio, 1 - train_ratio], seed=42)

        models = {
            "LinearRegression": LinearRegression(featuresCol="features", labelCol=target_column),
            "RandomForest": RandomForestRegressor(featuresCol="features", labelCol=target_column, numTrees=50, seed=42),
            "DecisionTree": DecisionTreeRegressor(featuresCol="features", labelCol=target_column, seed=42),
            "GradientBoostedTrees": GBTRegressor(featuresCol="features", labelCol=target_column, maxIter=50, seed=42),
        }

        evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2")
        rmse_evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")

        best_r2 = -999
        results = {}

        for name, model in models.items():
            pipeline = Pipeline(stages=feature_stages + [model])

            print(f"\nTraining {name}...")
            fitted_model = pipeline.fit(train_data)
            predictions = fitted_model.transform(test_data)

            r2 = evaluator.evaluate(predictions)
            rmse = rmse_evaluator.evaluate(predictions)

            results[name] = {"r2": round(r2, 4), "rmse": round(rmse, 4)}
            print(f"  {name} -> R2: {round(r2, 4)}, RMSE: {round(rmse, 4)}")

            if r2 > best_r2:
                best_r2 = r2
                self.best_model = fitted_model
                self.best_model_name = name

        print(f"\nBest Model: {self.best_model_name} (R2={round(best_r2, 4)})")
        return results

    # ── Select Best Model ─────────────────────────────────────
    def select_best_model(self, results):
        """Return the best model name and its metrics."""
        best_name = max(results, key=lambda k: results[k]["r2"])
        return {
            "best_model": best_name,
            "r2": results[best_name]["r2"],
            "rmse": results[best_name]["rmse"],
        }

    # ── Prediction Table ─────────────────────────────────────
    def create_prediction_table(self, df, target_column, output_table):
        """Generate predictions and save as Delta table."""
        if self.best_model is None:
            raise ValueError("No model trained. Call train_models first.")

        predictions = self.best_model.transform(df)

        pred_cols = self.feature_columns + [target_column, "prediction"]
        pred_df = predictions.select(*pred_cols)

        pred_df.write.mode("overwrite").saveAsTable(output_table)
        print(f"Prediction table saved: {output_table}")

        return pred_df

    # ── Full ML Pipeline Run ──────────────────────────────────
    def run(self, table_name, target_column, output_table, categorical_cols=None):
        """Main entry point for ML Agent."""
        df = self.spark.table(table_name)

        print("=" * 60)
        print(f"ML Pipeline for: {table_name}")
        print(f"Target: {target_column}")
        print("=" * 60)

        print("\n--- Feature Engineering ---")
        stages, features = self.feature_engineering(df, target_column, categorical_cols)

        print("\n--- Training Models ---")
        results = self.train_models(df, target_column, stages)

        print("\n--- Select Best Model ---")
        best = self.select_best_model(results)
        print(f"  Best: {best['best_model']} (R2={best['r2']}, RMSE={best['rmse']})")

        print("\n--- Prediction Table ---")
        pred_df = self.create_prediction_table(df, target_column, output_table)
        print(f"  Predictions saved to: {output_table}")
        print(f"  Record count: {pred_df.count()}")

        print("\n" + "=" * 60)
        print("ML Agent Completed")
        print("=" * 60)

        return {
            "model_results": results,
            "best_model": best,
            "prediction_table": output_table,
        }

# COMMAND ----------

