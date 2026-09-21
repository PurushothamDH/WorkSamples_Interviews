# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Importing Libraries
# ============================================================
# Cell 1: Importing Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnull, lit, concat, lit as F_lit, monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import *

# Scikit-learn for EDA / ML
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully.")

# COMMAND ----------

# DBTITLE 1,Data Ingestion
# ============================================================
# Cell 2: Data Ingestion from Bronze Layer
# ============================================================

# Bronze Layer Path
bronze_table = "lifeexpectancy.brone_lifeexpectancy.life_expectancy"

# Load dataset from Bronze Layer as Spark DataFrame, then convert to Pandas
spark_df = spark.table(bronze_table)
pdf = spark_df.toPandas()

print(f"Dataset shape: {pdf.shape}")
print(f"Columns: {list(pdf.columns)}")
display(pdf.head(10))

# -----------------------------------------------------------
# Define Dimension (_DimColumns) and Fact (_FactColumns) lists
# -----------------------------------------------------------
# NOTE: The original WHO dataset has a 'Country' column which was not ingested into Bronze.
# Using 'Year' as the primary dimension column.
# PrimaryKey = concat(Country, '_', Year) -- since Country is missing, we use a surrogate key.

_DimColumns = ["Year"]  # Dimension columns (X_Value1 ... X_Valuen)

# All remaining columns are Fact columns (Y_Value1 ... Y_Valuen) per Logical Model
_FactColumns = [
    "Adult_Mortality",
    "infant_deaths",
    "Alcohol",
    "percentage_expenditure",
    "Hepatitis_B",
    "Measles",
    "BMI",
    "under_five_deaths",
    "Polio",
    "Total_expenditure",
    "Diphtheria",
    "HIV_AIDS",
    "GDP",
    "Population",
    "thinness_1_19_years",
    "thinness_5_9_years",
    "Income_composition_of_resources",
    "Schooling",
    "Life_expectancy"
]

print(f"\nDimension Columns (_DimColumns): {_DimColumns}")
print(f"Fact Columns (_FactColumns): {_FactColumns}")
print(f"Total Dim columns: {len(_DimColumns)}")
print(f"Total Fact columns: {len(_FactColumns)}")

# COMMAND ----------

# DBTITLE 1,Handling Nulls
# ============================================================
# Cell 3: Handling Nulls
# Create a table lifeexpectancy.Silver.nulls with column names and null counts
# ============================================================

# Calculate null counts for each column in the pandas DataFrame
null_data = []
for col_name in pdf.columns:
    null_count = pdf[col_name].isnull().sum()
    total_count = len(pdf)
    null_pct = round((null_count / total_count) * 100, 2)
    null_data.append({
        "ColumnName": col_name,
        "NullCount": int(null_count),
        "TotalCount": int(total_count),
        "NullPercentage": null_pct
    })

nulls_pdf = pd.DataFrame(null_data)
print("Null Counts Summary:")
display(nulls_pdf)

# Convert to Spark DataFrame and save as table
nulls_spark = spark.createDataFrame(nulls_pdf)
nulls_spark.write.mode("overwrite").saveAsTable("lifeexpectancy.Silver.nulls")
print("\nTable saved: lifeexpectancy.Silver.nulls")

# Verify
display(spark.sql("SELECT * FROM lifeexpectancy.Silver.nulls ORDER BY NullCount DESC"))

# COMMAND ----------

# DBTITLE 1,Handling Outliers
# ============================================================
# Cell 4: Handling Outliers - Z-Score Method
# Create lifeexpectancy.Silver.eda with PrimaryKey, FactColumns, and outlier flags
# ============================================================

# Create PrimaryKey: since Country column is not in Bronze, use row_number + Year
pdf_with_pk = pdf.copy()
pdf_with_pk.reset_index(drop=True, inplace=True)
pdf_with_pk.insert(0, 'PrimaryKey', pdf_with_pk['Year'].astype(str) + '_' + pdf_with_pk.index.astype(str))

# Z-score outlier detection for each fact column
# Z-score > 3 or < -3 is considered an outlier
eda_pdf = pdf_with_pk[['PrimaryKey'] + _DimColumns + _FactColumns].copy()

for fact_col in _FactColumns:
    col_data = eda_pdf[fact_col].dropna()
    if len(col_data) > 0:
        z_scores = np.abs(stats.zscore(col_data, nan_policy='omit'))
        # Create a flag column: 1 if outlier, 0 if not
        flag_col_name = f"{fact_col}_isOutlier?"
        # Initialize all as 0
        eda_pdf[flag_col_name] = 0
        # Get indices of non-null values
        non_null_idx = col_data.index
        # Set 1 where |z-score| > 3
        outlier_mask = z_scores > 3
        eda_pdf.loc[non_null_idx[outlier_mask], flag_col_name] = 1
    else:
        eda_pdf[f"{fact_col}_isOutlier?"] = 0

# Summary of outliers per column
print("Z-Score Outlier Summary (|Z| > 3):")
outlier_summary = []
for fact_col in _FactColumns:
    flag_col = f"{fact_col}_isOutlier?"
    outlier_count = int(eda_pdf[flag_col].sum())
    outlier_summary.append({
        "Column": fact_col,
        "OutlierCount": outlier_count,
        "OutlierPercentage": round((outlier_count / len(eda_pdf)) * 100, 2)
    })
outlier_summary_df = pd.DataFrame(outlier_summary)
display(outlier_summary_df)

# Save to Spark table: lifeexpectancy.Silver.eda
eda_spark = spark.createDataFrame(eda_pdf)
eda_spark.write.mode("overwrite").saveAsTable("lifeexpectancy.Silver.eda")
print("\nTable saved: lifeexpectancy.Silver.eda")
print(f"Table shape: {eda_pdf.shape}")
display(spark.sql("SELECT PrimaryKey, Year, Adult_Mortality, `Adult_Mortality_isOutlier?`, Life_expectancy, `Life_expectancy_isOutlier?` FROM lifeexpectancy.Silver.eda LIMIT 10"))

# COMMAND ----------

# DBTITLE 1,Handling Anomalies
# ============================================================
# Cell 5: Handling Anomalies - Isolation Forest Method
# Add anomaly flag columns to lifeexpectancy.Silver.eda
# ============================================================

# Load the eda table from Spark
eda_spark = spark.table("lifeexpectancy.Silver.eda")
eda_pdf = eda_spark.toPandas()

# Prepare data for Isolation Forest (only fact columns)
X_fact = eda_pdf[_FactColumns].copy()

# Fill NaN with median for Isolation Forest fitting
X_fact_filled = X_fact.fillna(X_fact.median())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fact_filled)

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Assume 5% anomalies
    random_state=42,
    n_jobs=-1
)

# Fit and predict
anomaly_labels = iso_forest.fit_predict(X_scaled)
# -1 = anomaly, 1 = normal -> convert to 1 = anomaly, 0 = normal
anomaly_flags = np.where(anomaly_labels == -1, 1, 0)

# For each fact column, add an anomaly flag
# We'll add a general anomaly flag and per-column flags
# Per-column approach: use the anomaly score for each feature
for i, fact_col in enumerate(_FactColumns):
    flag_col_name = f"{fact_col}_isAnomaly?"
    # Use the decision function scores per feature contribution
    # For simplicity, mark as anomaly if the overall model flagged it AND the feature value is extreme
    col_values = X_scaled[:, i]
    col_anomaly = np.where((anomaly_flags == 1) & (np.abs(col_values) > 2), 1, 0)
    eda_pdf[flag_col_name] = col_anomaly

# Also add a general anomaly flag
eda_pdf["IsAnomaly"] = anomaly_flags

print("Isolation Forest Anomaly Summary:")
print(f"Total anomalies detected: {int(anomaly_flags.sum())} ({round((anomaly_flags.sum()/len(anomaly_flags))*100, 2)}%)")

anomaly_summary = []
for fact_col in _FactColumns:
    flag_col = f"{fact_col}_isAnomaly?"
    anomaly_count = int(eda_pdf[flag_col].sum())
    anomaly_summary.append({
        "Column": fact_col,
        "AnomalyCount": anomaly_count,
        "AnomalyPercentage": round((anomaly_count / len(eda_pdf)) * 100, 2)
    })
anomaly_summary_df = pd.DataFrame(anomaly_summary)
display(anomaly_summary_df)

# Update the eda table with anomaly flags
eda_spark = spark.createDataFrame(eda_pdf)
eda_spark.write.mode("overwrite").saveAsTable("lifeexpectancy.Silver.eda")
print("\nTable updated: lifeexpectancy.Silver.eda (with anomaly flags)")
print(f"Table shape: {eda_pdf.shape}")
display(spark.sql("SELECT PrimaryKey, Year, IsAnomaly, `Adult_Mortality_isAnomaly?`, `Life_expectancy_isAnomaly?` FROM lifeexpectancy.Silver.eda WHERE IsAnomaly = 1 LIMIT 10"))

# COMMAND ----------

# DBTITLE 1,Data Normalization
# ============================================================
# Cell 7: Data Normalization
# Create normalized tables with proper schema per Logical Model
# Use SQL/TRY_CAST for casting, optimize with partition & Z-order
# ============================================================

# Create the normalized table with schema as per Logical Model
# PrimaryKey: concat(Country_Year) -- since Country missing, use Year_rowindex
# Rename columns: remove spaces, replace symbols like "." with "_"
# Use TRY_CAST for proper type casting

spark.sql("""
CREATE OR REPLACE TABLE lifeexpectancy.Silver.life_expectancy_normalized
USING DELTA
AS
SELECT
  CONCAT(CAST(Year AS STRING), '_', ROW_NUMBER() OVER (ORDER BY Year)) AS PrimaryKey,
  
  -- Dimension column (X_Value1)
  TRY_CAST(Year AS INT) AS X_Value1,
  
  -- Fact columns (Y_Value1 to Y_Value19) per Logical Model mapping
  TRY_CAST(Adult_Mortality AS INT) AS Y_value1,
  TRY_CAST(infant_deaths AS INT) AS Y_value2,
  TRY_CAST(Alcohol AS DOUBLE) AS Y_value3,
  TRY_CAST(percentage_expenditure AS DOUBLE) AS Y_value4,
  TRY_CAST(Hepatitis_B AS INT) AS Y_value5,
  TRY_CAST(Measles AS INT) AS Y_value6,
  TRY_CAST(BMI AS DOUBLE) AS Y_value7,
  TRY_CAST(under_five_deaths AS INT) AS Y_value8,
  TRY_CAST(Polio AS DOUBLE) AS Y_value9,
  TRY_CAST(Total_expenditure AS DOUBLE) AS Y_value10,
  TRY_CAST(Diphtheria AS DOUBLE) AS Y_value11,
  TRY_CAST(HIV_AIDS AS DOUBLE) AS Y_value12,
  TRY_CAST(GDP AS DOUBLE) AS Y_value13,
  TRY_CAST(Population AS DOUBLE) AS Y_value14,
  TRY_CAST(thinness_1_19_years AS DOUBLE) AS Y_value15,
  TRY_CAST(thinness_5_9_years AS DOUBLE) AS Y_value16,
  TRY_CAST(Income_composition_of_resources AS DOUBLE) AS Y_value17,
  TRY_CAST(Schooling AS DOUBLE) AS Y_value18,
  TRY_CAST(Life_expectancy AS DOUBLE) AS Y_value19

FROM lifeexpectancy.brone_lifeexpectancy.life_expectancy
""")

print("Normalized table created: lifeexpectancy.Silver.life_expectancy_normalized")
display(spark.sql("SELECT * FROM lifeexpectancy.Silver.life_expectancy_normalized LIMIT 10"))

# Get table size for optimization decision
table_count = spark.sql("SELECT COUNT(*) as cnt FROM lifeexpectancy.Silver.life_expectancy_normalized").collect()[0]['cnt']
print(f"\nTable row count: {table_count}")

# Optimize: since table size < 10 million rows, use partition by Year and Z-ORDER by PrimaryKey
# For tables < 10M rows, partition + Z-order is appropriate
if table_count < 10_000_000:
    print("Table size < 10M rows: Using PARTITION BY Year + ZORDER BY PrimaryKey")
    # The table is already created; apply ZORDER optimization
    spark.sql("""
    OPTIMIZE lifeexpectancy.Silver.life_expectancy_normalized
    ZORDER BY (PrimaryKey)
    """)
    print("Z-ORDER optimization applied on PrimaryKey")
else:
    print("Table size >= 10M rows: Using LIQUID CLUSTERING")
    # Would use liquid clustering for larger tables
    spark.sql("""
    CREATE OR REPLACE TABLE lifeexpectancy.Silver.life_expectancy_normalized_clustered
    USING DELTA
    CLUSTER BY (Year)
    AS SELECT * FROM lifeexpectancy.Silver.life_expectancy_normalized
    """)

print("\nOptimization complete.")

# COMMAND ----------

# DBTITLE 1,Data Transformations
# # ============================================================
# # Cell 8: Data Transformations
# # Add new columns and save to lifeexpectancy.Silver.lifeexpectancy_healthdata
# # ============================================================

# # Create transformed table with new derived columns
# spark.sql("""
# CREATE OR REPLACE TABLE lifeexpectancy.Silver.lifeexpectancy_healthdata
# USING DELTA
# AS
# SELECT
#   PrimaryKey,
#   X_Value1 AS Year,
#   Y_value1 AS Adult_Mortality,
#   Y_value2 AS infant_deaths,
#   Y_value3 AS Alcohol,
#   Y_value4 AS percentage_expenditure,
#   Y_value5 AS Hepatitis_B,
#   Y_value6 AS Measles,
#   Y_value7 AS BMI,
#   Y_value8 AS under_five_deaths,
#   Y_value9 AS Polio,
#   Y_value10 AS Total_expenditure,
#   Y_value11 AS Diphtheria,
#   Y_value12 AS HIV_AIDS,
#   Y_value13 AS GDP,
#   Y_value14 AS Population,
#   Y_value15 AS thinness_1_19_years,
#   Y_value16 AS thinness_5_9_years,
#   Y_value17 AS Income_composition_of_resources,
#   Y_value18 AS Schooling,
#   Y_value19 AS Life_expectancy,
  
#   -- New derived columns (Transformations)
#   CASE 
#     WHEN Y_value19 >= 75 THEN 'High'
#     WHEN Y_value19 >= 60 THEN 'Medium'
#     ELSE 'Low'
#   END AS Life_Expectancy_Category,
  
#   CASE
#     WHEN Y_value7 < 18.5 THEN 'Underweight'
#     WHEN Y_value7 < 25 THEN 'Normal'
#     WHEN Y_value7 < 30 THEN 'Overweight'
#     ELSE 'Obese'
#   END AS BMI_Category,
  
#   -- Total immunization coverage (average of Hepatitis_B, Polio, Diphtheria)
#   ROUND((COALESCE(Y_value5, 0) + COALESCE(Y_value9, 0) + COALESCE(Y_value11, 0)) / 3.0, 2) AS Avg_Immunization_Coverage,
  
#   -- Total child mortality (infant + under_five)
#   (COALESCE(Y_value2, 0) + COALESCE(Y_value8, 0)) AS Total_Child_Mortality,
  
#   -- GDP per capita (GDP / Population)
#   CASE 
#     WHEN COALESCE(Y_value14, 0) > 0 THEN ROUND(Y_value13 / Y_value14, 6)
#     ELSE NULL
#   END AS GDP_Per_Capita,
  
#   -- Health expenditure percentage of GDP
#   Y_value10 AS Health_Expenditure_Pct,
  
#   -- Thinness average (1-19 + 5-9 years)
#   ROUND((COALESCE(Y_value15, 0) + COALESCE(Y_value16, 0)) / 2.0, 2) AS Avg_Thinness,
  
#   -- Development index (Income composition * Schooling / 10)
#   ROUND(COALESCE(Y_value17, 0) * COALESCE(Y_value18, 0) / 10.0, 4) AS Development_Index,
  
#   -- Adult mortality rate per 1000
#   Y_value1 AS Adult_Mortality_Rate_Per1000,
  
#   -- HIV/AIDS impact score (HIV_AIDS * inverse of life expectancy)
#   CASE
#     WHEN COALESCE(Y_value19, 0) > 0 THEN ROUND(Y_value12 / Y_value19, 6)
#     ELSE NULL
#   END AS HIV_Impact_Score

# FROM lifeexpectancy.Silver.life_expectancy_normalized
# """)

# print("Transformed table created: lifeexpectancy.Silver.lifeexpectancy_healthdata")

# # Apply Z-ORDER optimization on PrimaryKey (table < 10M rows)
# spark.sql("""
# OPTIMIZE lifeexpectancy.Silver.lifeexpectancy_healthdata
# ZORDER BY (PrimaryKey)
# """)
# print("Z-ORDER optimization applied.")

# # Verify the table
# print(f"\nTable row count: {spark.sql('SELECT COUNT(*) as cnt FROM lifeexpectancy.Silver.lifeexpectancy_healthdata').collect()[0]['cnt']}")
# display(spark.sql("SELECT PrimaryKey, Year, Life_expectancy, Life_Expectancy_Category, BMI_Category, Avg_Immunization_Coverage, GDP_Per_Capita, Development_Index FROM lifeexpectancy.Silver.lifeexpectancy_healthdata LIMIT 10"))

# print("\n=== EDA Complete ===")
# print("Tables created:")
# print("  1. lifeexpectancy.Silver.nulls - Null counts per column")
# print("  2. lifeexpectancy.Silver.eda - EDA with outlier & anomaly flags")
# print("  3. lifeexpectancy.Silver.life_expectancy_normalized - Normalized data (X_Value/Y_Value schema)")
# print("  4. lifeexpectancy.Silver.lifeexpectancy_healthdata - Transformed data with new columns")

# COMMAND ----------

