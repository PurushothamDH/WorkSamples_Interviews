# Databricks notebook source
# DBTITLE 1,Import All Libraries
# ============================================================
# Cell 1: Import All Libraries
# ============================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.preprocessing import (
    PolynomialFeatures, StandardScaler, LabelEncoder
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# XGBoost and LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("xgboost not installed - will skip XGBoost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("lightgbm not installed - will skip LightGBM")

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit, row_number
from pyspark.sql.window import Window

print("All libraries imported successfully.")
print(f"XGBoost available: {XGB_AVAILABLE}")
print(f"LightGBM available: {LGB_AVAILABLE}")

# COMMAND ----------

# DBTITLE 1,Load Dataset
# ============================================================
# Cell 2: Load Dataset as df
# ============================================================

# Load from Bronze Layer
bronze_table = "lifeexpectancy.brone_lifeexpectancy.life_expectancy"
spark_df = spark.table(bronze_table)
df = spark_df.toPandas()

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
display(df.head(10))
print(f"\nNull counts:\n{df.isnull().sum()}")

# COMMAND ----------

# DBTITLE 1,Null Imputation with Median/Mode
# ============================================================
# Cell 3: Replace Nulls - Median for Numeric, Mode for String
# ============================================================

# Identify numeric (fact) and string columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

print(f"Numeric columns: {numeric_cols}")
print(f"String columns: {string_cols}")

# Replace numeric column nulls/blanks with median values
for col_name in numeric_cols:
    median_val = df[col_name].median()
    null_count = df[col_name].isnull().sum()
    if null_count > 0:
        df[col_name] = df[col_name].fillna(median_val)
        print(f"  {col_name}: Filled {null_count} nulls with median={median_val}")

# Replace string column nulls/blanks with mode values
for col_name in string_cols:
    mode_val = df[col_name].mode().iloc[0] if len(df[col_name].mode()) > 0 else "Unknown"
    null_count = df[col_name].isnull().sum()
    blank_count = (df[col_name] == '').sum() if df[col_name].dtype == 'object' else 0
    total_replace = null_count + blank_count
    if total_replace > 0:
        df[col_name] = df[col_name].replace('', np.nan)
        df[col_name] = df[col_name].fillna(mode_val)
        print(f"  {col_name}: Filled {total_replace} nulls/blanks with mode='{mode_val}'")

print(f"\nNull counts after imputation:\n{df.isnull().sum()}")
print(f"\nTotal remaining nulls: {df.isnull().sum().sum()}")

# COMMAND ----------

# DBTITLE 1,Define X and Y Values
# ============================================================
# Cell 4: Define X and Y Values
# ============================================================

# Create PrimaryKey (since Country is not in Bronze, use Year + row index)
df.reset_index(drop=True, inplace=True)
df.insert(0, 'PrimaryKey', df['Year'].astype(str) + '_' + df.index.astype(str))

# Per Logical Model:
# X_Value1 = Year (Dimension)
# Y_values = all fact columns (features)
# Target = Life_expectancy (Y_value19 in the logical model)

# Define X (independent variables / features) and Y (dependent variable / target)
# X includes Year and all fact columns EXCEPT Life_expectancy (the target)
feature_cols = [c for c in df.columns if c not in ['PrimaryKey', 'Life_expectancy']]
target_col = 'Life_expectancy'

print(f"Feature columns (X): {feature_cols}")
print(f"Target column (Y): {target_col}")
print(f"Number of features: {len(feature_cols)}")

# Prepare X and y
X = df[feature_cols].copy()
y = df[target_col].copy()

# Encode any string columns if present (for safety)
for col_name in X.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    X[col_name] = le.fit_transform(X[col_name].astype(str))

# Ensure all values are numeric
X = X.astype(float)
y = y.astype(float)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nX columns mapped to X/Y values per Logical Model:")
print(f"  X_Value1 (Year): {feature_cols[0]}")
for i, col_name in enumerate(feature_cols[1:], 1):
    print(f"  Y_value{i}: {col_name}")
print(f"  Target (Y_value19): {target_col}")

# COMMAND ----------

# DBTITLE 1,Regression Models
# ============================================================
# Cell 5: Regression Models - Start with Multiple Linear Regression
# Continue with all remaining regression models until best RMSE/R2
# ============================================================

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results for all models
results = []
best_model = None
best_r2 = -np.inf
best_rmse = np.inf
best_predictions = None

# -------------------------------------------------------
# 1. Multiple Linear Regression
# -------------------------------------------------------
print("=" * 60)
print("1. Multiple Linear Regression")
print("=" * 60)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f"  RMSE: {rmse_lr:.4f}")
print(f"  R2:   {r2_lr:.4f}")
print(f"  MAE:  {mae_lr:.4f}")
results.append({"Model": "Multiple Linear Regression", "RMSE": rmse_lr, "R2": r2_lr, "MAE": mae_lr})
if r2_lr > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_lr, rmse_lr, "Multiple Linear Regression", y_pred_lr

# -------------------------------------------------------
# 2. Polynomial Regression (degree=2)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("2. Polynomial Regression (degree=2)")
print("=" * 60)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
r2_poly = r2_score(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
print(f"  RMSE: {rmse_poly:.4f}")
print(f"  R2:   {r2_poly:.4f}")
print(f"  MAE:  {mae_poly:.4f}")
results.append({"Model": "Polynomial Regression", "RMSE": rmse_poly, "R2": r2_poly, "MAE": mae_poly})
if r2_poly > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_poly, rmse_poly, "Polynomial Regression", y_pred_poly

# -------------------------------------------------------
# 3. Ridge Regression
# -------------------------------------------------------
print("\n" + "=" * 60)
print("3. Ridge Regression")
print("=" * 60)
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
print(f"  RMSE: {rmse_ridge:.4f}")
print(f"  R2:   {r2_ridge:.4f}")
print(f"  MAE:  {mae_ridge:.4f}")
results.append({"Model": "Ridge Regression", "RMSE": rmse_ridge, "R2": r2_ridge, "MAE": mae_ridge})
if r2_ridge > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_ridge, rmse_ridge, "Ridge Regression", y_pred_ridge

# -------------------------------------------------------
# 4. Lasso Regression
# -------------------------------------------------------
print("\n" + "=" * 60)
print("4. Lasso Regression")
print("=" * 60)
lasso_model = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
print(f"  RMSE: {rmse_lasso:.4f}")
print(f"  R2:   {r2_lasso:.4f}")
print(f"  MAE:  {mae_lasso:.4f}")
results.append({"Model": "Lasso Regression", "RMSE": rmse_lasso, "R2": r2_lasso, "MAE": mae_lasso})
if r2_lasso > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_lasso, rmse_lasso, "Lasso Regression", y_pred_lasso

# -------------------------------------------------------
# 5. Elastic Net Regression
# -------------------------------------------------------
print("\n" + "=" * 60)
print("5. Elastic Net Regression")
print("=" * 60)
elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
elastic_model.fit(X_train_scaled, y_train)
y_pred_elastic = elastic_model.predict(X_test_scaled)
rmse_elastic = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
r2_elastic = r2_score(y_test, y_pred_elastic)
mae_elastic = mean_absolute_error(y_test, y_pred_elastic)
print(f"  RMSE: {rmse_elastic:.4f}")
print(f"  R2:   {r2_elastic:.4f}")
print(f"  MAE:  {mae_elastic:.4f}")
results.append({"Model": "Elastic Net Regression", "RMSE": rmse_elastic, "R2": r2_elastic, "MAE": mae_elastic})
if r2_elastic > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_elastic, rmse_elastic, "Elastic Net Regression", y_pred_elastic

# -------------------------------------------------------
# 6. Logistic Regression (adapted for regression via binning)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("6. Logistic Regression (Binned Regression)")
print("=" * 60)
# Logistic Regression is for classification; adapt by binning target into classes
# then predict the midpoint of each bin as a regression proxy
n_bins = 20
y_bins = pd.cut(y_train, bins=n_bins, labels=False)
log_model = LogisticRegression(max_iter=10000, random_state=42, multi_class='multinomial')
log_model.fit(X_train_scaled, y_bins)
y_pred_log_bins = log_model.predict(X_test_scaled)
# Convert bin predictions back to continuous values using bin midpoints
bin_edges = pd.cut(y_train, bins=n_bins).cat.categories
bin_midpoints = [(interval.left + interval.right) / 2 for interval in bin_edges]
y_pred_log = np.array([bin_midpoints[int(b)] for b in y_pred_log_bins])
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
r2_log = r2_score(y_test, y_pred_log)
mae_log = mean_absolute_error(y_test, y_pred_log)
print(f"  RMSE: {rmse_log:.4f}")
print(f"  R2:   {r2_log:.4f}")
print(f"  MAE:  {mae_log:.4f}")
results.append({"Model": "Logistic Regression (Binned)", "RMSE": rmse_log, "R2": r2_log, "MAE": mae_log})
if r2_log > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_log, rmse_log, "Logistic Regression (Binned)", y_pred_log

# -------------------------------------------------------
# 7. Decision Tree Regression
# -------------------------------------------------------
print("\n" + "=" * 60)
print("7. Decision Tree Regression")
print("=" * 60)
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
print(f"  RMSE: {rmse_dt:.4f}")
print(f"  R2:   {r2_dt:.4f}")
print(f"  MAE:  {mae_dt:.4f}")
results.append({"Model": "Decision Tree Regression", "RMSE": rmse_dt, "R2": r2_dt, "MAE": mae_dt})
if r2_dt > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_dt, rmse_dt, "Decision Tree Regression", y_pred_dt

# -------------------------------------------------------
# 8. Random Forest Regression
# -------------------------------------------------------
print("\n" + "=" * 60)
print("8. Random Forest Regression")
print("=" * 60)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"  RMSE: {rmse_rf:.4f}")
print(f"  R2:   {r2_rf:.4f}")
print(f"  MAE:  {mae_rf:.4f}")
results.append({"Model": "Random Forest Regression", "RMSE": rmse_rf, "R2": r2_rf, "MAE": mae_rf})
if r2_rf > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_rf, rmse_rf, "Random Forest Regression", y_pred_rf

# -------------------------------------------------------
# 9a. Gradient Boosting - XGBoost
# -------------------------------------------------------
print("\n" + "=" * 60)
print("9a. Gradient Boosting - XGBoost")
print("=" * 60)
if XGB_AVAILABLE:
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    print(f"  RMSE: {rmse_xgb:.4f}")
    print(f"  R2:   {r2_xgb:.4f}")
    print(f"  MAE:  {mae_xgb:.4f}")
    results.append({"Model": "XGBoost", "RMSE": rmse_xgb, "R2": r2_xgb, "MAE": mae_xgb})
    if r2_xgb > best_r2:
        best_r2, best_rmse, best_model, best_predictions = r2_xgb, rmse_xgb, "XGBoost", y_pred_xgb
else:
    print("  XGBoost not available - skipping")

# -------------------------------------------------------
# 9b. Gradient Boosting - LightGBM
# -------------------------------------------------------
print("\n" + "=" * 60)
print("9b. Gradient Boosting - LightGBM")
print("=" * 60)
if LGB_AVAILABLE:
    lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_lgb = r2_score(y_test, y_pred_lgb)
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    print(f"  RMSE: {rmse_lgb:.4f}")
    print(f"  R2:   {r2_lgb:.4f}")
    print(f"  MAE:  {mae_lgb:.4f}")
    results.append({"Model": "LightGBM", "RMSE": rmse_lgb, "R2": r2_lgb, "MAE": mae_lgb})
    if r2_lgb > best_r2:
        best_r2, best_rmse, best_model, best_predictions = r2_lgb, rmse_lgb, "LightGBM", y_pred_lgb
else:
    print("  LightGBM not available - skipping")

# -------------------------------------------------------
# 10. Support Vector Regression (SVR)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("10. Support Vector Regression (SVR)")
print("=" * 60)
svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
print(f"  RMSE: {rmse_svr:.4f}")
print(f"  R2:   {r2_svr:.4f}")
print(f"  MAE:  {mae_svr:.4f}")
results.append({"Model": "SVR", "RMSE": rmse_svr, "R2": r2_svr, "MAE": mae_svr})
if r2_svr > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_svr, rmse_svr, "SVR", y_pred_svr

# -------------------------------------------------------
# 11. KNN Regression
# -------------------------------------------------------
print("\n" + "=" * 60)
print("11. KNN Regression")
print("=" * 60)
knn_model = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
r2_knn = r2_score(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
print(f"  RMSE: {rmse_knn:.4f}")
print(f"  R2:   {r2_knn:.4f}")
print(f"  MAE:  {mae_knn:.4f}")
results.append({"Model": "KNN Regression", "RMSE": rmse_knn, "R2": r2_knn, "MAE": mae_knn})
if r2_knn > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_knn, rmse_knn, "KNN Regression", y_pred_knn

# -------------------------------------------------------
# 12. Neural Network Regression (MLP)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("12. Neural Network Regression (MLP)")
print("=" * 60)
mlp_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
r2_mlp = r2_score(y_test, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
print(f"  RMSE: {rmse_mlp:.4f}")
print(f"  R2:   {r2_mlp:.4f}")
print(f"  MAE:  {mae_mlp:.4f}")
results.append({"Model": "Neural Network (MLP)", "RMSE": rmse_mlp, "R2": r2_mlp, "MAE": mae_mlp})
if r2_mlp > best_r2:
    best_r2, best_rmse, best_model, best_predictions = r2_mlp, rmse_mlp, "Neural Network (MLP)", y_pred_mlp

# -------------------------------------------------------
# Summary of All Models
# -------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY (sorted by R2 descending)")
print("=" * 60)
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
display(results_df)

print(f"\n*** BEST MODEL: {best_model} ***")
print(f"  R2:   {best_r2:.4f}")
print(f"  RMSE: {best_rmse:.4f}")

# Store best predictions for Cell 6
best_pred = best_predictions
X_test_indices = X_test.index

# COMMAND ----------

# DBTITLE 1,Create Predictions Table
# ============================================================
# Cell 6: Create Predictions Table
# Save to lifeexpectancy.Silver.prediction with PrimaryKey, Actual, Predicted, Diff
# ============================================================

# Build predictions DataFrame using the best model
predictions_pdf = pd.DataFrame({
    'PrimaryKey': df.loc[X_test_indices, 'PrimaryKey'].values,
    'Actual_Life_Expectancy': y_test.values,
    'Predicted_Life_Expectancy': best_pred,
    'Diff_Actual_Prediction': y_test.values - best_pred,
    'Best_Model': best_model
})

# Add Year for reference
predictions_pdf['Year'] = df.loc[X_test_indices, 'Year'].values

# Reorder columns
predictions_pdf = predictions_pdf[['PrimaryKey', 'Year', 'Actual_Life_Expectancy', 'Predicted_Life_Expectancy', 'Diff_Actual_Prediction', 'Best_Model']]

print(f"Predictions table shape: {predictions_pdf.shape}")
print(f"Best Model: {best_model}")
print(f"R2 Score: {best_r2:.4f}")
print(f"RMSE: {best_rmse:.4f}")
display(predictions_pdf.head(20))

# Save as Spark table: lifeexpectancy.Silver.prediction
predictions_spark = spark.createDataFrame(predictions_pdf)
predictions_spark.write.mode("overwrite").saveAsTable("lifeexpectancy.Silver.prediction")
print("\nTable saved: lifeexpectancy.Silver.prediction")

# Apply Z-ORDER optimization (table < 10M rows)
spark.sql("""
OPTIMIZE lifeexpectancy.Silver.prediction
ZORDER BY (PrimaryKey)
""")
print("Z-ORDER optimization applied.")

# Verify table
display(spark.sql("SELECT * FROM lifeexpectancy.Silver.prediction LIMIT 20"))
print(f"\n=== ML Models Complete ===")
print(f"Best performing model: {best_model} with R2={best_r2:.4f}, RMSE={best_rmse:.4f}")

# COMMAND ----------

