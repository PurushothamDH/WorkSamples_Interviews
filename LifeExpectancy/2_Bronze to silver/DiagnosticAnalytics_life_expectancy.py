# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Import All Libraries
# ============================================================
# Cell 1: Import All Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# PySpark
from pyspark.sql.functions import col, corr

# Scikit-learn for diagnostics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

print("All libraries imported successfully.")

# COMMAND ----------

# DBTITLE 1,Load Dataset and Impute Nulls
# ============================================================
# Cell 2: Load Dataset and Replace Nulls
# Replace fact column nulls with median, string column nulls with mode
# ============================================================

# Load from Bronze Layer
bronze_table = "lifeexpectancy.brone_lifeexpectancy.life_expectancy"
spark_df = spark.table(bronze_table)
df = spark_df.toPandas()

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Identify numeric (fact) and string columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
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

print(f"\nTotal remaining nulls: {df.isnull().sum().sum()}")
print(f"Dataset shape after imputation: {df.shape}")
display(df.describe())

# COMMAND ----------

# DBTITLE 1,Diagnostic Analysis Charts for Power BI
# ============================================================
# Cell 3: Diagnostic Analysis Charts for Power BI Reports
# Suggest charts with X value, Y value, titles, and additional attributes
# ============================================================

# Define column groups for analysis
target_col = 'Life_expectancy'
health_indicators = ['Adult_Mortality', 'infant_deaths', 'under_five_deaths', 'HIV_AIDS']
immunization_cols = ['Hepatitis_B', 'Polio', 'Diphtheria', 'Measles']
lifestyle_cols = ['Alcohol', 'BMI']
economic_cols = ['GDP', 'percentage_expenditure', 'Total_expenditure', 'Income_composition_of_resources']
social_cols = ['Schooling', 'thinness_1_19_years', 'thinness_5_9_years']

# Convert Decimal columns to float for numpy compatibility
for col_name in df.columns:
    if df[col_name].dtype == 'object':
        try:
            df[col_name] = df[col_name].astype(float)
        except (TypeError, ValueError):
            pass

# -------------------------------------------------------
# 1. Correlation Heatmap - All numeric variables
# -------------------------------------------------------
print("=" * 70)
print("CHART 1: Correlation Heatmap of All Variables")
print("=" * 70)
print("  Chart Type: Heatmap")
print("  X Value: All numeric columns")
print("  Y Value: All numeric columns (correlation coefficients)")
print("  Title: 'Correlation Matrix - Life Expectancy & Health Indicators'")
print("  Color: Correlation strength (Red=positive, Blue=negative)")
print("  Power BI: Use Matrix visual with conditional formatting")

fig, ax = plt.subplots(figsize=(16, 12))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, ax=ax, annot_kws={'size': 7})
ax.set_title('Correlation Matrix - Life Expectancy & Health Indicators', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 2. Life Expectancy Trend Over Years
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 2: Life Expectancy Trend Over Years")
print("=" * 70)
print("  Chart Type: Line Chart")
print("  X Value: Year")
print("  Y Value: Life_expectancy (Average)")
print("  Title: 'Average Life Expectancy Trend Over Years (2000-2015)'")
print("  Additional: Min and Max Life Expectancy as bands")
print("  Power BI: Line chart with min/max error bands")

yearly_stats = df.groupby('Year')[target_col].agg(['mean', 'min', 'max']).reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(yearly_stats['Year'], yearly_stats['mean'], 'b-o', linewidth=2, label='Average Life Expectancy')
ax.fill_between(yearly_stats['Year'], yearly_stats['min'], yearly_stats['max'], alpha=0.2, color='blue', label='Min-Max Range')
ax.set_xlabel('Year')
ax.set_ylabel('Life Expectancy')
ax.set_title('Average Life Expectancy Trend Over Years (2000-2015)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 3. Adult Mortality vs Life Expectancy (Scatter + Trend)
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 3: Adult Mortality vs Life Expectancy")
print("=" * 70)
print("  Chart Type: Scatter Plot with Trend Line")
print("  X Value: Adult_Mortality")
print("  Y Value: Life_expectancy")
print("  Title: 'Adult Mortality vs Life Expectancy'")
print("  Additional: Trend line, color by Year")
print("  Power BI: Scatter chart with trend line and color gradient by Year")

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['Adult_Mortality'], df['Life_expectancy'], c=df['Year'], cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
z = np.polyfit(df['Adult_Mortality'], df['Life_expectancy'], 1)
p = np.poly1d(z)
ax.plot(df['Adult_Mortality'], p(df['Adult_Mortality']), 'r--', linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
plt.colorbar(scatter, label='Year')
ax.set_xlabel('Adult Mortality')
ax.set_ylabel('Life Expectancy')
ax.set_title('Adult Mortality vs Life Expectancy', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 4. Immunization Coverage Impact on Life Expectancy
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 4: Immunization Coverage vs Life Expectancy")
print("=" * 70)
print("  Chart Type: Multiple Line Charts (Small Multiples)")
print("  X Value: Hepatitis_B / Polio / Diphtheria (immunization %)")
print("  Y Value: Life_expectancy")
print("  Title: 'Immunization Coverage vs Life Expectancy'")
print("  Additional: 3 subplots, one per immunization type")
print("  Power BI: Small multiples (3 line charts side by side)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, imm_col in enumerate(['Hepatitis_B', 'Polio', 'Diphtheria']):
    axes[i].scatter(df[imm_col], df['Life_expectancy'], alpha=0.5, color=f'C{i}')
    z = np.polyfit(df[imm_col].fillna(df[imm_col].median()), df['Life_expectancy'], 1)
    p = np.poly1d(z)
    x_sorted = np.sort(df[imm_col].fillna(df[imm_col].median()))
    axes[i].plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
    axes[i].set_xlabel(imm_col)
    axes[i].set_ylabel('Life Expectancy')
    axes[i].set_title(f'{imm_col} vs Life Expectancy')
    axes[i].grid(True, alpha=0.3)
plt.suptitle('Immunization Coverage vs Life Expectancy', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 5. GDP vs Life Expectancy (Bubble Chart)
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 5: GDP vs Life Expectancy (Bubble Chart)")
print("=" * 70)
print("  Chart Type: Bubble Scatter Chart")
print("  X Value: GDP")
print("  Y Value: Life_expectancy")
print("  Bubble Size: Population")
print("  Color: Year")
print("  Title: 'GDP vs Life Expectancy (Bubble Size = Population)'")
print("  Power BI: Scatter chart with bubble size and color by Year")

fig, ax = plt.subplots(figsize=(12, 8))
# Use log scale for GDP since values vary widely
scatter = ax.scatter(df['GDP'], df['Life_expectancy'], 
                     s=df['Population']/df['Population'].max()*500, 
                     c=df['Year'], cmap='plasma', alpha=0.5, edgecolors='black', linewidth=0.5)
ax.set_xscale('log')
ax.set_xlabel('GDP (log scale)')
ax.set_ylabel('Life Expectancy')
ax.set_title('GDP vs Life Expectancy (Bubble Size = Population)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Year')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 6. Schooling vs Life Expectancy by Year
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 6: Schooling vs Life Expectancy")
print("=" * 70)
print("  Chart Type: Scatter with Trend Line")
print("  X Value: Schooling (years of education)")
print("  Y Value: Life_expectancy")
print("  Title: 'Schooling vs Life Expectancy'")
print("  Additional: Trend line, correlation coefficient")
print("  Power BI: Scatter chart with trend line")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['Schooling'], df['Life_expectancy'], alpha=0.5, color='green')
z = np.polyfit(df['Schooling'].fillna(df['Schooling'].median()), df['Life_expectancy'], 1)
p = np.poly1d(z)
x_sorted = np.sort(df['Schooling'].fillna(df['Schooling'].median()))
ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
corr_val = df['Schooling'].corr(df['Life_expectancy'])
ax.set_xlabel('Schooling (years)')
ax.set_ylabel('Life Expectancy')
ax.set_title(f'Schooling vs Life Expectancy (r={corr_val:.3f})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 7. HIV/AIDS vs Life Expectancy
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 7: HIV/AIDS Impact on Life Expectancy")
print("=" * 70)
print("  Chart Type: Scatter Plot with Trend Line")
print("  X Value: HIV_AIDS")
print("  Y Value: Life_expectancy")
print("  Title: 'HIV/AIDS Impact on Life Expectancy'")
print("  Additional: Log scale for X-axis, trend line")
print("  Power BI: Scatter chart with log X-axis and trend line")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['HIV_AIDS'], df['Life_expectancy'], alpha=0.5, color='red')
ax.set_xscale('log')
z = np.polyfit(np.log(df['HIV_AIDS'].fillna(df['HIV_AIDS'].median()).clip(lower=0.001)), df['Life_expectancy'], 1)
p = np.poly1d(z)
x_log = np.linspace(np.log(df['HIV_AIDS'].fillna(df['HIV_AIDS'].median()).clip(lower=0.001).min()), 
                   np.log(df['HIV_AIDS'].fillna(df['HIV_AIDS'].median()).clip(lower=0.001).max()), 100)
ax.plot(np.exp(x_log), p(x_log), 'b--', linewidth=2, label='Trend (log)')
ax.set_xlabel('HIV/AIDS (log scale)')
ax.set_ylabel('Life Expectancy')
ax.set_title('HIV/AIDS Impact on Life Expectancy', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 8. Feature Importance (Correlation with Life Expectancy)
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 8: Feature Importance - Correlation with Life Expectancy")
print("=" * 70)
print("  Chart Type: Horizontal Bar Chart")
print("  X Value: Correlation coefficient")
print("  Y Value: Feature names")
print("  Title: 'Feature Importance - Correlation with Life Expectancy'")
print("  Color: Positive (green) / Negative (red)")
print("  Power BI: Horizontal bar chart with conditional color formatting")

feature_cols = [c for c in numeric_cols if c != target_col and c != 'Year']
correlations = []
for col_name in feature_cols:
    corr_val = df[col_name].corr(df[target_col])
    correlations.append({'Feature': col_name, 'Correlation': corr_val})
corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red' if x < 0 else 'green' for x in corr_df['Correlation']]
ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Correlation Coefficient')
ax.set_ylabel('Feature')
ax.set_title('Feature Importance - Correlation with Life Expectancy', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 9. Life Expectancy Distribution by Year (Box Plot)
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 9: Life Expectancy Distribution by Year")
print("=" * 70)
print("  Chart Type: Box Plot")
print("  X Value: Year")
print("  Y Value: Life_expectancy")
print("  Title: 'Life Expectancy Distribution by Year'")
print("  Additional: Show median, quartiles, outliers")
print("  Power BI: Box and whisker plot")

fig, ax = plt.subplots(figsize=(14, 6))
box_data = [df[df['Year'] == yr]['Life_expectancy'].values for yr in sorted(df['Year'].unique())]
bp = ax.boxplot(box_data, labels=sorted(df['Year'].unique()), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_xlabel('Year')
ax.set_ylabel('Life Expectancy')
ax.set_title('Life Expectancy Distribution by Year', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 10. Health Expenditure vs Life Expectancy (Stacked)
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 10: Total Health Expenditure vs Life Expectancy by Year")
print("=" * 70)
print("  Chart Type: Dual Y-Axis Chart")
print("  X Value: Year")
print("  Y1 Value: Average Total_expenditure (Bar)")
print("  Y2 Value: Average Life_expectancy (Line)")
print("  Title: 'Health Expenditure vs Life Expectancy Over Years'")
print("  Power BI: Combo chart (bar + line with dual Y-axis)")

yearly_exp = df.groupby('Year').agg({
    'Total_expenditure': 'mean',
    'Life_expectancy': 'mean'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(yearly_exp['Year'], yearly_exp['Total_expenditure'], color='skyblue', alpha=0.7, label='Avg Total Expenditure')
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Total Expenditure', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(yearly_exp['Year'], yearly_exp['Life_expectancy'], 'r-o', linewidth=2, markersize=8, label='Avg Life Expectancy')
ax2.set_ylabel('Average Life Expectancy', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.set_title('Health Expenditure vs Life Expectancy Over Years', fontsize=14, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.95))
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 11. Alcohol Consumption vs Life Expectancy
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 11: Alcohol Consumption vs Life Expectancy")
print("=" * 70)
print("  Chart Type: Scatter Plot with Regression Line")
print("  X Value: Alcohol")
print("  Y Value: Life_expectancy")
print("  Title: 'Alcohol Consumption vs Life Expectancy'")
print("  Color: Year")
print("  Power BI: Scatter chart with trend line and color by Year")

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['Alcohol'], df['Life_expectancy'], c=df['Year'], cmap='coolwarm', alpha=0.6, edgecolors='black', linewidth=0.5)
z = np.polyfit(df['Alcohol'].fillna(df['Alcohol'].median()), df['Life_expectancy'], 1)
p = np.poly1d(z)
x_sorted = np.sort(df['Alcohol'].fillna(df['Alcohol'].median()))
ax.plot(x_sorted, p(x_sorted), 'k--', linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
plt.colorbar(scatter, label='Year')
ax.set_xlabel('Alcohol Consumption')
ax.set_ylabel('Life Expectancy')
ax.set_title('Alcohol Consumption vs Life Expectancy', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 12. BMI vs Life Expectancy (Colored by Thinness)
# -------------------------------------------------------
print("\n" + "=" * 70)
print("CHART 12: BMI vs Life Expectancy (Colored by Thinness 1-19 years)")
print("=" * 70)
print("  Chart Type: Scatter Plot with Color Gradient")
print("  X Value: BMI")
print("  Y Value: Life_expectancy")
print("  Color: thinness_1_19_years")
print("  Title: 'BMI vs Life Expectancy (Colored by Thinness 1-19 yrs)'")
print("  Power BI: Scatter chart with color saturation by thinness")

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['BMI'], df['Life_expectancy'], c=df['thinness_1_19_years'], cmap='YlOrRd', 
                     alpha=0.7, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Thinness 1-19 years (%)')
ax.set_xlabel('BMI')
ax.set_ylabel('Life Expectancy')
ax.set_title('BMI vs Life Expectancy (Colored by Thinness 1-19 yrs)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Summary of Recommended Power BI Charts
# -------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY: Recommended Power BI Charts for Diagnostic Analysis")
print("=" * 70)

chart_recommendations = [
    {"#": 1, "Chart Type": "Heatmap", "X Value": "All numeric columns", "Y Value": "Correlation coefficients", "Title": "Correlation Matrix", "Power BI Visual": "Matrix with conditional formatting"},
    {"#": 2, "Chart Type": "Line Chart", "X Value": "Year", "Y Value": "Life_expectancy (avg)", "Title": "Life Expectancy Trend Over Years", "Power BI Visual": "Line chart with min/max bands"},
    {"#": 3, "Chart Type": "Scatter + Trend", "X Value": "Adult_Mortality", "Y Value": "Life_expectancy", "Title": "Adult Mortality vs Life Expectancy", "Power BI Visual": "Scatter with trend line, color by Year"},
    {"#": 4, "Chart Type": "Small Multiples", "X Value": "Hepatitis_B/Polio/Diphtheria", "Y Value": "Life_expectancy", "Title": "Immunization Coverage vs Life Expectancy", "Power BI Visual": "Small multiples (3 charts)"},
    {"#": 5, "Chart Type": "Bubble Chart", "X Value": "GDP", "Y Value": "Life_expectancy", "Title": "GDP vs Life Expectancy (Bubble=Population)", "Power BI Visual": "Scatter with bubble size, color by Year"},
    {"#": 6, "Chart Type": "Scatter + Trend", "X Value": "Schooling", "Y Value": "Life_expectancy", "Title": "Schooling vs Life Expectancy", "Power BI Visual": "Scatter with trend line"},
    {"#": 7, "Chart Type": "Scatter (log X)", "X Value": "HIV_AIDS", "Y Value": "Life_expectancy", "Title": "HIV/AIDS Impact on Life Expectancy", "Power BI Visual": "Scatter with log X-axis"},
    {"#": 8, "Chart Type": "Horizontal Bar", "X Value": "Correlation coefficient", "Y Value": "Feature names", "Title": "Feature Importance", "Power BI Visual": "Horizontal bar with color formatting"},
    {"#": 9, "Chart Type": "Box Plot", "X Value": "Year", "Y Value": "Life_expectancy", "Title": "Life Expectancy Distribution by Year", "Power BI Visual": "Box and whisker plot"},
    {"#": 10, "Chart Type": "Dual Y-Axis", "X Value": "Year", "Y1 Value": "Total_expenditure", "Y2 Value": "Life_expectancy", "Title": "Health Expenditure vs Life Expectancy", "Power BI Visual": "Combo chart (bar + line)"},
    {"#": 11, "Chart Type": "Scatter + Trend", "X Value": "Alcohol", "Y Value": "Life_expectancy", "Title": "Alcohol Consumption vs Life Expectancy", "Power BI Visual": "Scatter with trend, color by Year"},
    {"#": 12, "Chart Type": "Scatter (gradient)", "X Value": "BMI", "Y Value": "Life_expectancy", "Title": "BMI vs Life Expectancy (by Thinness)", "Power BI Visual": "Scatter with color saturation"}
]

recommendations_df = pd.DataFrame(chart_recommendations)
display(recommendations_df)

print("\n=== Diagnostic Analytics Complete ===")
print("12 charts generated for Power BI diagnostic analysis.")

# COMMAND ----------

