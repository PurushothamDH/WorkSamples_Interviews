import sys
import os

csv_path = os.path.join(os.path.dirname(__file__), 'final_cities.csv')
output_png = os.path.join(os.path.dirname(__file__), 'city_vs_area_top30.png')

try:
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('Missing required packages or import error:', e)
    print('Please install required packages with: pip install pandas matplotlib')
    sys.exit(2)

if not os.path.exists(csv_path):
    print(f'Data file not found: {csv_path}')
    sys.exit(1)

# Read CSV
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print('Failed to read CSV:', e)
    sys.exit(1)

# Normalize column names
col_candidates = [c for c in df.columns if 'area' in c.lower()]
if not col_candidates:
    print('No "area" column found in CSV. Columns:', df.columns.tolist())
    sys.exit(1)
area_col = col_candidates[0]

# Ensure city column
city_col = None
for c in df.columns:
    if 'city' == c.lower() or 'city' in c.lower():
        city_col = c
        break
if city_col is None:
    print('No "City" column found in CSV. Columns:', df.columns.tolist())
    sys.exit(1)

# Clean and convert area
df[area_col] = pd.to_numeric(df[area_col].astype(str).str.replace(',', '').str.replace('\u00A0','').str.strip(), errors='coerce')
# Drop missing or zero areas
df_clean = df.dropna(subset=[area_col]).copy()
# Keep only positive area values
df_clean = df_clean[df_clean[area_col] > 0]
if df_clean.empty:
    print('No valid positive area values found.')
    sys.exit(1)

# Select top N by area
TOP_N = 30
df_top = df_clean.sort_values(by=area_col, ascending=False).head(TOP_N)

# Prepare labels
labels = df_top[city_col].astype(str)
values = df_top[area_col]

# Plot
plt.figure(figsize=(14, 8))
bars = plt.bar(labels, values, color='tab:blue')
plt.xticks(rotation=75, ha='right')
plt.ylabel(f'{area_col}')
plt.title('City vs Area (top {} by Area)'.format(len(df_top)))
plt.tight_layout()

# Annotate bars with values
for bar in bars:
    h = bar.get_height()
    plt.annotate(f'{h:.0f}',
                 xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3),
                 textcoords='offset points',
                 ha='center', va='bottom', fontsize=8)

# Save
plt.savefig(output_png, dpi=150)
print('Saved chart to', output_png)
