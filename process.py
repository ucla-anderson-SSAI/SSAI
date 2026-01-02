import pandas as pd

df = pd.read_csv("HMData.csv")

# Count rows per product id
rows_per_id = df.groupby("id").size()

# Keep only ids with exactly 12 rows (one per month)
valid_ids = rows_per_id[rows_per_id == 12].index

df_filtered = df[df["id"].isin(valid_ids)]

print(f"Original: {len(df)} rows, {df['id'].nunique()} products")
print(f"Filtered: {len(df_filtered)} rows, {df_filtered['id'].nunique()} products")

df_filtered.to_csv("HMData_filtered.csv", index=False)