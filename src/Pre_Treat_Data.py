import pandas as pd

# ------------------------------------------------------------------
# Chemins
# ------------------------------------------------------------------
input_path = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\austin\15minute_data_austin_with_temp_hourly_matched.csv"
output_path = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\austin\15minute_data_austin_modified.csv"

df = pd.read_csv(input_path)

# --- Colonnes à sommer (grid + solar + solar2)
sum_cols = ["grid", "solar", "solar2"]

# Vérifie qu'elles existent
missing = [c for c in sum_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans le CSV: {missing}\nColonnes dispo: {list(df.columns)}")

# Conversion robuste en numérique + NaN -> 0
for c in sum_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Somme (écrase grid avec le total)
df["grid"] = df["grid"] + df["solar"] + df["solar2"]

# (Optionnel) si tu veux supprimer daytime et local_15min
df = df.drop(columns=[c for c in ["daytime", "local_15min"] if c in df.columns])

# (Optionnel) réordonner year month day hour minute temp juste après dataid
cols_to_move = ["year", "month", "day", "hour", "minute", "temp"]
cols_to_move = [c for c in cols_to_move if c in df.columns]

remaining_cols = [c for c in df.columns if c not in cols_to_move]
idx = remaining_cols.index("dataid") + 1
df = df[remaining_cols[:idx] + cols_to_move + remaining_cols[idx:]]

df.to_csv(output_path, index=False)
print("✅ Sauvegardé:", output_path)