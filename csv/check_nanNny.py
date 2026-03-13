import pandas as pd

# -----------------------------
# Lecture robuste du CSV brut
# -----------------------------
def read_raw_csv_robust(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df.columns = df.columns.str.strip()
    if "dataid" in df.columns:
        return df

    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    if "dataid" in df.columns:
        return df

    raise ValueError("Impossible de lire le CSV correctement.")


# -----------------------------
# Chargement New York
# -----------------------------
file_path = "data/15minute_data_newyork.csv"
df = read_raw_csv_robust(file_path)

print("\nColonnes détectées :")
print(df.columns.tolist())

# Colonnes heater et furnace
heater_cols = [c for c in df.columns if c.startswith("heater")]
furnace_cols = [c for c in df.columns if c.startswith("furnace")]

print("\nColonnes heater :", heater_cols)
print("Colonnes furnace :", furnace_cols)

# -----------------------------
# Analyse NaN
# -----------------------------
print("\n========== ANALYSE NaN ==========\n")

for col in heater_cols + furnace_cols:

    df[col] = pd.to_numeric(df[col], errors="coerce")

    nb_nan = df[col].isna().sum()
    total = len(df)
    pct = (nb_nan / total) * 100

    print(f"{col:12s} | NaN = {nb_nan:8d} / {total} ({pct:6.2f}%)")

print("\n==================================")