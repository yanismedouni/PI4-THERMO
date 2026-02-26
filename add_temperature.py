"""
Fusionne les données de température (horaires) avec les données énergétiques (15 min).
- Interpole les températures à 15 minutes
- Merge sur la colonne local_15min
- Sauvegarde le résultat dans un nouveau CSV

Usage:
    python add_temperature.py
"""

import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — modifie ces chemins si nécessaire
# ─────────────────────────────────────────────
ENERGY_FILE  = "output/austin_processed_energy_data.csv"
WEATHER_FILE = "data/open-meteo-30.26N97.69W157m.csv"
OUTPUT_FILE  = "output/austin_processed_energy_data.csv"  # écrase le fichier ou change le nom

# ─────────────────────────────────────────────
# Chargement des données énergétiques
# ─────────────────────────────────────────────
print("Chargement des données énergétiques...")
df_energy = pd.read_csv(ENERGY_FILE)
df_energy["local_15min"] = pd.to_datetime(df_energy["local_15min"])

# ─────────────────────────────────────────────
# Chargement des données météo
# Saute les 3 premières lignes (metadata) et commence à la ligne 4
# ─────────────────────────────────────────────
print("Chargement des données météo...")
df_weather = pd.read_csv(WEATHER_FILE, skiprows=3)
df_weather.columns = df_weather.columns.str.strip()

# Renommer les colonnes
df_weather = df_weather.rename(columns={
    "time": "local_15min",
    df_weather.columns[1]: "temperature"
})

df_weather["local_15min"] = pd.to_datetime(df_weather["local_15min"])
df_weather = df_weather[["local_15min", "temperature"]].dropna()
df_weather = df_weather.sort_values("local_15min").reset_index(drop=True)

# ─────────────────────────────────────────────
# Interpolation à 15 minutes
# ─────────────────────────────────────────────
print("Interpolation des températures à 15 minutes...")
df_weather = df_weather.set_index("local_15min")
df_weather = df_weather.resample("15min").interpolate(method="linear")
df_weather = df_weather.reset_index()

# ─────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────
print("Fusion des données...")
df_result = pd.merge(df_energy, df_weather, on="local_15min", how="left")

# Placer la colonne température après 'temp' si elle existe, sinon en 2e position
if "temp" in df_result.columns:
    cols = df_result.columns.tolist()
    cols.remove("temperature")
    idx = cols.index("temp") + 1
    cols.insert(idx, "temperature")
    df_result = df_result[cols]

# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────
df_result.to_csv(OUTPUT_FILE, index=False)
print(f"\n✔ Fichier sauvegardé : {OUTPUT_FILE}")
print(f"  Lignes : {len(df_result):,}")
print(f"  Températures manquantes : {df_result['temperature'].isna().sum():,}")