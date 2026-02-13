# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 21:57:36 2026

@author: Edith-Irene
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# =============================
# PARAMÈTRES
# =============================
MAX_GAP = 20   # 4 × 15 min = 1 heure

# =============================
# 1) Chargement des données
# =============================
csv_path = "C:/Users/Edith-Irene/Desktop/PIGE/15minute_data_austin/15minute_data_austin/15minute_data_austin.csv"
df = pd.read_csv(csv_path)

# =============================
# 2) Nettoyage du timestamp
# =============================
s = df["local_15min"].astype(str).str.strip()
s = s.str.replace("T", " ", regex=False)
s = s.str.replace(r"\.\d{1,6}", "", regex=True)
s = s.str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True)

dt = pd.to_datetime(s, errors="coerce", utc=True)
df["local_15min"] = dt.dt.tz_localize(None)

# =============================
# 3) Sélection du ménage
# =============================
dataid = 661
df = df[df["dataid"] == dataid]

# =============================
# 4) Filtrage sur 3 mois
# =============================
# date_debut = "2018-05-01"
# date_fin   = "2018-07-31"

# mask = (df["local_15min"] >= date_debut) & (df["local_15min"] <= date_fin)
# df = df.loc[mask]

# =============================
# 5) Sélection puissance réseau
# =============================
df = df[["local_15min", "grid"]].copy()
df["grid"] = pd.to_numeric(df["grid"], errors="coerce")

if df.empty:
    print("Aucune donnée sur cette période.")
    raise SystemExit

# =============================
# 6) Mise sur grille 15 minutes
# =============================
df = (
    df
    .groupby("local_15min", as_index=False)
    .mean(numeric_only=True)
)

df = df.set_index("local_15min")
df = df.asfreq("15min")
df = df.reset_index()

# =============================
# 7) Préparation spline cubique
# =============================
y = df["grid"]
t = np.arange(len(y))

mask_valid = ~y.isna()
t_valid = t[mask_valid]
y_valid = y[mask_valid]

cs = CubicSpline(t_valid, y_valid, bc_type="natural")

# =============================
# 8) Détection des trous continus
# =============================
df["is_nan"] = y.isna()
df["nan_group"] = (df["is_nan"] != df["is_nan"].shift()).cumsum()

nan_lengths = (
    df[df["is_nan"]]
    .groupby("nan_group")
    .size()
)

# =============================
# 9) Interpolation UNIQUEMENT des trous courts
# =============================
y_interp = y.copy()

for group_id, length in nan_lengths.items():
    if length <= MAX_GAP:
        idx = df[df["nan_group"] == group_id].index
        y_interp.loc[idx] = cs(t[idx])

# contrainte physique
y_interp[y_interp < 0] = 0

df["grid_interp"] = y_interp

# =============================
# 10) Agrégation horaire
# =============================
df = df.set_index("local_15min")
power_hourly = df["grid_interp"]

df_export = (
    df
    .reset_index()[["local_15min", "grid_interp"]]
)

df_export.to_csv(
    "grid_15min_interpole.csv",
    index=False,
    float_format="%.6f"
)


# =============================
# 11) Tracé
# =============================
plt.figure(figsize=(14,5))
plt.plot(power_hourly.index, power_hourly.values)
plt.xlabel("Temps")
plt.ylabel("Puissance moyenne (kW)")
plt.title(
    f"Puissance moyenne horaire – ménage {dataid}\n"
    f"(Spline cubique, trous ≤ {MAX_GAP*15} min)"
)
plt.grid(True)
plt.tight_layout()
plt.show()