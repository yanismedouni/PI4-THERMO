"""
Projet THERMO
Méthode du coude – Climatisation Austin (2018)
Interpolation locale des trous courts (style csv_grid)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import CubicSpline


# ============================================================
# PARAMÈTRES interpolation
# ============================================================

MAX_GAP = 8          # <= 8 pas de 15 min = 2h
EDGE_MARGIN = 8
WINDOW = 8


# ============================================================
# Fonction interpolation trous courts (style grid)
# ============================================================

def interpolate_short_gaps(series):
    y = series.copy()
    t = np.arange(len(y))

    is_nan = y.isna()
    groups = (is_nan != is_nan.shift()).cumsum()
    lengths = y[is_nan].groupby(groups[is_nan]).size()

    y_interp = y.copy()

    for group_id, length in lengths.items():
        if length > MAX_GAP:
            continue

        idx = y[groups == group_id].index

        if idx.min() < EDGE_MARGIN or idx.max() > len(y) - EDGE_MARGIN:
            continue

        i_start = max(idx.min() - WINDOW, 0)
        i_end = min(idx.max() + WINDOW + 1, len(y))

        y_local = y.iloc[i_start:i_end]
        t_local = t[i_start:i_end]

        mask = ~y_local.isna()
        if mask.sum() < 4:
            continue

        cs = CubicSpline(t_local[mask], y_local[mask], bc_type="natural")
        y_interp.loc[idx] = cs(t[idx])

    y_interp[y_interp < 0] = 0
    return y_interp


# ============================================================
# 1. Charger dataset prétraité
# ============================================================

df = pd.read_csv(
    "output/processed_energy_data.csv",
    parse_dates=["local_15min"]
)

print("Dataset chargé :", df.shape)


# ============================================================
# 2. Filtrer 2018 uniquement
# ============================================================

df = df[
    (df["local_15min"] >= "2018-01-01") &
    (df["local_15min"] <= "2018-12-31")
].copy()

print("Après filtre 2018 :", df.shape)


# ============================================================
# 3. Construire air_total (Austin → air1 dominant)
# ============================================================

df["air_total"] = pd.to_numeric(df["air1"], errors="coerce")

# Interpolation par client (important)
df_list = []

for dataid, df_client in df.groupby("dataid"):
    df_client = df_client.sort_values("local_15min").copy()
    df_client["air_total"] = interpolate_short_gaps(df_client["air_total"])
    df_list.append(df_client)

df = pd.concat(df_list)

# Supprimer NaN restants (trous longs)
df = df[df["air_total"].notna()]

print("Après interpolation :", df.shape)


# ============================================================
# 4. Méthode du coude
# ============================================================

X = df["air_total"].values.reshape(-1, 1)

inertias = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    print(f"K={k} | Inertia={kmeans.inertia_:.2f}")


# ============================================================
# 5. Plot
# ============================================================

plt.figure(figsize=(8,5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertia")
plt.title("Méthode du coude – Climatisation Austin (2018)")
plt.grid(True)
plt.tight_layout()
plt.show()
