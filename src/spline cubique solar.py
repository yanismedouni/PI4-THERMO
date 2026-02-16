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
MAX_GAP = 8   # 4 × 15 min = 1 heure

# =============================
# 1) Chargement des données
# =============================
csv_path = "C:/Users/Edith-Irene/Desktop/PIGE/15minute_data_austin/15minute_data_austin/15minute_data_austin.csv"
df = pd.read_csv(csv_path)

# =============================
# 2) Nettoyage du timestamp
# =============================
# 1) Forcer tout en UTC (évite l'erreur mixed)
dt_utc = pd.to_datetime(df["local_15min"], errors="coerce", utc=True)

# 2) Convertir vers timezone fixe -06:00 (Austin standard)
dt_local = dt_utc.dt.tz_convert("Etc/GMT+6")

# 3) Supprimer le fuseau sans décaler l'heure
df["local_15min"] = dt_local.dt.tz_localize(None)

# =============================
# 3) Sélection du ménage
# =============================
clients = df["dataid"].unique()

for dataid in clients:

    print(f"Traitement du client {dataid}")

    df_client = df[df["dataid"] == dataid].copy()

    # =============================
    # 4) Filtrage sur période
    # =============================
    date_debut = "2018-10-01"
    date_fin   = "2018-10-16"

    mask = (
        (df_client["local_15min"] >= date_debut) &
        (df_client["local_15min"] <= date_fin)
    )
    df_client = df_client.loc[mask]

    if df_client.empty:
        continue

    # =============================
    # 5) Sélection solar
    # =============================
    df_client = df_client[["local_15min", "solar"]].copy()
    df_client["solar"] = pd.to_numeric(df_client["solar"], errors="coerce")
    df_client.loc[df_client["solar"] < 0, "solar"] = 0

    # =============================
    # 6) Mise sur grille 15 minutes
    # =============================
    df_client = (
        df_client
        .groupby("local_15min", as_index=False)
        .mean(numeric_only=True)
    )

    df_client = df_client.set_index("local_15min")
    df_client = df_client.asfreq("15min")
    df_client = df_client.reset_index()

    y_before = df_client["solar"].copy()

    # =============================
    # Interpolation spline cubique
    # =============================
    y = df_client["solar"]
    t = np.arange(len(y))

    df_client["is_nan"] = y.isna()
    df_client["nan_group"] = (
        df_client["is_nan"] != df_client["is_nan"].shift()
    ).cumsum()

    nan_lengths = (
        df_client[df_client["is_nan"]]
        .groupby("nan_group")
        .size()
    )

    y_interp = y.copy()

    EDGE_MARGIN = 8
    WINDOW = 8

    for group_id, length in nan_lengths.items():

        if length > MAX_GAP:
            continue

        idx = df_client[df_client["nan_group"] == group_id].index

        if idx.min() < EDGE_MARGIN:
            continue
        if idx.max() > len(y) - EDGE_MARGIN:
            continue

        i_start = max(idx.min() - WINDOW, 0)
        i_end   = min(idx.max() + WINDOW + 1, len(y))

        y_local = y.iloc[i_start:i_end]
        t_local = t[i_start:i_end]

        mask_local = ~y_local.isna()

        if mask_local.sum() < 4:
            continue

        cs_local = CubicSpline(
            t_local[mask_local],
            y_local[mask_local],
            bc_type="natural"
        )

        y_interp.loc[idx] = cs_local(t[idx])

    y_interp[y_interp < 0] = 0

    df_client["solar_interp"] = y_interp

    # =============================
    # Tracé pour ce client
    # =============================
    df_client = df_client.set_index("local_15min")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, 5),
        sharex=True,
        sharey=True
    )

    axes[0].plot(df_client.index, y_before, color="tab:blue")
    axes[0].set_title("Avant interpolation")
    axes[0].grid(True)

    axes[1].plot(df_client.index, df_client["solar_interp"], color="tab:orange")
    axes[1].set_title("Après interpolation")
    axes[1].grid(True)

    fig.suptitle(
        f"Client {dataid} – Solar\n"
        f"Spline cubique (trous ≤ {MAX_GAP*15} min)"
    )

    plt.tight_layout()
    plt.show()
