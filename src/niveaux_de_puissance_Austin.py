"""
Projet THERMO
Austin 2018 — Climatisation
KMeans k=4 + statistiques descriptives par cluster
Clients filtrés
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ==============================
# PARAMÈTRES
# ==============================

CHEMIN_CSV = "output/processed_energy_data_austin.csv"
COL_DATE = "local_15min"
COL_CLIENT = "dataid"

K = 4
RANDOM_STATE = 42

CLIENTS_AUSTUSTIN = [
    661, 1642, 2335, 2361, 2818, 3039,
    3456, 3538, 4031, 4373, 4767, 5746
]

# ==============================
# MAIN
# ==============================

def main():

    # 1) Charger données
    df = pd.read_csv(CHEMIN_CSV, parse_dates=[COL_DATE])

    # 2) Filtrer clients
    df = df[df[COL_CLIENT].isin(CLIENTS_AUSTUSTIN)].copy()
    print(f"[INFO] Lignes après filtre clients : {len(df):,}")

    # 3) Filtrer année 2018
    df = df[(df[COL_DATE] >= "2018-01-01") &
            (df[COL_DATE] <= "2018-12-31")].copy()
    print(f"[INFO] Lignes 2018 : {len(df):,}")

    # 4) Utiliser la colonne clim
    df["clim_total"] = pd.to_numeric(df["clim"], errors="coerce")
    df.loc[df["clim_total"] < 0, "clim_total"] = 0

    df = df[df["clim_total"].notna()].copy()
    print(f"[INFO] Lignes valides : {len(df):,}")

    # 5) KMeans
    X = df["clim_total"].values.reshape(-1, 1)

    modele = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init="auto"
    )
    modele.fit(X)

    # Centres bruts -> triés
    centres = modele.cluster_centers_.reshape(-1)
    idx_tri = np.argsort(centres)
    centres_tries = centres[idx_tri]

    labels_bruts = modele.labels_
    inv = np.empty_like(idx_tri)
    inv[idx_tri] = np.arange(len(idx_tri))
    df["cluster"] = inv[labels_bruts]

    noms_clusters = ["OFF", "ON_bas", "ON_intermédiaire", "ON_haut"]

    # 6) Tableau statistiques
    recap = (
        df.groupby("cluster")["clim_total"]
        .agg(
            moyenne_kw="mean",
            mediane_kw="median",
            std_kw="std",
            min_kw="min",
            max_kw="max",
            nb_points="count"
        )
        .reset_index()
        .sort_values("cluster")
    )

    recap["mode"] = recap["cluster"].map(dict(enumerate(noms_clusters)))

    # Affichages
    print("\n=== Centres KMeans (kW) ===")
    for nom, c in zip(noms_clusters, centres_tries):
        print(f"{nom:16s}: {c:.4f} kW")

    print("\n=== Statistiques par cluster ===")
    print(recap[[
        "mode",
        "moyenne_kw",
        "mediane_kw",
        "std_kw",
        "min_kw",
        "max_kw",
        "nb_points"
    ]])


if __name__ == "__main__":
    main()