"""
Projet THERMO
Californie (San Diego) — Climatisation
KMeans k=4 + statistiques descriptives par cluster
Clients filtrés
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ==============================
# PARAMÈTRES
# ==============================

CHEMIN_CSV = "output/processed_energy_data_california.csv"
COL_DATE = "local_15min"
COL_CLIENT = "dataid"

K = 4
RANDOM_STATE = 42

# Clients Californie (San Diego)
CLIENTS_CA = [
    1731, 4495, 8342,          # 2018
    3938, 4934, 5938, 8061, 9775,  # 2016
    203, 1450, 1524            # 2015
]

# Fenêtres temporelles (selon tes années)
# -> tu peux laisser large ou restreindre.
DATE_DEBUT = "2015-01-01"
DATE_FIN   = "2018-12-31 23:59:59"

# Noms des clusters (k=4)
NOMS_CLUSTERS = ["OFF", "ON_bas", "ON_intermédiaire", "ON_haut"]


# ==============================
# MAIN
# ==============================

def main():

    # 1) Charger données
    df = pd.read_csv(CHEMIN_CSV, parse_dates=[COL_DATE])

    # 2) Filtrer clients Californie
    df = df[df[COL_CLIENT].isin(CLIENTS_CA)].copy()
    print(f"[INFO] Lignes après filtre clients : {len(df):,}")

    # 3) Filtrer période (2015–2018)
    df = df[(df[COL_DATE] >= DATE_DEBUT) & (df[COL_DATE] <= DATE_FIN)].copy()
    print(f"[INFO] Lignes période CA : {len(df):,}")

    # 4) Utiliser la colonne clim
    if "clim" not in df.columns:
        raise ValueError("Colonne 'clim' introuvable dans le fichier processed California.")

    df["clim_total"] = pd.to_numeric(df["clim"], errors="coerce")
    df.loc[df["clim_total"] < 0, "clim_total"] = 0

    df = df[df["clim_total"].notna()].copy()
    print(f"[INFO] Lignes valides : {len(df):,}")

    if len(df) < K:
        raise ValueError(f"Pas assez de points valides pour K={K} (points={len(df)}).")

    # 5) KMeans 1D
    X = df["clim_total"].values.reshape(-1, 1)

    modele = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init="auto"
    )
    modele.fit(X)

    # Centres -> triés
    centres = modele.cluster_centers_.reshape(-1)
    idx_tri = np.argsort(centres)
    centres_tries = centres[idx_tri]

    # labels : OFF < ON_bas < ON_intermédiaire < ON_haut
    labels_bruts = modele.labels_
    inv = np.empty_like(idx_tri)
    inv[idx_tri] = np.arange(len(idx_tri))
    df["cluster"] = inv[labels_bruts]

    # 6) Statistiques par cluster
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

    recap["mode"] = recap["cluster"].map(dict(enumerate(NOMS_CLUSTERS)))

    # 7) Affichages
    print("\n=== Centres KMeans (kW) ===")
    for nom, c in zip(NOMS_CLUSTERS, centres_tries):
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