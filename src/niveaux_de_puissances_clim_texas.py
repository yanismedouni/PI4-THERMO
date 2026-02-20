"""
Projet THERMO — Détection des niveaux de puissance
Texas (Austin) : janvier 2018 → décembre 2018
ANALYSE UNIQUEMENT DE LA CLIMATISATION
+ Correction lecture CSV
+ Filtrage sur une liste de clients donnée
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from dataclasses import dataclass
from typing import List


# =============================================================================
# PARAMÈTRES
# =============================================================================

CHEMIN_CSV = "/Users/admin/Documents/Projet intégrateur de grande envergure ELE8080/15minute_data_austin/15minute_data_austin.csv"

K_CLUSTERS = 3
COL_DATETIME = "local_15min"
RANDOM_STATE = 42

#  Clients Austin à conserver
CLIENTS_AUSTIN = [
    661, 1642, 2335, 2361, 2818, 3039, 3456, 3538, 4031, 4373, 4767, 5746
]

# Noms possibles pour la colonne ID client
CANDIDATS_COL_CLIENT = ["dataid", "data_id", "DataID", "id", "ID", "customer_id"]


# =============================================================================
# STRUCTURE RESULTAT
# =============================================================================

@dataclass
class ResultatModes:
    appareil: str
    k: int
    centres_kw_ordonnes: np.ndarray
    labels_modes: List[str]
    inertia: float
    df_modes: pd.DataFrame


# =============================================================================
# LECTURE CSV + DÉTECTION COLONNE CLIENT
# =============================================================================

def lire_csv_robuste(chemin_csv: str) -> pd.DataFrame:
    """
    Lit un CSV en auto-détectant le séparateur (',' ou ';') et nettoie les noms de colonnes.
    """
    df = pd.read_csv(chemin_csv, sep=None, engine="python")

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)  # enlève BOM
        .str.strip()
    )
    return df


def trouver_colonne_client(df: pd.DataFrame) -> str:
    """
    Trouve la colonne contenant l'identifiant client.
    """
    cols = set(df.columns)

    for cand in CANDIDATS_COL_CLIENT:
        if cand in cols:
            return cand

    #  recherche insensible à la casse
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in CANDIDATS_COL_CLIENT:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    raise KeyError(
        "Impossible de trouver la colonne d'ID client. "
        f"Colonnes dispo (début): {list(df.columns)[:30]} ... "
        f"(candidats: {CANDIDATS_COL_CLIENT})"
    )


def filtrer_clients(df: pd.DataFrame, col_client: str) -> pd.DataFrame:
    """
    Conserve uniquement les clients listés dans CLIENTS_AUSTIN.
    """
    df = df.copy()
    df[col_client] = pd.to_numeric(df[col_client], errors="coerce")
    return df[df[col_client].isin(CLIENTS_AUSTIN)].copy()


# =============================================================================
# FILTRAGE PERIODE (gestion fuseau horaire)
# =============================================================================

def filtrer_periode_texas_2018(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre : 2018-01-01 00:00:00 → 2018-12-31 23:59:59
    Compatible timestamps tz-aware ou tz-naive.
    """
    df = df.copy()
    df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME], errors="coerce")

    if getattr(df[COL_DATETIME].dt, "tz", None) is not None:
        tz = df[COL_DATETIME].dt.tz
        debut = pd.Timestamp("2018-01-01 00:00:00", tz=tz)
        fin = pd.Timestamp("2018-12-31 23:59:59", tz=tz)
    else:
        debut = pd.Timestamp("2018-01-01 00:00:00")
        fin = pd.Timestamp("2018-12-31 23:59:59")

    return df[(df[COL_DATETIME] >= debut) & (df[COL_DATETIME] <= fin)].copy()


# =============================================================================
# AGRÉGATION CLIMATISATION UNIQUEMENT
# =============================================================================

def agregation_climatisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    air_total = air1 + air2 + air3
    """
    df = df.copy()

    colonnes_clim = ["air1", "air2", "air3"]
    colonnes_existantes = [c for c in colonnes_clim if c in df.columns]

    if len(colonnes_existantes) == 0:
        raise ValueError("Aucune colonne climatisation trouvée (air1/air2/air3).")

    df["air_total"] = df[colonnes_existantes].fillna(0).sum(axis=1)
    return df


# =============================================================================
# KMEANS 1D
# =============================================================================

def entrainer_kmeans(valeurs: np.ndarray) -> KMeans:
    modele = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    modele.fit(valeurs.reshape(-1, 1))
    return modele


def ordonner_centres(centres: np.ndarray):
    indices = np.argsort(centres)
    centres_tries = centres[indices]

    if K_CLUSTERS == 3:
        labels = ["OFF", "ON_bas", "ON_haut"]
    else:
        labels = ["OFF", "ON_bas", "ON_moyen", "ON_haut"]

    return centres_tries, labels, indices


def remapper_labels(labels_pred: np.ndarray, indices: np.ndarray) -> np.ndarray:
    inv = np.empty_like(indices)
    inv[indices] = np.arange(len(indices))
    return inv[labels_pred]


# =============================================================================
# IDENTIFICATION DES MODES CLIMATISATION
# =============================================================================

def identifier_modes(df: pd.DataFrame, col_client: str) -> ResultatModes:
    """
    Applique K-means sur air_total et assigne OFF/ON_bas/ON_haut (ou 4 niveaux).
    """
    puissance = pd.to_numeric(df["air_total"], errors="coerce")

    masque = puissance.notna() & np.isfinite(puissance.values)
    valeurs_valides = puissance[masque].values

    if len(valeurs_valides) < K_CLUSTERS:
        raise ValueError(f"Pas assez de points valides pour K={K_CLUSTERS} (points={len(valeurs_valides)}).")

    modele = entrainer_kmeans(valeurs_valides)

    centres = modele.cluster_centers_.reshape(-1)
    centres_tries, labels, indices = ordonner_centres(centres)

    labels_pred = modele.predict(valeurs_valides.reshape(-1, 1))
    labels_remap = remapper_labels(labels_pred, indices)

    mode_index = np.full(len(df), np.nan)
    mode_label = np.array([None] * len(df), dtype=object)

    mode_index[masque.values] = labels_remap

    for i, nom in enumerate(labels):
        mode_label[mode_index == i] = nom

    df_modes = df[[COL_DATETIME, col_client]].copy()
    df_modes["air_total"] = puissance
    df_modes["mode_index"] = mode_index
    df_modes["mode_label"] = mode_label

    return ResultatModes(
        appareil="air_total",
        k=K_CLUSTERS,
        centres_kw_ordonnes=centres_tries,
        labels_modes=labels,
        inertia=float(modele.inertia_),
        df_modes=df_modes
    )


# =============================================================================
# TABLEAU RÉCAP
# =============================================================================

def tableau_recap(resultat: ResultatModes) -> pd.DataFrame:
    df_valide = resultat.df_modes[resultat.df_modes["mode_label"].notna()]

    recap = (
        df_valide
        .groupby("mode_label")["air_total"]
        .agg(
            moyenne_kw="mean",
            mediane_kw="median",
            std_kw="std",
            min_kw="min",
            max_kw="max",
            nb_points="count"
        )
        .reset_index()
    )

    return recap


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Lecture du fichier (auto-sep + nettoyage colonnes)...")
    df = lire_csv_robuste(CHEMIN_CSV)

    print("Détection colonne client...")
    col_client = trouver_colonne_client(df)
    print(f"[OK] Colonne client détectée : {col_client}")

    print("Filtrage des clients Austin sélectionnés...")
    df = filtrer_clients(df, col_client)
    print(f"[INFO] Lignes après filtre clients : {len(df)}")

    print("Filtrage Texas (janv→déc 2018)...")
    df = filtrer_periode_texas_2018(df)
    print(f"[INFO] Lignes après filtre période : {len(df)}")

    print("Agrégation climatisation uniquement...")
    df = agregation_climatisation(df)

    print("Clustering climatisation...")
    resultat = identifier_modes(df, col_client)

    print("\n=== Centres climatisation (kW) ===")
    print(resultat.centres_kw_ordonnes)

    print("\n=== Tableau recap climatisation ===")
    print(tableau_recap(resultat))


if __name__ == "__main__":
    main()
