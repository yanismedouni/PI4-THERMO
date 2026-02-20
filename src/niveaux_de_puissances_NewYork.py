"""
Projet THERMO — Détection des niveaux de puissance (avec prétraitement)
New York : mai 2019 → octobre 2019
ANALYSE UNIQUEMENT DU CHAUFFAGE

Pipeline :
1) Prétraitement (grid_interp + solar_interp + EV + colonnes extra) -> CSV propre
2) Lecture du CSV propre
3) Filtrage clients (IDs donnés)
4) Filtrage période
5) Agrégation chauffage (furnace/heater)
6) K-means 1D -> OFF / ON_bas / ON_haut
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

# Import du prétraitement (déjà intégré dans ta branche)
from src.pretraitement import process_energy_data


# =============================================================================
# PARAMÈTRES
# =============================================================================

K_CLUSTERS = 3
COL_DATETIME = "local_15min"
RANDOM_STATE = 42

# Clients à conserver (New York)
CLIENTS_NY = [
    27, 387, 558, 914, 950, 1240, 1417, 3000, 3488, 3517, 5058, 5587
]

# Noms possibles pour la colonne ID client
CANDIDATS_COL_CLIENT = ["dataid", "data_id", "DataID", "id", "ID", "customer_id"]

# Seuil EV (kW) pour retirer car1+car2 du grid si dépassement
SEUIL_EV_KW = 3.0


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
# OUTILS CHEMINS (robuste)
# =============================================================================

def chemin_repo_racine() -> Path:
    """
    Détermine la racine du repo en remontant depuis ce fichier jusqu'à trouver un .git.
    Si non trouvé, prend le dossier courant.
    """
    p = Path(__file__).resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


# =============================================================================
# LECTURE CSV + NORMALISATION COLONNES
# =============================================================================

def lire_csv_robuste(chemin_csv: Path) -> pd.DataFrame:
    """
    Lit un CSV en auto-détectant le séparateur (',' ou ';') et nettoie les noms de colonnes.
    """
    df = pd.read_csv(chemin_csv, sep=None, engine="python")

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def trouver_colonne_client(df: pd.DataFrame) -> str:
    """
    Retourne le nom de la colonne client détectée.
    """
    cols = set(df.columns)

    for cand in CANDIDATS_COL_CLIENT:
        if cand in cols:
            return cand

    cols_lower = {c.lower(): c for c in df.columns}
    for cand in CANDIDATS_COL_CLIENT:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    raise KeyError(
        "Impossible de trouver la colonne d'ID client. "
        f"Colonnes dispo (début): {list(df.columns)[:30]} ... "
        f"(candidats: {CANDIDATS_COL_CLIENT})"
    )


# =============================================================================
# FILTRAGE CLIENTS
# =============================================================================

def filtrer_clients(df: pd.DataFrame, col_client: str) -> pd.DataFrame:
    """
    Conserve uniquement les clients listés dans CLIENTS_NY.
    """
    df = df.copy()
    df[col_client] = pd.to_numeric(df[col_client], errors="coerce")
    return df[df[col_client].isin(CLIENTS_NY)].copy()


# =============================================================================
# FILTRAGE PERIODE (gestion timezone)
# =============================================================================

def filtrer_periode_ny(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre : 2019-05-01 00:00:00 → 2019-10-31 23:59:59
    Compatible timestamps tz-aware ou tz-naive.
    """
    df = df.copy()
    df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME], errors="coerce")

    if getattr(df[COL_DATETIME].dt, "tz", None) is not None:
        tz = df[COL_DATETIME].dt.tz
        debut = pd.Timestamp("2019-05-01 00:00:00", tz=tz)
        fin = pd.Timestamp("2019-10-31 23:59:59", tz=tz)
    else:
        debut = pd.Timestamp("2019-05-01 00:00:00")
        fin = pd.Timestamp("2019-10-31 23:59:59")

    return df[(df[COL_DATETIME] >= debut) & (df[COL_DATETIME] <= fin)].copy()


# =============================================================================
# AGRÉGATION CHAUFFAGE UNIQUEMENT
# =============================================================================

def agregation_chauffage(df: pd.DataFrame) -> pd.DataFrame:
    """
    heat_total = furnace1 + furnace2 + heater1 + heater2 + heater3
    """
    df = df.copy()

    colonnes_chauffage = ["furnace1", "furnace2", "heater1", "heater2", "heater3"]
    colonnes_existantes = [c for c in colonnes_chauffage if c in df.columns]

    if len(colonnes_existantes) == 0:
        raise ValueError(
            "Aucune colonne chauffage trouvée. "
            f"Attendu au moins une parmi: {colonnes_chauffage}"
        )

    df["heat_total"] = df[colonnes_existantes].fillna(0).sum(axis=1)
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
# IDENTIFICATION DES MODES CHAUFFAGE
# =============================================================================

def identifier_modes(df: pd.DataFrame, col_client: str) -> ResultatModes:
    """
    Applique K-means sur heat_total et assigne OFF/ON_bas/ON_haut (ou 4 niveaux).
    """
    puissance = pd.to_numeric(df["heat_total"], errors="coerce")

    masque = puissance.notna() & np.isfinite(puissance.values)
    valeurs_valides = puissance[masque].values

    if len(valeurs_valides) < K_CLUSTERS:
        raise ValueError(
            f"Pas assez de points valides pour K={K_CLUSTERS} (points={len(valeurs_valides)})."
        )

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
    df_modes["heat_total"] = puissance
    df_modes["mode_index"] = mode_index
    df_modes["mode_label"] = mode_label

    return ResultatModes(
        appareil="heat_total",
        k=K_CLUSTERS,
        centres_kw_ordonnes=centres_tries,
        labels_modes=labels,
        inertia=float(modele.inertia_),
        df_modes=df_modes,
    )


# =============================================================================
# TABLEAU RÉCAP
# =============================================================================

def tableau_recap(resultat: ResultatModes) -> pd.DataFrame:
    df_valide = resultat.df_modes[resultat.df_modes["mode_label"].notna()]

    recap = (
        df_valide
        .groupby("mode_label")["heat_total"]
        .agg(
            moyenne_kw="mean",
            mediane_kw="median",
            std_kw="std",
            min_kw="min",
            max_kw="max",
            nb_points="count",
        )
        .reset_index()
    )
    return recap


# =============================================================================
# PRÉTRAITEMENT : GÉNÉRER CSV PROPRE NEW YORK
# =============================================================================

def generer_csv_propre_newyork(
    repo_root: Path,
    ev_threshold_kw: float = SEUIL_EV_KW,
) -> Path:
    """
    Lance le prétraitement et retourne le chemin du CSV propre généré.

    IMPORTANT : adapte ici les chemins grid/solar/raw New York selon ton repo.
    """
    # Adapte selon TON arborescence repo
    grid_path = repo_root / "csv" / "output" / "grid_interp.csv"
    solar_path = repo_root / "csv" / "output" / "solar_interp.csv"
    raw_path = repo_root / "data" / "15minute_data_newyork.csv"

    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processed_energy_data_newyork.csv"

    # Lancer le prétraitement
    process_energy_data(
        grid_file=str(grid_path),
        solar_file=str(solar_path),
        raw_file=str(raw_path),
        output_file=str(output_path),
        ev_threshold_kw=ev_threshold_kw,
    )

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    repo_root = chemin_repo_racine()
    print(f"[INFO] Racine repo détectée : {repo_root}")

    print("\n[1/6] Prétraitement -> génération CSV propre New York...")
    chemin_csv_propre = generer_csv_propre_newyork(repo_root)
    print(f"[OK] CSV propre : {chemin_csv_propre}")

    print("\n[2/6] Lecture du CSV propre (auto-sep + nettoyage colonnes)...")
    df = lire_csv_robuste(chemin_csv_propre)

    print("\n[3/6] Détection colonne client...")
    col_client = trouver_colonne_client(df)
    print(f"[OK] Colonne client détectée : {col_client}")

    print("\n[4/6] Filtrage des clients sélectionnés (New York)...")
    df = filtrer_clients(df, col_client)
    print(f"[INFO] Lignes après filtre clients : {len(df)}")

    print("\n[5/6] Filtrage période New York (mai→oct 2019)...")
    df = filtrer_periode_ny(df)
    print(f"[INFO] Lignes après filtre période : {len(df)}")

    print("\n[6/6] Agrégation + clustering chauffage...")
    df = agregation_chauffage(df)
    resultat = identifier_modes(df, col_client)

    print("\n=== Centres chauffage (kW) ===")
    print(resultat.centres_kw_ordonnes)

    print("\n=== Tableau recap chauffage ===")
    print(tableau_recap(resultat))


if __name__ == "__main__":
    main()
