"""
Chargement et normalisation des données - THERMO NILM
Supporte deux formats d'entrée :
  - "region"       : processed_energy_data_*.csv  (colonnes year/month/day/hour/minute)
  - "desagregation": resultats_desagregation_*.csv (colonne timestamp)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES COMMUNS
# ══════════════════════════════════════════════════════════════════════

def load_results_csv(
    csv_path: str,
    encoding: str = "utf-8",
    sep: Optional[str] = None,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        encoding=encoding,
        sep=sep or None,
        engine="python",
        usecols=usecols,
    )


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}")


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION COMMUNE
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent

# Fichiers de données par région
REGIONS = {
    "Austin"    : "processed_energy_data_austin.csv",
    "California": "processed_energy_data_california.csv",
    "New York"  : "processed_energy_data_newyork.csv",
}

DATAID      = None   # None = tous les clients, int = un seul client
SEUIL_ON    = 0.05   # kW - seuil de détection état ON
PAS_MINUTES = 15     # résolution temporelle des données

ORDRE_SAISONS = ["Hiver", "Printemps", "Été", "Automne"]

MAP_SAISON = {
    12: "Hiver",     1: "Hiver",     2: "Hiver",
     3: "Printemps", 4: "Printemps", 5: "Printemps",
     6: "Été",       7: "Été",       8: "Été",
     9: "Automne",  10: "Automne",  11: "Automne",
}

# Couleurs pour les 3 régions prédéfinies
COULEURS_REGIONS = {
    "Austin"    : "#2E75B6",
    "California": "#E06C2E",
    "New York"  : "#2E8B57",
}

# Palette de secours pour les labels non prédéfinis (mode désagrégation)
_PALETTE_FALLBACK = [
    "#2E75B6", "#E06C2E", "#2E8B57", "#9B59B6", "#E74C3C",
    "#1ABC9C", "#F39C12", "#2980B9", "#D35400", "#27AE60",
]


def get_couleur(label: str) -> str:
    """Retourne la couleur associée au label, ou une couleur de secours."""
    if label in COULEURS_REGIONS:
        return COULEURS_REGIONS[label]
    idx = abs(hash(label)) % len(_PALETTE_FALLBACK)
    return _PALETTE_FALLBACK[idx]


# ══════════════════════════════════════════════════════════════════════
# FORMAT "RÉGION" : processed_energy_data_*.csv
# ══════════════════════════════════════════════════════════════════════

def charger_region(nom_region: str, nom_fichier: str) -> pd.DataFrame:
    """
    Charge un fichier de données régionales.
    Colonnes attendues : dataid, year, month, day, hour, minute,
                         temp, grid, clim, chauffage
    """
    csv_path = BASE_DIR / "data" / nom_fichier
    if not csv_path.exists():
        print(f"    Fichier introuvable : {csv_path} - région ignorée.")
        return pd.DataFrame()

    cols = ["dataid", "year", "month", "day", "hour", "minute",
            "temp", "grid", "clim", "chauffage"]

    df = load_results_csv(str(csv_path), usecols=cols)

    try:
        _require_cols(df, cols)
    except ValueError as e:
        print(f"    {nom_region} : {e} - région ignorée.")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime({
        "year": df["year"], "month": df["month"], "day": df["day"],
        "hour": df["hour"], "minute": df["minute"],
    }, errors="coerce")

    n_nat = df["datetime"].isna().sum()
    if n_nat > 0:
        print(f"   {nom_region} : {n_nat} horodatages invalides → exclus")
        df = df.dropna(subset=["datetime"])

    for col in ["grid", "clim", "chauffage", "temp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["clim"]      = df["clim"].fillna(0)
    df["chauffage"] = df["chauffage"].fillna(0)
    df = df.dropna(subset=["grid"])

    if DATAID is not None:
        df = df[df["dataid"] == DATAID]

    df = df.set_index("datetime").sort_index()
    df["saison"] = df["month"].map(MAP_SAISON)
    df["region"] = nom_region
    df["date"]   = df.index.normalize()

    print(f"  {nom_region} : {len(df):,} observations chargées")
    return df


# ══════════════════════════════════════════════════════════════════════
# FORMAT "DÉSAGRÉGATION" : resultats_desagregation_*.csv
# ══════════════════════════════════════════════════════════════════════

def charger_desagregation(csv_path: str | Path) -> pd.DataFrame:
    """
    Charge un fichier de résultats de désagrégation.
    Colonnes attendues : timestamp, P_total, T_ext, P_reel_clim
    (+ p_BASE, P_estime_clim, o_climatisation, aggregate_estimated optionnelles)

    Le dataid est extrait du nom de fichier :
      resultats_desagregation_<dataid>_<date>_<periode>.csv

    Retourne un DataFrame avec les mêmes colonnes que charger_region() :
      clim, grid, temp, chauffage, dataid, saison, region, date
      year, month, day, hour, minute
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"    Fichier introuvable : {csv_path}")
        return pd.DataFrame()

    # Extraire dataid depuis le nom de fichier
    parts = csv_path.stem.split("_")
    try:
        dataid = int(parts[2])
    except (IndexError, ValueError):
        dataid = 0
        print(f"    Impossible d'extraire le dataid depuis '{csv_path.name}', dataid=0 utilisé.")

    df = load_results_csv(str(csv_path))

    try:
        _require_cols(df, ["timestamp", "P_total", "T_ext", "P_reel_clim"])
    except ValueError as e:
        print(f"    {csv_path.name} : {e} - fichier ignoré.")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_nat = df["datetime"].isna().sum()
    if n_nat > 0:
        print(f"   {csv_path.name} : {n_nat} horodatages invalides → exclus")
        df = df.dropna(subset=["datetime"])

    for col in ["P_total", "T_ext", "P_reel_clim"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["P_total"])

    # Normalisation des noms de colonnes → même interface que charger_region()
    df = df.rename(columns={
        "P_total"    : "grid",
        "T_ext"      : "temp",
        "P_reel_clim": "clim",
    })
    df["clim"]      = df["clim"].fillna(0)
    df["chauffage"] = 0.0  # non disponible dans ce format
    df["dataid"]    = dataid

    df = df.set_index("datetime").sort_index()

    # Colonnes temporelles extraites de l'index
    df["year"]   = df.index.year
    df["month"]  = df.index.month
    df["day"]    = df.index.day
    df["hour"]   = df.index.hour
    df["minute"] = df.index.minute

    df["saison"] = df["month"].map(MAP_SAISON)
    df["region"] = str(dataid)
    df["date"]   = df.index.normalize()

    print(f"  Client {dataid} ({csv_path.stem.split('_')[3]}) : {len(df):,} observations chargées")
    return df


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE UNIFIÉ
# ══════════════════════════════════════════════════════════════════════

def charger_sources(
    mode: str,
    fichiers_desagregation: Optional[list] = None,
) -> dict:
    """
    Retourne un dict {label: DataFrame} selon le mode choisi.

    mode="region"
        Charge les régions définies dans REGIONS.
        Exemple : {"Austin": df_austin, "New York": df_ny, ...}

    mode="desagregation"
        Charge les fichiers listés dans fichiers_desagregation et les combine
        en un seul DataFrame (tous les clients ensemble).
        Si la liste est vide, charge tous les fichiers du dossier data/.
        Exemple : {"Désagrégation": df_tous_clients}
    """
    if mode == "region":
        dfs = {}
        for region, fichier in REGIONS.items():
            df = charger_region(region, fichier)
            if not df.empty:
                dfs[region] = df
        return dfs

    elif mode == "desagregation":
        if fichiers_desagregation:
            chemins = [BASE_DIR / "data" / f for f in fichiers_desagregation]
        else:
            chemins = sorted((BASE_DIR / "data").glob("resultats_desagregation_*.csv"))

        fragments = []
        for chemin in chemins:
            df = charger_desagregation(chemin)
            if not df.empty:
                fragments.append(df)

        if not fragments:
            return {}

        combined = pd.concat(fragments).sort_index()
        combined["region"] = "Désagrégation"
        n_clients = combined["dataid"].nunique()
        print(f"\n  {n_clients} clients combinés — {len(combined):,} observations au total")
        return {"Désagrégation": combined}

    else:
        raise ValueError(f"Mode inconnu : '{mode}'. Utiliser 'region' ou 'desagregation'.")
