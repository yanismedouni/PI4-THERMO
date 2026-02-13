"""
Code pour importer les données et les prétraiter.

Created on Fri Feb 12 2026
@author: catherinehenri
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration du prétraitement.

    Attributs
    ---------
    id_col : str
        Nom de la colonne d'identifiant client.
    ts_col : str
        Nom de la colonne temporelle.
    """
    id_col: str = "dataid"
    ts_col: str = "local_15min"



def load_data(path: str, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Charge un fichier CSV.

    Paramètres
    ----------
    path : str
        Chemin du fichier CSV.
    usecols : list[str] | None
        Colonnes à charger. Si None, toutes les colonnes sont chargées.

    Retours
    -------
    pd.DataFrame
        DataFrame chargé.
    """
    try:
        return pd.read_csv(path, usecols=usecols, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, usecols=usecols, encoding="latin-1")



def parse_naive_datetime_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convertit une colonne d'horodatages en datetime pandas naïf.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée.
    col : str
        Colonne à convertir.

    Retours
    -------
    pd.DataFrame
        Copie de df avec `col` convertie en datetime (NaT si invalide).
    """
    s = df[col].astype(str).str.strip()
    s = s.str.replace("T", " ", regex=False)
    s = s.str.replace(r"\.\d{1,6}", "", regex=True)
    s = s.str.replace(r"(Z|z)$", "", regex=True)
    s = s.str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True).str.strip()

    out = df.copy()
    out[col] = pd.to_datetime(s, errors="coerce")
    return out


def clip_negative_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Convertit des colonnes en numérique et tronque les valeurs négatives à zéro.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée.
    cols : Iterable[str]
        Colonnes à convertir et tronquer.

    Retours
    -------
    pd.DataFrame
        Copie de df avec colonnes traitées.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lower=0)
    return out


def filter_by_ids(df: pd.DataFrame, id_col: str, ids: Iterable[int]) -> pd.DataFrame:
    """
    Filtre un DataFrame en conservant seulement un ensemble d'identifiants.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée.
    id_col : str
        Colonne identifiant.
    ids : Iterable[int]
        Identifiants à conserver.

    Retours
    -------
    pd.DataFrame
        DataFrame filtré.
    """
    ids_list = list(ids)
    return df[df[id_col].isin(ids_list)].copy()


def filter_by_months(df: pd.DataFrame, ts_col: str, months: Optional[Iterable[int]]) -> pd.DataFrame:
    """
    Filtre un DataFrame selon les mois disponibles.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée.
    ts_col : str
        Colonne temporelle (datetime).
    months : Iterable[int] | None
        Mois à conserver (1-12). Si None, aucun filtrage.

    Retours
    -------
    pd.DataFrame
        DataFrame filtré.
    """
    if months is None:
        return df
    months_list = list(months)
    out = df.copy()
    out = out.dropna(subset=[ts_col])
    return out[out[ts_col].dt.month.isin(months_list)].copy()


def compute_equipment_series_sum(
    df: pd.DataFrame,
    cols: list[str],
    out_col: str,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Crée une colonne d'équipement comme somme de plusieurs canaux.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée.
    cols : list[str]
        Colonnes à sommer.
    out_col : str
        Colonne de sortie.
    min_count : int
        Paramètre `min_count` de pandas pour éviter de fabriquer des zéros.

    Retours
    -------
    pd.DataFrame
        Copie de df avec `out_col` ajouté.
    """
    out = df.copy()
    existing = [c for c in cols if c in out.columns]
    if not existing:
        out[out_col] = pd.Series([pd.NA] * len(out), index=out.index, dtype="float64")
        return out
    tmp = [pd.to_numeric(out[c], errors="coerce") for c in existing]
    out[out_col] = pd.concat(tmp, axis=1).sum(axis=1, min_count=min_count)
    return out