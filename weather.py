"""
Code pour importer les données et les prétraiter.

Created on Fri Feb 12 2026
@author: catherinehenri
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import meteostat as ms


@dataclass(frozen=True)
class RegionSpec:
    """
    Spécification d'une région.

    Attributs
    ---------
    name : str
        Nom de la région.
    point : ms.Point
        Coordonnées Meteostat.
    tz_target : str
        Fuseau horaire local du dataset (pour aligner la température).
    """
    name: str
    point: ms.Point
    tz_target: str


def build_temp_15min(
    point: ms.Point,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz_source: str = "UTC",
    tz_target: str = "America/New_York",
    out_col: str = "temp",
) -> pd.DataFrame:
    """
    Récupère la température Meteostat (horaire), aligne au fuseau local,
    et interpole à 15 minutes.

    Paramètres
    ----------
    point : ms.Point
        Coordonnées.
    start, end : pd.Timestamp
        Fenêtre temporelle.
    tz_source : str
        Fuseau présumé Meteostat.
    tz_target : str
        Fuseau cible.
    out_col : str
        Nom de colonne de sortie.

    Retours
    -------
    pd.DataFrame
        Index datetime naïf (15min), colonne `out_col`.

    Exceptions
    ----------
    ValueError
        Si aucune donnée n'est récupérée ou si la colonne température est absente.
    """
    stations = ms.stations.nearby(point, limit=4)
    ts_hour = ms.hourly(stations, start, end)
    df = ms.interpolate(ts_hour, point).fetch()
    if df.empty:
        raise ValueError("Aucune donnée Meteostat récupérée.")

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(tz_source)
    idx = idx.tz_convert(tz_target).tz_localize(None)
    df = df.copy()
    df.index = idx
    df = df[~df.index.duplicated(keep="first")]

    col_in = "temp" if "temp" in df.columns else ("t" if "t" in df.columns else None)
    if col_in is None:
        raise ValueError("Colonne température introuvable dans Meteostat.")

    out = df[[col_in]].rename(columns={col_in: out_col})
    out = out.resample("15min").interpolate(method="time")
    return out


def build_temp_map_by_region(
    regions: Dict[str, RegionSpec],
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz_source: str = "UTC",
    out_col: str = "temp",
) -> Dict[str, pd.DataFrame]:
    """
    Construit un dictionnaire {region_name: temperature_15min}.

    Paramètres
    ----------
    regions : dict[str, RegionSpec]
        Régions.
    start, end : pd.Timestamp
        Fenêtre temporelle globale.
    tz_source : str
        Fuseau source Meteostat.
    out_col : str
        Nom de la colonne température.

    Retours
    -------
    dict[str, pd.DataFrame]
        Températures 15 min par région.
    """
    out: Dict[str, pd.DataFrame] = {}
    for name, spec in regions.items():
        out[name] = build_temp_15min(
            point=spec.point,
            start=start,
            end=end,
            tz_source=tz_source,
            tz_target=spec.tz_target,
            out_col=out_col,
        )
    return out
    