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


def _to_local_naive_index(
    idx: pd.DatetimeIndex,
    tz_source: str,
    tz_target: str,
) -> pd.DatetimeIndex:
    if idx.tz is None:
        idx = idx.tz_localize(tz_source)
    return idx.tz_convert(tz_target).tz_localize(None)


def build_temp_15min(
    point: ms.Point,
    start,
    end,
    tz_source: str = "UTC",
    tz_target: str = "America/New_York",
    out_col: str = "temp",
    station_limit: int = 6,
) -> pd.DataFrame:
    """
    Meteostat 2.x: stations.nearby -> hourly -> interpolate -> resample 15min

    Robustesse:
    - si interpolate() échoue (elevation/redirect), on retombe sur la station la + proche (sans interpolate)
    """
    stations = ms.stations.nearby(point, limit=station_limit)
    ts_hour = ms.hourly(stations, start, end)

    # 1) Essai interpolation (meilleure qualité)
    try:
        df = ms.interpolate(ts_hour, point).fetch()
    except Exception:
        # 2) Fallback: prendre la station la plus proche (sans interpolation)
        df = ts_hour.fetch()

    if df is None or df.empty:
        return pd.DataFrame(columns=[out_col])

    # timezone -> naïf local
    df.index = _to_local_naive_index(df.index, tz_source=tz_source, tz_target=tz_target)
    df = df[~df.index.duplicated(keep="first")]

    # colonne température
    col_in = "temp" if "temp" in df.columns else ("t" if "t" in df.columns else None)
    if col_in is None:
        return pd.DataFrame(columns=[out_col])

    out = df[[col_in]].rename(columns={col_in: out_col})

    # 15 min
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
    