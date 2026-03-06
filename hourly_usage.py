#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code pour trouver la consommation horaire pour une saison, 1 mois et 1 semaine. 

Created on Fri Feb 19 2026
@author: catherinehenri
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from weather import RegionSpec, build_temp_map_by_region  # pas indispensable ici, mais garde la cohérence
from probability import EquipmentSpec, ProbaRunConfig


@dataclass(frozen=True)
class HourlyUsageResult:
    """
    Résultats pour l'analyse P(ON | heure).

    Attributs
    ---------
    season_hourly : pd.DataFrame
        P(ON) par heure sur toute la saison (mois applicables).
    monthly_hourly : pd.DataFrame
        P(ON) par heure et par mois (dans la saison).
    peak_week_hourly : pd.DataFrame
        P(ON) par heure pour la semaine sélectionnée (milieu de saison, utilisation maximale).
    peak_week_window : tuple[pd.Timestamp, pd.Timestamp]
        (début, fin) de la semaine sélectionnée (bornes inclusives côté début).
    """
    season_hourly: pd.DataFrame
    monthly_hourly: pd.DataFrame
    peak_week_hourly: pd.DataFrame
    peak_week_window: Tuple[pd.Timestamp, pd.Timestamp]


def _make_on_series(energy: pd.Series, seuil_on: float, use_ge: bool) -> pd.Series:
    if use_ge:
        return (energy >= seuil_on).astype(int)
    return (energy > seuil_on).astype(int)


def _hourly_stats_equitable_by_id(
    df: pd.DataFrame,
    id_col: str,
    ts_col: str,
    on_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Calcule P(ON) ainsi : on calcule d'abord par ID, puis on moyenne sur les IDs.

    Paramètres
    ----------
    df : pd.DataFrame
        Données filtrées, avec colonnes id_col, ts_col, on_col.
    id_col : str
        Colonne identifiant client.
    ts_col : str
        Colonne datetime.
    on_col : str
        Colonne binaire ON/OFF.
    group_cols : list[str]
        Colonnes de groupement supplémentaires (ex: ["hour"] ou ["month","hour"]).

    Retours
    -------
    pd.DataFrame
        Contient group_cols + p_on_mean + n_ids + n_points_total.
    """
    work = df[[id_col, ts_col, on_col] + group_cols].copy()

    per_id = (
        work.groupby([id_col] + group_cols, observed=True)[on_col]
        .agg(p_on="mean", n_points="size")
        .reset_index()
    )

    agg = (
        per_id.groupby(group_cols, observed=True)
        .agg(
            p_on_mean=("p_on", "mean"),
            n_ids=(id_col, "nunique"),
            n_points_total=("n_points", "sum"),
        )
        .reset_index()
    )

    return agg


def _select_peak_week_from_available_months(
    df_season: pd.DataFrame,
    ts_col: str,
    on_col: str,
    min_coverage_days: int = 4,
    window_days: int = 7,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Choisit la fenêtre de 7 jours avec la plus haute moyenne ON,
    en cherchant uniquement dans les mois réellement présents dans df_season.

    Retour: (start, end) avec end exclusif.
    """
    work = df_season.copy()
    work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
    work = work.dropna(subset=[ts_col])

    if work.empty:
        raise ValueError("Saison vide : impossible de sélectionner une semaine.")

    months_present = sorted(work[ts_col].dt.month.dropna().unique().tolist())
    if not months_present:
        raise ValueError("Aucun mois disponible pour sélectionner une semaine.")

    day_start = work[ts_col].min().normalize()
    day_end = work[ts_col].max().normalize()
    candidate_starts = pd.date_range(day_start, day_end, freq="D")

    best_start = None
    best_score = -np.inf

    for s in candidate_starts:
        if s.month not in months_present:
            continue

        e = s + pd.Timedelta(days=window_days)
        sub = work[(work[ts_col] >= s) & (work[ts_col] < e)]

        if sub.empty:
            continue

        days_cov = pd.to_datetime(sub[ts_col]).dt.date.nunique()
        if days_cov < min_coverage_days:
            continue

        score = pd.to_numeric(sub[on_col], errors="coerce").mean()
        if pd.notna(score) and float(score) > best_score:
            best_score = float(score)
            best_start = s

    if best_start is None:
        s = day_start
        while s.month not in months_present and s <= day_end:
            s += pd.Timedelta(days=1)
        best_start = s

    return best_start, best_start + pd.Timedelta(days=window_days)



def estimate_hourly_usage_multi_region(
    df: pd.DataFrame,
    regions: Dict[str, RegionSpec],
    equipment: EquipmentSpec,
    cfg: ProbaRunConfig,
    id_to_region_col: Optional[str] = None,
) -> HourlyUsageResult:
    """
    Estime P(ON | heure) pour un équipement, pour:
    1) Saison (mois applicables)
    2) Par mois dans la saison
    3) Une semaine du milieu de saison avec P(ON) la plus élevée

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir cfg.id_col, cfg.ts_col, cfg.region_col, equipment.energy_col.
    regions : dict[str, RegionSpec]
    equipment : EquipmentSpec
        Décrit l'équipement (seuil, mois applicables).
    cfg : ProbaRunConfig
        Configuration d'exécution (colonne id/temps/région, per_client, dropna...).
    id_to_region_col : str | None
        Non nécessaire si cfg.region_col est déjà présent.

    Retours
    -------
    HourlyUsageResult
    """
    required = {cfg.id_col, cfg.ts_col, cfg.region_col, equipment.energy_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes requises manquantes : {required}")

    work = df[[cfg.id_col, cfg.ts_col, cfg.region_col, equipment.energy_col]].copy()
    work[cfg.ts_col] = pd.to_datetime(work[cfg.ts_col], errors="coerce")
    work = work.dropna(subset=[cfg.ts_col, cfg.region_col])
    work[equipment.energy_col] = pd.to_numeric(work[equipment.energy_col], errors="coerce").clip(lower=0)

    if equipment.months is not None:
        months_list = list(equipment.months)
        work = work[work[cfg.ts_col].dt.month.isin(months_list)].copy()
        if work.empty:
            raise ValueError("Aucune donnée après filtrage par mois (saison).")

    work["on"] = _make_on_series(work[equipment.energy_col], equipment.seuil_on, equipment.use_ge)
    work["hour"] = work[cfg.ts_col].dt.hour
    work["month"] = work[cfg.ts_col].dt.month

    if cfg.per_client:
        season_hourly = _hourly_stats_equitable_by_id(
            df=work, id_col=cfg.id_col, ts_col=cfg.ts_col, on_col="on", group_cols=["hour"]
        )
        monthly_hourly = _hourly_stats_equitable_by_id(
            df=work, id_col=cfg.id_col, ts_col=cfg.ts_col, on_col="on", group_cols=["month", "hour"]
        )
    else:
        season_hourly = (
            work.groupby("hour", observed=True)["on"]
            .agg(p_on_mean="mean", n_points_total="size")
            .reset_index()
        )
        season_hourly["n_ids"] = work[cfg.id_col].nunique()

        monthly_hourly = (
            work.groupby(["month", "hour"], observed=True)["on"]
            .agg(p_on_mean="mean", n_points_total="size")
            .reset_index()
        )
        monthly_hourly["n_ids"] = work[cfg.id_col].nunique()

    season_hourly = season_hourly.sort_values("hour").reset_index(drop=True)
    monthly_hourly = monthly_hourly.sort_values(["month", "hour"]).reset_index(drop=True)

    w_start, w_end = _select_peak_week_from_available_months(
        df_season=work,
        ts_col=cfg.ts_col,
        on_col="on",
        min_coverage_days=4,
        window_days=7,
    )

    work_week = work[(work[cfg.ts_col] >= w_start) & (work[cfg.ts_col] < w_end)].copy()
    if work_week.empty:
        raise ValueError("Semaine sélectionnée vide (vérifie la couverture temporelle).")

    if cfg.per_client:
        peak_week_hourly = _hourly_stats_equitable_by_id(
            df=work_week, id_col=cfg.id_col, ts_col=cfg.ts_col, on_col="on", group_cols=["hour"]
        )
    else:
        peak_week_hourly = (
            work_week.groupby("hour", observed=True)["on"]
            .agg(p_on_mean="mean", n_points_total="size")
            .reset_index()
        )
        peak_week_hourly["n_ids"] = work_week[cfg.id_col].nunique()

    peak_week_hourly = peak_week_hourly.sort_values("hour").reset_index(drop=True)

    return HourlyUsageResult(
        season_hourly=season_hourly,
        monthly_hourly=monthly_hourly,
        peak_week_hourly=peak_week_hourly,
        peak_week_window=(w_start, w_end),
    )
