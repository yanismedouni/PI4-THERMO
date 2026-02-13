"""
Code pour importer les données et les prétraiter.

Created on Fri Feb 12 2026
@author: catherinehenri
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from weather import RegionSpec, build_temp_map_by_region


@dataclass(frozen=True)
class EquipmentSpec:
    """
    Spécification d'un équipement (signal d'énergie + paramètres de proba).

    Attributs
    ---------
    name : str
        Nom de l'équipement.
    energy_col : str
        Colonne énergie (kW) à utiliser pour ON/OFF.
    seuil_on : float
        Seuil (kW) pour déclarer ON.
    months : Iterable[int] | None
        Mois à analyser (1-12). None = tous.
    bin_start, bin_stop, bin_step : float
        Paramètres des bins de température.
    use_ge : bool
        Si True, ON = energy >= seuil_on, sinon ON = energy > seuil_on.
    """
    name: str
    energy_col: str
    seuil_on: float
    months: Optional[Iterable[int]]
    bin_start: float
    bin_stop: float
    bin_step: float
    use_ge: bool = True


@dataclass(frozen=True)
class ProbaRunConfig:
    """
    Configuration de l'estimation.

    Attributs
    ---------
    id_col : str
        Colonne identifiant client.
    ts_col : str
        Colonne temporelle.
    region_col : str
        Colonne région (ou zone) associée à chaque client.
    temp_col : str
        Nom de colonne température.
    per_client : bool
        Si True, moyenne équitable par client. Sinon, moyenne pondérée par points.
    dropna_temp : bool
        Si True, retire les lignes sans température.
    """
    id_col: str = "dataid"
    ts_col: str = "local_15min"
    region_col: str = "region"
    temp_col: str = "temp"
    per_client: bool = True
    dropna_temp: bool = True


def attach_region_column(
    df: pd.DataFrame,
    id_col: str,
    id_to_region: Dict[int, str],
    region_col: str = "region",
    default_region: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ajoute une colonne région à partir d'un mapping id -> région.

    Paramètres
    ----------
    df : pd.DataFrame
        Données.
    id_col : str
        Colonne identifiant.
    id_to_region : dict[int, str]
        Mapping identifiant -> région.
    region_col : str
        Colonne de sortie.
    default_region : str | None
        Valeur par défaut si un id est absent du mapping (None = NaN).

    Retours
    -------
    pd.DataFrame
        Copie avec colonne région.
    """
    out = df.copy()
    out[region_col] = out[id_col].map(id_to_region)
    if default_region is not None:
        out[region_col] = out[region_col].fillna(default_region)
    return out


def estimate_proba_on_by_temp_multi_region(
    df: pd.DataFrame,
    regions: Dict[str, RegionSpec],
    equipment: EquipmentSpec,
    cfg: ProbaRunConfig,
    tz_source_meteostat: str = "UTC",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Estime P(ON | bin(T)) pour un équipement, en tenant compte de plusieurs régions.

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir cfg.id_col, cfg.ts_col, cfg.region_col, equipment.energy_col.
        La colonne temps doit être datetime (ou convertible).
    regions : dict[str, RegionSpec]
        Spécifications Meteostat des régions.
    equipment : EquipmentSpec
        Paramètres de l'équipement (seuil, bins, mois).
    cfg : ProbaRunConfig
        Paramètres d'exécution (pondération, colonnes).
    tz_source_meteostat : str
        Fuseau source Meteostat.
    verbose : bool
        Affiche des métriques de contrôle.

    Retours
    -------
    pd.DataFrame
        Colonnes :
        - temp_bin
        - temp_center
        - p_on_mean (proba moyenne sur clients ou points)
        - n_ids
        - n_points_total
    """
    required = {cfg.id_col, cfg.ts_col, cfg.region_col, equipment.energy_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes requises manquantes : {required}")

    work = df[[cfg.id_col, cfg.ts_col, cfg.region_col, equipment.energy_col]].copy()
    work[cfg.ts_col] = pd.to_datetime(work[cfg.ts_col], errors="coerce")
    work = work.dropna(subset=[cfg.ts_col])
    work[equipment.energy_col] = pd.to_numeric(work[equipment.energy_col], errors="coerce").clip(lower=0)
    work = work.dropna(subset=[cfg.region_col])

    if equipment.months is not None:
        months_list = list(equipment.months)
        work = work[work[cfg.ts_col].dt.month.isin(months_list)].copy()
        if work.empty:
            raise ValueError("Aucune donnée après filtrage par mois.")

    start_global = work[cfg.ts_col].min()
    end_global = work[cfg.ts_col].max()

    temp_map = build_temp_map_by_region(
        regions=regions,
        start=start_global,
        end=end_global,
        tz_source=tz_source_meteostat,
        out_col=cfg.temp_col,
    )

    bins = np.arange(equipment.bin_start, equipment.bin_stop + equipment.bin_step, equipment.bin_step)

    per_id_rows: list[pd.DataFrame] = []
    for id_, g in work.groupby(cfg.id_col):
        region = g[cfg.region_col].iloc[0]
        if region not in temp_map:
            continue

        g = g.sort_values(cfg.ts_col).set_index(cfg.ts_col)
        temp_df = temp_map[region].loc[g.index.min(): g.index.max()]
        joined = g.join(temp_df, how="left")
        joined[cfg.temp_col] = pd.to_numeric(joined[cfg.temp_col], errors="coerce")

        if cfg.dropna_temp:
            joined = joined.dropna(subset=[cfg.temp_col])

        if joined.empty:
            continue

        energy = pd.to_numeric(joined[equipment.energy_col], errors="coerce")

        if equipment.use_ge:
            joined["on"] = (energy >= equipment.seuil_on).astype(int)
        else:
            joined["on"] = (energy > equipment.seuil_on).astype(int)

        joined["temp_bin"] = pd.cut(joined[cfg.temp_col], bins=bins, include_lowest=True)

        stat = (
            joined.groupby("temp_bin", observed=True)["on"]
            .agg(p_on="mean", n_points="size")
            .reset_index()
        )
        stat["dataid"] = id_
        stat["region"] = region
        per_id_rows.append(stat)

        if verbose:
            miss_t = g.join(temp_df, how="left")[cfg.temp_col].isna().mean()
            pct_on = joined["on"].mean()
            n = len(joined)
            print(f"[INFO] {equipment.name} | ID {id_} | region={region} | N={n} | temp_NA={miss_t:.1%} | pct(ON)={pct_on:.1%}")

    if not per_id_rows:
        raise ValueError("Aucun ID n'a produit de statistiques (vérifie mapping régions/données).")

    all_ids = pd.concat(per_id_rows, ignore_index=True)

    if cfg.per_client:
        agg = (
            all_ids.groupby("temp_bin", observed=True)
            .agg(
                p_on_mean=("p_on", "mean"),
                n_ids=("dataid", "nunique"),
                n_points_total=("n_points", "sum"),
            )
            .reset_index()
        )
    else:
        all_ids["w"] = all_ids["n_points"].astype(float)
        agg = (
            all_ids.groupby("temp_bin", observed=True)
            .apply(lambda x: pd.Series({
                "p_on_mean": np.average(x["p_on"], weights=x["w"]) if x["w"].sum() > 0 else np.nan,
                "n_ids": x["dataid"].nunique(),
                "n_points_total": x["n_points"].sum(),
            }))
            .reset_index()
        )

    agg["temp_center"] = agg["temp_bin"].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    return agg.sort_values("temp_center").reset_index(drop=True)