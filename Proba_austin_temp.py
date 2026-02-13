#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évalue la probabilité que la moyenne des clients aient leur climatiseur (air1) 
à ON (0,01 KW), en fonction de la température.
Created on Fri Feb 12 2026
@author: catherinehenri
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import meteostat as ms

# -----------------------------
# Parsing timestamps naïfs
# -----------------------------
def parse_naive_datetime_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].astype(str).str.strip()
    s = s.str.replace("T", " ", regex=False)
    s = s.str.replace(r"\.\d{1,6}", "", regex=True)
    s = s.str.replace(r"(Z|z)$", "", regex=True)
    s = s.str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True).str.strip()
    df = df.copy()
    df[col] = pd.to_datetime(s, errors="coerce")
    return df

# -----------------------------
# Meteostat -> Températures 15 min (API 2.x)
# -----------------------------
def build_temp_15min_from_meteostat(point: ms.Point, start, end,
                                    tz_source: str = "UTC",
                                    tz_target: str = "America/Chicago",
                                    temp_col_out: str = "temp") -> pd.DataFrame:
    """
    Récupère les températures horaires Meteostat pour un point et une période,
    convertit le fuseau, enlève la timezone (naïf), déduplique,
    et densifie à une résolution de 15 minutes par interpolation.
    """
    stations = ms.stations.nearby(point, limit=4)
    ts_hour = ms.hourly(stations, start, end)
    df = ms.interpolate(ts_hour, point).fetch()
    if df.empty:
        raise ValueError("Aucune donnée Meteostat récupérée.")
    df.index = df.index.tz_localize(tz_source).tz_convert(tz_target).tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    col_temp_in = "temp" if "temp" in df.columns else ("t" if "t" in df.columns else None)
    if col_temp_in is None:
        raise ValueError("Colonne température introuvable dans Meteostat.")
    temp_15 = df[[col_temp_in]].rename(columns={col_temp_in: temp_col_out})
    temp_15 = temp_15.resample("15min").interpolate(method="time")
    return temp_15

# -----------------------------
# Calcul probabilité globale
# -----------------------------
def compute_global_proba_by_temp(df: pd.DataFrame, ids_a_garder: list[int],
                                 point: ms.Point, start, end,
                                 seuil_on: float = 0.01,
                                 bin_start: float = 10,
                                 bin_stop: float = 40,
                                 bin_step: float = 1) -> pd.DataFrame:
    df = df.copy()
    if not {"dataid", "local_15min", "air1"}.issubset(df.columns):
        raise ValueError("df doit contenir 'dataid', 'local_15min', 'air1'.")
    df = parse_naive_datetime_col(df, "local_15min")
    df = df.set_index("local_15min").sort_index()
    df = df[df["dataid"].isin(ids_a_garder)]
    df["air1"] = pd.to_numeric(df["air1"], errors="coerce").clip(lower=0)

    temp_15 = build_temp_15min_from_meteostat(point=point, start=start, end=end)
    df_w = df.join(temp_15, how="left")
    df_w["temp"] = pd.to_numeric(df_w["temp"], errors="coerce")
    df_w["clim_on"] = (df_w["air1"] > seuil_on).astype(int)

    bins_temp = np.arange(bin_start, bin_stop, bin_step)
    df_w["temp_bin"] = pd.cut(df_w["temp"], bins=bins_temp, include_lowest=True)

    proba_temp_global = (
        df_w.groupby("temp_bin", observed=True)["clim_on"]
            .mean()
            .reset_index()
            .rename(columns={"clim_on": "p_on"})
    )
    proba_temp_global["temp_center"] = proba_temp_global["temp_bin"].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    return proba_temp_global.sort_values("temp_center").reset_index(drop=True)

# -----------------------------
# Moyenne air1 par ID
# -----------------------------
def mean_air1_by_id(df: pd.DataFrame, ids_a_garder: list[int] | None = None,
                    clip_negative_to_zero: bool = True) -> pd.DataFrame:
    required = {"dataid", "air1"}
    if not required.issubset(df.columns):
        raise ValueError(f"Le DataFrame doit contenir {required}.")
    out = df.copy()
    if ids_a_garder is not None:
        out = out[out["dataid"].isin(ids_a_garder)]
    out["air1"] = pd.to_numeric(out["air1"], errors="coerce")
    if clip_negative_to_zero:
        out["air1"] = out["air1"].clip(lower=0)
    agg = (
        out.groupby("dataid", as_index=False)
           .agg(mean_air1_kW=("air1", "mean"),
                n_points=("air1", "count"))
           .sort_values("dataid")
           .reset_index(drop=True)
    )
    return agg

def filter_ids_with_nonzero_mean(df: pd.DataFrame, ids_input: list[int]) -> list[int]:
    tbl = mean_air1_by_id(df, ids_a_garder=ids_input, clip_negative_to_zero=True)
    tbl = tbl[pd.notna(tbl["mean_air1_kW"]) & (tbl["mean_air1_kW"] > 0)]
    return tbl["dataid"].tolist()

# -----------------------------
# MAIN
# -----------------------------
def main():
    csv_path = "/Users/catherinehenri/Desktop/Genie Elec/Session A2025/ELE8080/Dev/15minute_data_austin/15minute_data_austin.csv"
    usecols = ["dataid", "local_15min", "air1"]

    ids_init = [661, 1642, 2335, 2361, 2818, 3039, 3456, 3538, 4031, 4373, 4767, 5746]

    df = pd.read_csv(csv_path, usecols=usecols)

    ids_actifs = filter_ids_with_nonzero_mean(df, ids_init)
    print("IDs actifs (moyenne air1 > 0) :", ids_actifs)
    if not ids_actifs:
        raise ValueError("Aucun ID actif (moyenne air1 > 0).")

    austin = ms.Point(30.2672, -97.7431)
    start = pd.to_datetime(df["local_15min"].min(), errors="coerce")
    end   = pd.to_datetime(df["local_15min"].max(), errors="coerce")

    proba_temp_global = compute_global_proba_by_temp(
        df=df,
        ids_a_garder=ids_actifs,
        point=austin,
        start=start,
        end=end,
        seuil_on=0.01,
        bin_start=15,
        bin_stop=40,
        bin_step=1
    )

    print("\nProbabilité moyenne que la clim soit ON :")
    print(proba_temp_global)

    plt.figure(figsize=(9, 5))
    plt.plot(proba_temp_global["temp_center"], proba_temp_global["p_on"],
             marker="o", color="firebrick", linewidth=2)
    plt.title("Probabilité que la climatisation soit ON – Austin")
    plt.xlabel("Température extérieure (°C)")
    plt.ylabel("P(clim ON)")
    plt.yticks(np.arange(0, 1.0001, 0.05))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
