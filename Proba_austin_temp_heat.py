
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probabilité que le chauffage (furnace1 + furnace2) soit ON en fonction de la
température extérieure (Austin, America/Chicago) – API Meteostat 2.x.

Created on Fri Jan 9 2026
@author: catherinehenri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import meteostat as ms  

# -----------------------------
# Parsing: timestamps naïfs
# -----------------------------
def parse_naive_datetime_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].astype(str).str.strip()
    s = s.str.replace("T", " ", regex=False)
    s = s.str.replace(r"\.\d{1,6}", "", regex=True)
    s = s.str.replace(r"(Z|z)$", "", regex=True)
    s = s.str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True).str.strip()
    out = df.copy()
    out[col] = pd.to_datetime(s, errors="coerce")
    return out

def _to_local_naive_index(idx: pd.DatetimeIndex, tz_target: str) -> pd.DatetimeIndex:
    """Si l'index est tz-aware, convertit vers tz_target puis retire le tz (naïf)."""
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        return idx.tz_convert(tz_target).tz_localize(None)
    return idx  # déjà naïf

# -----------------------------
# Meteostat → Températures 15 min
# -----------------------------

def build_temp_15min_from_meteostat_ms2(point: ms.Point,
                                        start, end,
                                        tz_target: str = "America/Chicago",
                                        temp_col_out: str = "temp") -> pd.DataFrame:

    stations = ms.stations.nearby(point, limit=4)
    ts_hour = ms.hourly(stations, start, end)            
    df = ms.interpolate(ts_hour, point).fetch()      

    if df.empty:
        return pd.DataFrame(columns=[temp_col_out])

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz_target).tz_localize(None)
    else:
        df.index = df.index.tz_convert(tz_target).tz_localize(None)

    df = df[~df.index.duplicated(keep="first")]

    if ms.Parameter.TEMP in df.columns:
        col_temp_in = ms.Parameter.TEMP
    elif "temp" in df.columns:
        col_temp_in = "temp"
    elif "t" in df.columns:
        col_temp_in = "t"
    else:
        return pd.DataFrame(columns=[temp_col_out])

    temp_15 = (df[[col_temp_in]]
               .rename(columns={col_temp_in: temp_col_out})
               .resample("15min")
               .interpolate(method="time"))
    return temp_15


# -----------------------------
# Proba P(ON | Temp) générique
# -----------------------------
def compute_global_proba_by_temp_generic(
    df: pd.DataFrame,
    ids_a_garder: list[int],
    point: ms.Point,
    start, end,
    energy_col: str,           # "furnace_total" ou "air1"
    seuil_on: float,           # seuil ON (kW)
    bin_start: float,
    bin_stop: float,
    bin_step: float,
    months: tuple[int, ...] | list[int] | None = None,
    tz_target: str = "America/Chicago",
    temp_col_out: str = "temp"
) -> pd.DataFrame:

    df = df.copy()
    required = {"dataid", "local_15min", energy_col}
    if not required.issubset(df.columns):
        raise ValueError(f"df doit contenir {required}.")

    df = parse_naive_datetime_col(df, "local_15min")
    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce").clip(lower=0)
    df = df[df["dataid"].isin(ids_a_garder)].set_index("local_15min").sort_index()

    if months:
        df = df[df.index.month.isin(months)]
        if df.empty:
            raise ValueError("Aucune donnée après filtrage par mois.")

    # Température 15 min (Meteostat 2.x)
    temp_15 = build_temp_15min_from_meteostat_ms2(point=point, start=start, end=end,
                                                  tz_target=tz_target, temp_col_out=temp_col_out)
    df_w = df.join(temp_15, how="left")
    df_w[temp_col_out] = pd.to_numeric(df_w[temp_col_out], errors="coerce")

    # Binarisation ON/OFF puis binning température
    df_w["equip_on"] = (df_w[energy_col] > seuil_on).astype(int)
    bins_temp = np.arange(bin_start, bin_stop, bin_step)
    df_w["temp_bin"] = pd.cut(df_w[temp_col_out], bins=bins_temp, include_lowest=True)

    proba_temp_global = (
        df_w.groupby("temp_bin", observed=True)["equip_on"]
            .mean()
            .reset_index()
            .rename(columns={"equip_on": "p_on"})
    )
    proba_temp_global["temp_center"] = proba_temp_global["temp_bin"].apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )
    return proba_temp_global.sort_values("temp_center").reset_index(drop=True)

# -----------------------------
# Agrégations & filtre d'IDs actifs
# -----------------------------
def mean_energy_by_id(df: pd.DataFrame,
                      energy_col: str,
                      ids_a_garder: list[int] | None = None,
                      clip_negative_to_zero: bool = True) -> pd.DataFrame:
    required = {"dataid", energy_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Le DataFrame doit contenir {required}.")
    out = df.copy()
    if ids_a_garder is not None:
        out = out[out["dataid"].isin(ids_a_garder)]
    out[energy_col] = pd.to_numeric(out[energy_col], errors="coerce")
    if clip_negative_to_zero:
        out[energy_col] = out[energy_col].clip(lower=0)
    return (
        out.groupby("dataid", as_index=False)
           .agg(mean_kW=(energy_col, "mean"),
                n_points=(energy_col, "count"))
           .sort_values("dataid")
           .reset_index(drop=True)
    )

def filter_ids_by_mean_threshold(df: pd.DataFrame,
                                 energy_col: str,
                                 ids_input: list[int],
                                 threshold_kW: float = 0.005,
                                 comparator: str = "gt") -> list[int]:
    tbl = mean_energy_by_id(df, energy_col=energy_col, ids_a_garder=ids_input)
    if comparator == "gt":
        cond = (pd.notna(tbl["mean_kW"])) & (tbl["mean_kW"] >= threshold_kW)
    elif comparator == "lt":
        cond = (pd.notna(tbl["mean_kW"])) & (tbl["mean_kW"] < threshold_kW)
    else:
        raise ValueError("comparator doit être 'gt' ou 'lt'.")
    return tbl.loc[cond, "dataid"].tolist()


def compute_proportion_clients_actifs(df, ids_a_garder, energy_col, seuil_on, ratio_threshold,
                                      bin_start, bin_stop, bin_step, months=None):
    """
    Calcule la proportion de clients actifs par bin de température.
    Un client est actif dans un bin si >= ratio_threshold du temps ON.
    """
    df = df.copy()
    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce").clip(lower=0)
    df = df[df["dataid"].isin(ids_a_garder)]

    if "local_15min" in df.columns:
        df = df.set_index("local_15min")
    else:
        # Si la colonne n'existe pas, on exige un DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise KeyError("Le DataFrame doit avoir 'local_15min' en colonne ou un DatetimeIndex.")
    df = df.sort_index()

    # Filtre de mois, s'il est fourni
    if months:
        df = df[df.index.month.isin(months)]
        if df.empty:
            raise ValueError("Aucune donnée après filtrage par mois.")

    if "temp" not in df.columns:
        raise KeyError("Colonne 'temp' absente. Assure la jointure Meteostat avant l'appel.")

    bins_temp = np.arange(bin_start, bin_stop, bin_step)
    results = []

    for id_ in ids_a_garder:
        sub = df[df["dataid"] == id_]
        if sub.empty:
            continue
        # Binarisation ON/OFF
        sub["equip_on"] = (sub[energy_col] > seuil_on).astype(int)
        # Binning température
        sub["temp_bin"] = pd.cut(sub["temp"], bins=bins_temp, include_lowest=True)
        # Calcul ratio par bin
        stat = sub.groupby("temp_bin", observed=True)["equip_on"].agg(["mean", "count"])
        stat["is_active"] = (stat["mean"] >= ratio_threshold).astype(int)
        stat["dataid"] = id_
        results.append(stat.reset_index())

    if not results:
        raise ValueError("Aucun ID n'a produit de statistiques dans compute_proportion_clients_actifs.")

    all_ids = pd.concat(results, ignore_index=True)
    agg = (all_ids.groupby("temp_bin", observed=True)
           .agg(n_clients_active=("is_active", "sum"),
                n_clients_total=("dataid", "nunique"))
           .reset_index())
    agg["p_clients_active"] = agg["n_clients_active"] / agg["n_clients_total"]
    agg["temp_center"] = agg["temp_bin"].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    return agg.sort_values("temp_center").reset_index(drop=True)


# -----------------------------
# MAIN – Austin : chauffage
# -----------------------------
def main():
    # ⚠️ CSV Austin: nécessite furnace1 & furnace2
    csv_path = "/Users/catherinehenri/Desktop/Genie Elec/Session A2025/ELE8080/Dev/15minute_data_austin/15minute_data_austin.csv"
    usecols = ["dataid", "local_15min", "furnace1", "furnace2"]

    # IDs Austin (adapte si besoin)
    ids_init = [661, 1642, 2335, 2361, 2818, 3039, 3456, 3538, 4031, 4373, 4767, 5746]

    # 1) Charger
    df = pd.read_csv(csv_path, usecols=usecols)

    # 2) furnace_total robuste aux NaN (option A)
    f1 = pd.to_numeric(df["furnace1"], errors="coerce").clip(lower=0)
    f2 = pd.to_numeric(df["furnace2"], errors="coerce").clip(lower=0)
    df["furnace_total"] = pd.concat([f1, f2], axis=1).sum(axis=1, min_count=1)

    # 3) Période & point Meteostat (Austin, America/Chicago)
    start = pd.to_datetime(df["local_15min"].min(), errors="coerce")
    end   = pd.to_datetime(df["local_15min"].max(), errors="coerce")
    austin = ms.Point(30.2672, -97.7431)

    # 4) Mois froids (ajuste selon ta couverture Austin)
    months_cold = (10, 11, 12, 1, 2, 3)

    # 5) IDs actifs chauffage (moyenne sur mois froids)
    df_parsed = parse_naive_datetime_col(df, "local_15min")
    df_parsed["furnace_total"] = pd.to_numeric(df_parsed["furnace_total"], errors="coerce").clip(lower=0)
    df_parsed = df_parsed.set_index("local_15min").sort_index()
    df_cold = df_parsed[df_parsed.index.month.isin(months_cold)].reset_index()

    thr_mean_heat = 0.005   # seuil "actif"
    ids_actifs_heat = filter_ids_by_mean_threshold(
        df=df_cold, energy_col="furnace_total",
        ids_input=ids_init, threshold_kW=thr_mean_heat, comparator="gt"
    )
    print(f"IDs actifs chauffage (mean furnace_total ≥ {thr_mean_heat} kW) :", ids_actifs_heat)
    if not ids_actifs_heat:
        raise ValueError("Aucun ID actif chauffage selon le seuil de moyenne & mois sélectionnés.")

    # 6) Proba P(chauffage ON | T)
    seuil_on_heat = 0.2     # teste aussi 0.3 / 0.5
    bins_heat = dict(bin_start=-5, bin_stop=25, bin_step=1)

    proba_heat = compute_global_proba_by_temp_generic(
        df=df,
        ids_a_garder=ids_actifs_heat,
        point=austin,
        start=start, end=end,
        energy_col="furnace_total",
        seuil_on=seuil_on_heat,
        bin_start=bins_heat["bin_start"],
        bin_stop=bins_heat["bin_stop"],
        bin_step=bins_heat["bin_step"],
        months=months_cold,
        tz_target="America/Chicago",
        temp_col_out="temp"
    )

    # # 7) Affichage
    # print("\nProbabilité moyenne que le chauffage soit ON – Austin :")
    # print(proba_heat[["temp_center", "p_on"]])

    # plt.figure(figsize=(9, 5))
    # plt.plot(proba_heat["temp_center"], proba_heat["p_on"], marker="o", color="navy", linewidth=2)
    # plt.title("Probabilité que le chauffage soit ON – Austin")
    # plt.xlabel("Température extérieure (°C)")
    # plt.ylabel("P(chauffage ON)")
    # plt.yticks(np.arange(0, 1.0001, 0.05))
    # plt.ylim(0, 1)
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    

    temp_15 = build_temp_15min_from_meteostat_ms2(point=austin, start=start, end=end,
                                                  tz_target="America/Chicago", temp_col_out="temp")
    df = parse_naive_datetime_col(df, "local_15min")
    df = df.set_index("local_15min").sort_index()
    df = df.join(temp_15, how="left")


    proba_clients_active = compute_proportion_clients_actifs(
        df=df,
        ids_a_garder=ids_actifs_heat,
        energy_col="furnace_total",
        seuil_on=0.2,            # seuil ON (kW)
        ratio_threshold=0.25,     # client actif si ≥ 40% du temps ON
        bin_start=-5, bin_stop=25, bin_step=1,
        months=(11, 12, 1, 2)    # mois froids
)


    print("\nProportion de clients actifs par bin :")
    print(proba_clients_active[["temp_center", "p_clients_active"]])
    

    # Graphique 1 : Temps ON
    plt.figure(figsize=(9, 5))
    plt.plot(proba_heat["temp_center"], proba_heat["p_on"],
             marker="o", color="navy", linewidth=2)
    plt.title("Probabilité que le chauffage soit ON – Austin")
    plt.xlabel("Température extérieure (°C)")
    plt.ylabel("P(chauffage ON)")
    plt.yticks(np.arange(0, 1.0001, 0.05))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Graphique 2 : Clients actifs
    plt.figure(figsize=(9, 5))
    plt.plot(proba_clients_active["temp_center"], proba_clients_active["p_clients_active"],
             marker="s", color="orange", linewidth=2)
    plt.title("Proportion de clients actifs – Chauffage Austin")
    plt.xlabel("Température extérieure (°C)")
    plt.ylabel("Proportion de clients actifs")
    plt.yticks(np.arange(0, 1.0001, 0.05))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
