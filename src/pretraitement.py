"""
Energy Consumption Data Processing Script
Multi-régions : Austin / New York / California
"""

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Lecture robuste CSV brut (évite bug séparateur ;)
# ──────────────────────────────────────────────────────────────────────
def read_raw_csv_robust(path: str) -> pd.DataFrame:

    # Essai séparateur virgule
    df = pd.read_csv(path, sep=",")
    df.columns = df.columns.str.strip()
    if "dataid" in df.columns and "local_15min" in df.columns:
        return df

    # Essai séparateur point-virgule
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    if "dataid" in df.columns and "local_15min" in df.columns:
        return df

    raise ValueError(
        f"Impossible de lire correctement {path}. "
        f"Colonnes détectées: {df.columns.tolist()}"
    )


# ──────────────────────────────────────────────────────────────────────
# Colonnes "extra" à récupérer du brut (sans TCLs individuels)
# ──────────────────────────────────────────────────────────────────────
_EXTRA_COLS = [
    "airwindowunit1",
]


# ──────────────────────────────────────────────────────────────────────
def add_solar_to_grid(df):
    df["solar"] = df["solar"].fillna(0)
    df["grid"] = df["grid"].fillna(0) + df["solar"]
    return df


# ──────────────────────────────────────────────────────────────────────
def remove_ev_consumption_above_threshold(df, threshold_kw=3.0):

    df["car"] = df["car1"].fillna(0) + df["car2"].fillna(0)
    df.loc[df["car"] > threshold_kw, "grid"] = df["grid"] - df["car"]
    return df


# ──────────────────────────────────────────────────────────────────────
def load_tcls_from_csv(tcls_file: str) -> pd.DataFrame:
    """
    Charge clim + chauffage depuis le fichier généré par csv_tcls.
    Attendu: dataid, local_15min, clim, chauffage
    """
    df_tcls = pd.read_csv(tcls_file, parse_dates=["local_15min"])
    df_tcls.columns = df_tcls.columns.str.strip()

    required = {"dataid", "local_15min", "clim", "chauffage"}
    missing = required - set(df_tcls.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {tcls_file}: {sorted(missing)}. "
            f"Colonnes détectées: {df_tcls.columns.tolist()}"
        )

    # Sécurité: conversion numérique (si jamais)
    df_tcls["clim"] = pd.to_numeric(df_tcls["clim"], errors="coerce")
    df_tcls["chauffage"] = pd.to_numeric(df_tcls["chauffage"], errors="coerce")

    return df_tcls[["dataid", "local_15min", "clim", "chauffage"]].copy()


# ──────────────────────────────────────────────────────────────────────
def load_extra_columns_from_raw(raw_file: str) -> pd.DataFrame:
    """
    Charge seulement les colonnes extra du brut (hors TCLs individuels),
    normalise le timestamp, et agrège à 15 minutes.
    """
    df_raw = read_raw_csv_robust(raw_file)

    df_raw["local_15min"] = (
        pd.to_datetime(df_raw["local_15min"], errors="coerce", utc=True)
        .dt.tz_convert("Etc/GMT+6")
        .dt.tz_localize(None)
    )

    present_extras = [c for c in _EXTRA_COLS if c in df_raw.columns]
    missing = set(_EXTRA_COLS) - set(present_extras)

    if missing:
        print(f"  Warning: columns not found in raw file (NaN): {sorted(missing)}")

    keep = ["dataid", "local_15min"] + present_extras
    df_extra = df_raw[keep].copy()

    for col in present_extras:
        df_extra[col] = pd.to_numeric(df_extra[col], errors="coerce")
        df_extra.loc[df_extra[col] < 0, col] = 0

    agg = {c: "mean" for c in present_extras}
    df_extra = df_extra.groupby(["dataid", "local_15min"], as_index=False).agg(agg)

    for col in _EXTRA_COLS:
        if col not in df_extra.columns:
            df_extra[col] = float("nan")

    dt = df_extra["local_15min"]
    df_extra.insert(2, "year", dt.dt.year)
    df_extra.insert(3, "month", dt.dt.month)
    df_extra.insert(4, "day", dt.dt.day)
    df_extra.insert(5, "hour", dt.dt.hour)
    df_extra.insert(6, "minute", dt.dt.minute)

    return df_extra


# ──────────────────────────────────────────────────────────────────────
def merge_extra_columns(df: pd.DataFrame, df_extra: pd.DataFrame) -> pd.DataFrame:

    df_merged = pd.merge(df, df_extra, on=["dataid", "local_15min"], how="left")

    if "temp" not in df_merged.columns:
        df_merged["temp"] = float("nan")

    ordered = [
        "dataid", "local_15min",
        "year", "month", "day", "hour", "minute",
        "temp",
        "grid_original", "grid", "solar", "car",
        "clim", "chauffage",
        "airwindowunit1",
    ]

    extra_cols = [c for c in df_merged.columns if c not in ordered]
    return df_merged[ordered + extra_cols]


# ──────────────────────────────────────────────────────────────────────
def process_energy_data(grid_file, solar_file, tcls_file, raw_file, output_file, ev_threshold_kw=3.0):

    df_grid = pd.read_csv(grid_file, parse_dates=["local_15min"])
    df_solar = pd.read_csv(solar_file, parse_dates=["local_15min"])

    df_grid = df_grid.rename(columns={"grid_interp": "grid"})
    df_solar = df_solar.rename(columns={"solar_interp": "solar"})

    df = pd.merge(df_grid, df_solar, on=["dataid", "local_15min"], how="outer")

    # ── Ajouter clim + chauffage depuis csv_tcls ───────────────────────
    df_tcls = load_tcls_from_csv(tcls_file)
    df = pd.merge(df, df_tcls, on=["dataid", "local_15min"], how="left")

    # EV
    df_raw = read_raw_csv_robust(raw_file)

    ev_cols = ["dataid", "local_15min"]
    for col in ("car1", "car2"):
        if col in df_raw.columns:
            ev_cols.append(col)
        else:
            print(f"  Warning: column '{col}' not found — assuming 0.")

    df_ev = df_raw[ev_cols].copy()

    df_ev["local_15min"] = (
        pd.to_datetime(df_ev["local_15min"], utc=True, errors="coerce")
        .dt.tz_convert("Etc/GMT+6")
        .dt.tz_localize(None)
    )

    agg_cols = {c: "mean" for c in ("car1", "car2") if c in df_ev.columns}
    df_ev = df_ev.groupby(["dataid", "local_15min"], as_index=False).agg(agg_cols)

    for col in ("car1", "car2"):
        if col not in df_ev.columns:
            df_ev[col] = 0.0

    df = pd.merge(df, df_ev, on=["dataid", "local_15min"], how="left")

    df["grid_original"] = df["grid"].values.copy()

    df = add_solar_to_grid(df)
    df = remove_ev_consumption_above_threshold(df, threshold_kw=ev_threshold_kw)

    # Extras depuis brut (hors TCLs)
    df_extra = load_extra_columns_from_raw(raw_file)
    df = merge_extra_columns(df, df_extra)

    df.to_csv(output_file, index=False)
    print(f"  ✔ Saved → {output_file}")

    return df


# ──────────────────────────────────────────────────────────────────────
#        MAIN PROGRAM MULTI-RÉGIONS
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    REGIONS = {
        "austin": {
            "grid":  "csv/output/grid_interp_austin.csv",
            "solar": "csv/output/solar_interp_austin.csv",
            "tcls":  "csv/output/tcls_interp_austin.csv",
            "raw":   "data/15minute_data_austin.csv",
            "out":   "output/processed_energy_data_austin.csv",
        },
        "newyork": {
            "grid":  "csv/output/grid_interp_newyork.csv",
            "solar": "csv/output/solar_interp_newyork.csv",
            "tcls":  "csv/output/tcls_interp_newyork.csv",
            "raw":   "data/15minute_data_newyork.csv",
            "out":   "output/processed_energy_data_newyork.csv",
        },
        "california": {
            "grid":  "csv/output/grid_interp_california.csv",
            "solar": "csv/output/solar_interp_california.csv",
            "tcls":  "csv/output/tcls_interp_california.csv",
            "raw":   "data/15minute_data_california.csv",
            "out":   "output/processed_energy_data_california.csv",
        },
    }

    for region, cfg in REGIONS.items():

        print(f"\n================ {region.upper()} ================\n")

        process_energy_data(
            grid_file=cfg["grid"],
            solar_file=cfg["solar"],
            tcls_file=cfg["tcls"],
            raw_file=cfg["raw"],
            output_file=cfg["out"],
            ev_threshold_kw=3.0,
        )