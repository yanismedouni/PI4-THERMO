"""
Energy Consumption Data Processing Script

This script processes energy consumption data:
1. Loads interpolated grid consumption from csv/grid_interp.csv
2. Loads interpolated solar production from csv/solar_interp.csv
3. Loads EV consumption (car1, car2) from the raw CSV
4. Adds solar production to grid consumption
5. Removes EV consumption from grid if it exceeds 3kW threshold
6. Merges extra appliance + temperature columns from the raw CSV
7. Outputs a clean CSV with all relevant columns
"""

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Columns to pull from the raw CSV (besides dataid / local_15min)
# ──────────────────────────────────────────────────────────────────────
_EXTRA_COLS = [
    "air1", "air2", "air3",
    "airwindowunit1",
    "furnace1", "furnace2",
    "heater1", "heater2", "heater3",
]


# ──────────────────────────────────────────────────────────────────────
def add_solar_to_grid(df):
    """
    Add solar production to grid consumption.
    (solar2 no longer exists as a separate column since solar_interp
    already aggregates all solar sources upstream.)
    """
    df['solar'] = df['solar'].fillna(0)
    df['grid'] = df['grid'].fillna(0) + df['solar']
    return df


# ──────────────────────────────────────────────────────────────────────
def remove_ev_consumption_above_threshold(df, threshold_kw=3.0):
    """
    Remove EV consumption (car1 + car2) from grid only if total exceeds threshold.
    """
    df['car'] = df['car1'].fillna(0) + df['car2'].fillna(0)
    df.loc[df['car'] > threshold_kw, 'grid'] = df['grid'] - df['car']
    return df


# ──────────────────────────────────────────────────────────────────────
def load_extra_columns_from_raw(raw_file: str) -> pd.DataFrame:
    """
    Load the extra appliance + temperature columns from the raw Pecan Street CSV.

    Returns a DataFrame with columns:
        dataid, local_15min, year, month, day, hour, minute,
        temp, air1, air2, air3, airwindowunit1,
        furnace1, furnace2, heater1, heater2, heater3

    Timestamps are normalised to UTC-6 / tz-naive (same convention as
    csv_grid.py / csv_solar.py) so they merge cleanly with the
    interpolated data.
    """
    if raw_file.endswith(".csv"):
        df_raw = pd.read_csv(raw_file)
    else:
        df_raw = pd.read_excel(raw_file)

    df_raw.columns = df_raw.columns.str.strip()

    # ── timestamp normalisation (UTC → Etc/GMT+6 → tz-naive) ──────────
    df_raw["local_15min"] = (
        pd.to_datetime(df_raw["local_15min"], errors="coerce", utc=True)
        .dt.tz_convert("Etc/GMT+6")
        .dt.tz_localize(None)
    )

    # ── keep only the columns we need ─────────────────────────────────
    present_extras = [c for c in _EXTRA_COLS if c in df_raw.columns]
    missing = set(_EXTRA_COLS) - set(present_extras)
    if missing:
        print(f"  Warning: columns not found in raw file (will be NaN): {sorted(missing)}")

    keep = ["dataid", "local_15min"] + present_extras
    df_extra = df_raw[keep].copy()

    # ── numeric conversion + clip negatives to 0 for power channels ───
    power_cols = [c for c in present_extras if c != "temp"]
    for col in power_cols:
        df_extra[col] = pd.to_numeric(df_extra[col], errors="coerce")
        df_extra.loc[df_extra[col] < 0, col] = 0

    if "temp" in df_extra.columns:
        df_extra["temp"] = pd.to_numeric(df_extra["temp"], errors="coerce")

    # ── aggregate to 15-min grid (same logic as csv_grid.py) ──────────
    agg = {c: "mean" for c in present_extras}
    df_extra = df_extra.groupby(["dataid", "local_15min"], as_index=False).agg(agg)

    # ── add missing columns as NaN so the schema is always consistent ─
    for col in _EXTRA_COLS:
        if col not in df_extra.columns:
            df_extra[col] = float("nan")

    # ── extract datetime components ────────────────────────────────────
    dt = df_extra["local_15min"]
    df_extra.insert(2, "year",   dt.dt.year)
    df_extra.insert(3, "month",  dt.dt.month)
    df_extra.insert(4, "day",    dt.dt.day)
    df_extra.insert(5, "hour",   dt.dt.hour)
    df_extra.insert(6, "minute", dt.dt.minute)

    return df_extra


# ──────────────────────────────────────────────────────────────────────
def merge_extra_columns(df: pd.DataFrame, df_extra: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the extra columns onto the main processed dataframe
    and enforce a clean final column order.
    """
    df_merged = pd.merge(df, df_extra, on=["dataid", "local_15min"], how="left")

    # temp is filled later by open-meteo (temperature.py) — add placeholder if absent
    if "temp" not in df_merged.columns:
        df_merged["temp"] = float("nan")

    ordered = [
        "dataid", "local_15min",
        "year", "month", "day", "hour", "minute",
        "temp",
        "grid_original", "grid", "solar", "car",
        "air1", "air2", "air3", "airwindowunit1",
        "furnace1", "furnace2",
        "heater1", "heater2", "heater3",
    ]
    # Keep any unexpected columns at the end rather than silently drop them
    extra_cols = [c for c in df_merged.columns if c not in ordered]
    return df_merged[ordered + extra_cols]


# ──────────────────────────────────────────────────────────────────────
def process_energy_data(grid_file, solar_file, raw_file, output_file, ev_threshold_kw=3.0):
    """
    Main function to process energy consumption data.

    Parameters
    ----------
    grid_file      : path to grid_interp.csv  (output of csv/csv_grid.py)
    solar_file     : path to solar_interp.csv (output of csv/csv_solar.py)
    raw_file       : path to the original 15minute_data_austin.csv
    output_file    : path for the processed output CSV
    ev_threshold_kw: EV consumption threshold in kW (default 3.0)
    """

    # ------------------------------------------------------------------
    # 1. Load interpolated grid & solar
    # ------------------------------------------------------------------
    df_grid  = pd.read_csv(grid_file,  parse_dates=["local_15min"])
    df_solar = pd.read_csv(solar_file, parse_dates=["local_15min"])

    df_grid  = df_grid.rename(columns={"grid_interp":  "grid"})
    df_solar = df_solar.rename(columns={"solar_interp": "solar"})

    # ------------------------------------------------------------------
    # 2. Merge grid + solar on (dataid, local_15min)
    # ------------------------------------------------------------------
    df = pd.merge(df_grid, df_solar, on=["dataid", "local_15min"], how="outer")

    # ------------------------------------------------------------------
    # 3. Load EV columns (car1, car2) from the raw CSV
    # ------------------------------------------------------------------
    if raw_file.endswith('.csv'):
        df_raw = pd.read_csv(raw_file)
    else:
        df_raw = pd.read_excel(raw_file)

    df_raw.columns = df_raw.columns.str.strip()

    ev_cols = ["dataid", "local_15min"]
    for col in ("car1", "car2"):
        if col in df_raw.columns:
            ev_cols.append(col)
        else:
            print(f"  Warning: column '{col}' not found in raw file — assuming 0.")

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

    # ------------------------------------------------------------------
    # 4. Merge EV data into main dataframe
    # ------------------------------------------------------------------
    df = pd.merge(df, df_ev, on=["dataid", "local_15min"], how="left")

    # ------------------------------------------------------------------
    # 5. Save original grid before modifications
    # ------------------------------------------------------------------
    df['grid_original'] = df['grid'].values.copy()

    # ------------------------------------------------------------------
    # 6. Apply solar + EV transformations
    # ------------------------------------------------------------------
    df = add_solar_to_grid(df)
    df = remove_ev_consumption_above_threshold(df, threshold_kw=ev_threshold_kw)

    # ------------------------------------------------------------------
    # 7. Load & merge extra appliance + temperature columns
    # ------------------------------------------------------------------
    df_extra = load_extra_columns_from_raw(raw_file)
    df = merge_extra_columns(df, df_extra)

    # ------------------------------------------------------------------
    # 8. Save output
    # ------------------------------------------------------------------
    try:
        df.to_csv(output_file, index=False)
        print(f"  ✔ Saved → {output_file}")
    except PermissionError:
        print(f"  ✗ ERROR: Cannot write to {output_file} — file may be open in Excel.")
        return None

    return df


# ──────────────────────────────────────────────────────────────────────
#        MAIN PROGRAM
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    grid_path   = "../csv/output/grid_interp.csv"
    solar_path  = "../csv/output/solar_interp.csv"
    raw_path    = "../data/15minute_data_austin.csv"
    output_path = "../output/processed_energy_data.csv"

    process_energy_data(
        grid_file=grid_path,
        solar_file=solar_path,
        raw_file=raw_path,
        output_file=output_path,
        ev_threshold_kw=3.0,
    )