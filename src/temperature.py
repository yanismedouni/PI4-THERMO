"""
Temperature merging module.

For each region, loads the corresponding open-meteo hourly CSV,
converts timestamps from GMT to the region's local timezone,
interpolates linearly to 15-minute resolution, then replaces the
'temp' column in the processed energy DataFrame (or CSV file).

Region → timezone mapping (tz-naive after conversion, matching
the convention used throughout the pipeline):
    austin      → Etc/GMT+6   (UTC-6)
    california  → Etc/GMT+8   (UTC-8)
    newyork     → Etc/GMT+5   (UTC-5)
    puertorico  → Etc/GMT+4   (UTC-4)
"""

import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Region configuration
# ──────────────────────────────────────────────────────────────────────
REGION_CONFIG = {
    "austin": {
        "meteo_file": "open-meteo-austin.csv",
        "timezone":   "Etc/GMT+6",
    },
    "california": {
        "meteo_file": "open-meteo-california.csv",
        "timezone":   "Etc/GMT+8",
    },
    "newyork": {
        "meteo_file": "open-meteo-newyork.csv",
        "timezone":   "Etc/GMT+5",
    },
    "puertorico": {
        "meteo_file": "open-meteo-puertorico.csv",
        "timezone":   "Etc/GMT+4",
    },
}


# ──────────────────────────────────────────────────────────────────────
def load_meteo_csv(meteo_path: str | Path) -> pd.DataFrame:
    """
    Parse an open-meteo CSV (which has a 3-row header block before
    the actual time/temperature columns) and return a clean DataFrame
    with columns: [time, temperature_2m].

    The open-meteo format looks like:
        latitude,longitude,...
        (blank line)
        time,temperature_2m (°C)
        2018-01-01T00:00,0.0
        ...
    """
    # Skip the metadata rows: find where 'time' column starts
    with open(meteo_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Find the header row index
    header_idx = next(
        i for i, line in enumerate(lines)
        if line.strip().startswith("time,")
    )

    df = pd.read_csv(meteo_path, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]

    # Rename temperature column to a clean name regardless of unit suffix
    temp_col = next(c for c in df.columns if c.startswith("temperature"))
    df = df.rename(columns={"time": "time", temp_col: "temperature_2m"})

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["temperature_2m"] = pd.to_numeric(df["temperature_2m"], errors="coerce")
    df = df.dropna(subset=["time", "temperature_2m"]).reset_index(drop=True)

    return df[["time", "temperature_2m"]]


# ──────────────────────────────────────────────────────────────────────
def build_temperature_series(meteo_path: str | Path, timezone: str) -> pd.Series:
    """
    Load an open-meteo CSV, convert from GMT to the target timezone,
    then interpolate linearly to 15-minute resolution.

    Returns a Series indexed by tz-naive local timestamps (15-min grid),
    named 'temp'.
    """
    df = load_meteo_csv(meteo_path)

    # open-meteo timestamps are UTC/GMT — localise then convert
    df["time"] = (
        df["time"]
        .dt.tz_localize("UTC")
        .dt.tz_convert(timezone)
        .dt.tz_localize(None)           # drop tzinfo → tz-naive
    )

    df = df.set_index("time").sort_index()

    # Reindex to 15-min grid and forward-fill each hourly value across its 4 slots
    idx_15min = pd.date_range(
        start=df.index.min(),
        end=df.index.max() + pd.Timedelta(minutes=45),
        freq="15min",
    )
    series = (
        df["temperature_2m"]
        .reindex(idx_15min)
        .ffill()
    )
    series.name = "temp"
    series.index.name = "local_15min"

    return series


# ──────────────────────────────────────────────────────────────────────
def merge_temperature(
    df: pd.DataFrame,
    region: str,
    data_dir: str | Path,
) -> pd.DataFrame:
    """
    Replace the 'temp' column in *df* with open-meteo interpolated
    temperature for the given region.

    Parameters
    ----------
    df        : processed energy DataFrame (must have 'local_15min' column)
    region    : one of 'austin', 'california', 'newyork', 'puertorico'
    data_dir  : directory containing the open-meteo CSV files

    Returns
    -------
    DataFrame with 'temp' column replaced by open-meteo values.
    Rows whose timestamp falls outside the meteo coverage stay NaN.
    """
    data_dir = Path(data_dir)
    config   = REGION_CONFIG[region]
    meteo_path = data_dir / config["meteo_file"]

    if not meteo_path.exists():
        print(f"  ⚠ Fichier température introuvable : {meteo_path} — colonne 'temp' inchangée.")
        return df

    temp_series = build_temperature_series(meteo_path, config["timezone"])

    df = df.copy()
    df["local_15min"] = pd.to_datetime(df["local_15min"])
    df["temp"] = df["local_15min"].map(temp_series)

    return df


# ──────────────────────────────────────────────────────────────────────
def merge_temperature_file(
    csv_path: str | Path,
    region: str,
    data_dir: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load a processed energy CSV, replace its 'temp'
    column, and optionally save it back.

    Parameters
    ----------
    csv_path    : path to the processed energy CSV
    region      : region key (see REGION_CONFIG)
    data_dir    : directory with open-meteo files
    output_path : where to save the result (None → overwrite csv_path)

    Returns the updated DataFrame.
    """
    df = pd.read_csv(csv_path, parse_dates=["local_15min"])
    df = merge_temperature(df, region, data_dir)

    out = Path(output_path) if output_path else Path(csv_path)
    df.to_csv(out, index=False)

    return df