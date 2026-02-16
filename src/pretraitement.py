"""
Energy Consumption Data Processing Script

This script processes energy consumption data:
1. Loads interpolated grid consumption from csv/grid_interp.csv
2. Loads interpolated solar production from csv/solar_interp.csv
3. Loads EV consumption (car1, car2) from the raw CSV
4. Adds solar production to grid consumption
5. Removes EV consumption from grid if it exceeds 3kW threshold
6. Outputs a clean CSV with only relevant columns
"""

import pandas as pd


def add_solar_to_grid(df):
    """
    Add solar production to grid consumption.
    (solar2 no longer exists as a separate column since solar_interp
    already aggregates all solar sources upstream.)
    """
    df['solar'] = df['solar'].fillna(0)
    df['grid'] = df['grid'].fillna(0) + df['solar']
    return df


def remove_ev_consumption_above_threshold(df, threshold_kw=3.0):
    """
    Remove EV consumption (car1 + car2) from grid only if total exceeds threshold.
    """
    df['car'] = df['car1'].fillna(0) + df['car2'].fillna(0)

    df.loc[df['car'] > threshold_kw, 'grid'] = (
        df['grid'] - df['car']
    )

    return df


def process_energy_data(grid_file, solar_file, raw_file, output_file, ev_threshold_kw=3.0):
    """
    Main function to process energy consumption data.

    Parameters
    ----------
    grid_file  : path to grid_interp.csv  (output of csv/csv_grid.py)
    solar_file : path to solar_interp.csv (output of csv/csv_solar.py)
    raw_file   : path to the original 15minute_data_austin.csv
                 (used only to retrieve EV columns car1, car2)
    output_file: path for the processed output CSV
    """

    # ------------------------------------------------------------------
    # 1. Load interpolated grid & solar
    # ------------------------------------------------------------------
    df_grid  = pd.read_csv(grid_file,  parse_dates=["local_15min"])
    df_solar = pd.read_csv(solar_file, parse_dates=["local_15min"])

    print(f"Loaded {len(df_grid)}  rows from {grid_file}")
    print(f"Loaded {len(df_solar)} rows from {solar_file}")

    # Rename interpolated columns to match the rest of the pipeline
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
        df_raw = pd.read_csv(raw_file, parse_dates=["local_15min"])
    else:
        df_raw = pd.read_excel(raw_file, parse_dates=["local_15min"])

    # Normalise timestamps so the merge keys align
    df_raw.columns = df_raw.columns.str.strip()

    # Keep only the columns we need from the raw file
    ev_cols = ["dataid", "local_15min"]
    for col in ("car1", "car2"):
        if col in df_raw.columns:
            ev_cols.append(col)
        else:
            print(f"  Warning: column '{col}' not found in raw file — assuming 0.")

    df_ev = df_raw[ev_cols].copy()

    # Aggregate to 15-min grid (same as csv_grid.py does)
    df_ev["local_15min"] = pd.to_datetime(df_ev["local_15min"], utc=True, errors="coerce")
    df_ev["local_15min"] = df_ev["local_15min"].dt.tz_convert("Etc/GMT+6").dt.tz_localize(None)

    agg_cols = {c: "mean" for c in ("car1", "car2") if c in df_ev.columns}
    df_ev = df_ev.groupby(["dataid", "local_15min"], as_index=False).agg(agg_cols)

    # Add missing EV columns as zeros if they were absent in the raw file
    for col in ("car1", "car2"):
        if col not in df_ev.columns:
            df_ev[col] = 0.0

    # ------------------------------------------------------------------
    # 4. Merge EV data into main dataframe
    # ------------------------------------------------------------------
    df = pd.merge(df, df_ev, on=["dataid", "local_15min"], how="left")

    print(f"\nMerged dataset: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # ------------------------------------------------------------------
    # 5. Save original grid before modifications
    # ------------------------------------------------------------------
    df['grid_original'] = df['grid'].values.copy()

    # ------------------------------------------------------------------
    # 6. Apply transformations
    # ------------------------------------------------------------------
    df = add_solar_to_grid(df)
    df = remove_ev_consumption_above_threshold(df, threshold_kw=ev_threshold_kw)

    # ------------------------------------------------------------------
    # 7. Keep only relevant columns
    # ------------------------------------------------------------------
    output_columns = ['dataid', 'local_15min', 'solar', 'car', 'grid_original', 'grid']
    df_output = df[output_columns].copy()

    # ------------------------------------------------------------------
    # 8. Save output
    # ------------------------------------------------------------------
    try:
        df_output.to_csv(output_file, index=False)
        print(f"\n✔ Successfully saved to {output_file}")
    except PermissionError:
        print(f"\n✗ ERROR: Cannot write to {output_file}")
        print("  The file may be open in Excel. Please close it and try again.")
        return None

    print("\nFirst few rows:")
    print(df_output.head())
    print("\nSummary statistics:")
    print(df_output.describe())

    return df_output


# ==========================
#        MAIN PROGRAM
# ==========================
if __name__ == "__main__":

    grid_path   = "../csv/output/grid_interp.csv"
    solar_path  = "../csv/output/solar_interp.csv"
    raw_path    = "../data/15minute_data_austin.csv"   # still needed for car1/car2
    output_path = "../output/processed_energy_data.csv"

    process_energy_data(
        grid_file=grid_path,
        solar_file=solar_path,
        raw_file=raw_path,
        output_file=output_path,
        ev_threshold_kw=3.0,
    )
