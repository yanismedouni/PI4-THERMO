import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os  # ← AJOUT

import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

MAX_GAP = 8
EDGE_MARGIN = 8
WINDOW = 8

REGIONS = {
    "austin": {
        "csv_path": "data/15minute_data_austin.csv",
        "output_csv": "csv/output/solar_interp_austin.csv",
    },
    "newyork": {
        "csv_path": "data/15minute_data_newyork.csv",
        "output_csv": "csv/output/solar_interp_newyork.csv",
    },
    "california": {
        "csv_path": "data/15minute_data_california.csv",
        "output_csv": "csv/output/solar_interp_california.csv",
    },
}

def lire_csv_auto_sep(chemin_csv: str) -> pd.DataFrame:
    df = pd.read_csv(chemin_csv, sep=",", low_memory=False)
    df.columns = df.columns.str.strip()
    if "dataid" in df.columns and "local_15min" in df.columns:
        return df

    df = pd.read_csv(chemin_csv, sep=";", low_memory=False)
    df.columns = df.columns.str.strip()
    if "dataid" in df.columns and "local_15min" in df.columns:
        return df

    raise ValueError(f"Impossible de lire {chemin_csv}. Colonnes: {df.columns.tolist()}")
def interpoler_solar_par_client(df):

    df.columns = df.columns.str.strip()

    # ⚠️ solar peut être solar + solar2 selon région
    solar_cols = [c for c in ["solar", "solar2"] if c in df.columns]

    if not solar_cols:
        raise ValueError("Aucune colonne solar trouvée")

    # timestamp
    dt_utc = pd.to_datetime(df["local_15min"], errors="coerce", utc=True)
    dt_local = dt_utc.dt.tz_convert("Etc/GMT+6")
    df["local_15min"] = dt_local.dt.tz_localize(None)

    df["solar_total"] = df[solar_cols].fillna(0).sum(axis=1)

    df_all = []

    for dataid in df["dataid"].unique():

        df_client = df[df["dataid"] == dataid].copy()
        df_client = df_client[["local_15min", "solar_total"]]

        df_client = df_client.groupby("local_15min", as_index=False).mean()
        df_client = df_client.set_index("local_15min").asfreq("15min").reset_index()

        y = df_client["solar_total"]
        t = np.arange(len(y))

        df_client["is_nan"] = y.isna()
        df_client["nan_group"] = (df_client["is_nan"] != df_client["is_nan"].shift()).cumsum()

        nan_lengths = df_client[df_client["is_nan"]].groupby("nan_group").size()
        y_interp = y.copy()

        for group_id, length in nan_lengths.items():
            if length > MAX_GAP:
                continue

            idx = df_client[df_client["nan_group"] == group_id].index

            if idx.min() < EDGE_MARGIN or idx.max() > len(y) - EDGE_MARGIN:
                continue

            i_start = max(idx.min() - WINDOW, 0)
            i_end   = min(idx.max() + WINDOW + 1, len(y))

            y_local = y.iloc[i_start:i_end]
            t_local = t[i_start:i_end]

            mask = ~y_local.isna()
            if mask.sum() < 4:
                continue

            cs = CubicSpline(t_local[mask], y_local[mask], bc_type="natural")
            y_interp.loc[idx] = cs(t[idx])

        y_interp[y_interp < 0] = 0
        df_client["solar_interp"] = y_interp
        df_client["dataid"] = dataid

        df_all.append(df_client[["dataid", "local_15min", "solar_interp"]])

    return pd.concat(df_all, ignore_index=True)


if __name__ == "__main__":

    for region, cfg in REGIONS.items():

        print(f"\n=== SOLAR {region.upper()} ===")

        df = lire_csv_auto_sep(cfg["csv_path"])
        df_out = interpoler_solar_par_client(df)

        os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
        df_out.to_csv(cfg["output_csv"], index=False)

        print(f"[OK] {cfg['output_csv']} généré")