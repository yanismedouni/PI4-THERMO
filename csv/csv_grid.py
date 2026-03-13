import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os




import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# =============================
# PARAMÈTRES
# =============================
MAX_GAP = 8          # 8 × 15 min = 2 heures
EDGE_MARGIN = 8
WINDOW = 8

# =============================
# CONFIG RÉGIONS (1 exécution = 3 sorties)
# =============================
REGIONS = {
    "austin": {
        "csv_path": "data/15minute_data_austin.csv",
        "output_csv": "csv/output/grid_interp_austin.csv",
    },
    "newyork": {
        "csv_path": "data/15minute_data_newyork.csv",
        "output_csv": "csv/output/grid_interp_newyork.csv",
    },
    "california": {
        "csv_path": "data/15minute_data_california.csv",
        "output_csv": "csv/output/grid_interp_california.csv",
    },
}


def lire_csv_auto_sep(chemin_csv: str) -> pd.DataFrame:
    """
    Lecture robuste: essaye d'abord ',' puis ';'
    et vérifie que 'dataid' et 'local_15min' existent.
    """
    # 1) essai virgule
    df = pd.read_csv(chemin_csv, sep=",", low_memory=False)
    df.columns = df.columns.str.strip()
    if "dataid" in df.columns and "local_15min" in df.columns:
        return df

    # 2) essai point-virgule
    df = pd.read_csv(chemin_csv, sep=";", low_memory=False)
    df.columns = df.columns.str.strip()
    if "dataid" in df.columns and "local_15min" in df.columns:
        return df

    # 3) diagnostic utile
    raise ValueError(
        f"Impossible de détecter le séparateur pour {chemin_csv}. "
        f"Colonnes lues: {df.columns.tolist()}"
    )


def interpoler_grid_par_client(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le traitement identique à ton script :
    - normalisation timestamp
    - grille 15 min
    - interpolation spline sur trous courts
    - sortie: dataid, local_15min, grid_interp
    """
    # Nettoyage des espaces cachés dans les noms de colonnes
    df.columns = df.columns.str.strip()

    # Vérifs colonnes minimales
    for col in ("dataid", "local_15min", "grid"):
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' introuvable. Colonnes: {df.columns.tolist()}")

    # Nettoyage du timestamp (UTC → Etc/GMT+6 → tz-naive)
    dt_utc = pd.to_datetime(df["local_15min"], errors="coerce", utc=True)
    dt_local = dt_utc.dt.tz_convert("Etc/GMT+6")
    df["local_15min"] = dt_local.dt.tz_localize(None)

    df_all_clients = []
    clients = df["dataid"].unique()

    for dataid in clients:
        print(f"Traitement du client {dataid}")

        df_client = df[df["dataid"] == dataid].copy()

        # Sélection grid
        df_client = df_client[["local_15min", "grid"]].copy()
        df_client["grid"] = pd.to_numeric(df_client["grid"], errors="coerce")
        df_client.loc[df_client["grid"] < 0, "grid"] = 0

        # Mise sur grille 15 minutes
        df_client = df_client.groupby("local_15min", as_index=False).mean(numeric_only=True)
        df_client = df_client.set_index("local_15min").asfreq("15min").reset_index()

        y = df_client["grid"]
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

            mask_local = ~y_local.isna()
            if mask_local.sum() < 4:
                continue

            cs_local = CubicSpline(t_local[mask_local], y_local[mask_local], bc_type="natural")
            y_interp.loc[idx] = cs_local(t[idx])

        y_interp[y_interp < 0] = 0
        df_client["grid_interp"] = y_interp
        df_client["dataid"] = dataid

        df_all_clients.append(df_client[["dataid", "local_15min", "grid_interp"]])

    return pd.concat(df_all_clients, ignore_index=True)


if __name__ == "__main__":

    for region, cfg in REGIONS.items():
        csv_path = cfg["csv_path"]
        output_csv = cfg["output_csv"]

        print(f"\n================ {region.upper()} ================\n")
        print(f"[INFO] Lecture : {csv_path}")

        df = lire_csv_auto_sep(csv_path)

        print("[INFO] Colonnes lues :")
        print(df.columns.tolist())

        df_out = interpoler_grid_par_client(df)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_out.to_csv(output_csv, index=False)
        print(f"\n[OK] Fichier CSV généré : {output_csv}")