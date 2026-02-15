
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# =============================
# PARAMÈTRES
# =============================
MAX_GAP = 8   # 4 × 15 min = 1 heure

# =============================
# 1) Chargement des données
# =============================
csv_path = "C:/Users/Edith-Irene/Desktop/PIGE/15minute_data_austin/15minute_data_austin/15minute_data_austin.csv"
df = pd.read_csv(csv_path)

# =============================
# 2) Nettoyage du timestamp
# =============================
s = df["local_15min"].astype(str).str.strip()
s = s.str.replace("T", " ", regex=False)
s = s.str.replace(r"\.\d{1,6}", "", regex=True)
s = s.str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True)

dt = pd.to_datetime(s, errors="coerce", utc=True)
df["local_15min"] = dt.dt.tz_localize(None)

# =============================
# 3) Sélection du ménage
# =============================
# =============================
# Création d'un DataFrame pour tous les clients
# =============================
df_all_clients = pd.DataFrame()
clients = df["dataid"].unique()

for dataid in clients:

    print(f"Traitement du client {dataid}")

    df_client = df[df["dataid"] == dataid].copy()

    # Filtrage sur période
    # date_debut = "2018-10-01"
    # date_fin   = "2018-10-16"
    # mask = (df_client["local_15min"] >= date_debut) & (df_client["local_15min"] <= date_fin)
    # df_client = df_client.loc[mask]
    # if df_client.empty:
    #     continue

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

    EDGE_MARGIN = 8
    WINDOW = 8

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

    # Ajouter la colonne dataid
    df_client["dataid"] = dataid

    # Concaténer au DataFrame global
    df_all_clients = pd.concat([df_all_clients, df_client[["dataid", "local_15min", "grid_interp"]]], ignore_index=True)

# =============================
# Sauvegarde finale
# =============================
output_csv = "C:/Users/Edith-Irene/Desktop/PIGE/15minute_data_austin/15minute_data_austin/cleaned_data_all_clients.csv"
df_all_clients.to_csv(output_csv, index=False)
print(f"Fichier CSV généré : {output_csv}")

