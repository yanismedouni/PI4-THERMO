import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------------------------
# CHEMINS de donnée (issue de prétraitement.py)
# ------------------------------------------------------------------
DATA_PATH = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\austin\15minute_data_austin_modified.csv"
OUT_DIR   = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\out_dbscan"

# >>> NOUVEAU: sortie filtrée
OUT_PATH_FILTERED = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\austin\15minute_data_austin_modified_dbscan_filtered.csv"

# ------------------------------------------------------------------
# PARAMÈTRES
# ------------------------------------------------------------------
WEEKS_SELECTED = [(2018, 1), (2018, 2), (2018, 3)]  # (iso_year, iso_week)
N_CLIENTS = 25
RANDOM_SEED = 42

# TUNING PARAMS DBSCAN
MINPTS = 3
EPS_PERCENTILE = 85
EPS_BOOST = 1.10

# Seuils température
T_CHAUDE = 15
T_FROIDE = 5

# Qualité minimale par semaine-client
MIN_POINTS_WEEK = int(7 * 96 * 0.75)

DO_PLOT = True

# ------------------------------------------------------------------
# Fonctions pour DBSCAN et extraction de features
# ------------------------------------------------------------------
def eps_from_kdist_percentile(Xs: np.ndarray, k: int, q: float = 60.0) -> float:
    nbrs = NearestNeighbors(n_neighbors=k).fit(Xs)
    distances, _ = nbrs.kneighbors(Xs)
    kdist = np.sort(distances[:, k - 1])
    return float(np.percentile(kdist, q))

def compute_features_week(df_week: pd.DataFrame) -> pd.DataFrame:
    """
    f1: corr(P,T)
    f2: médiane (Pmax-Pmin)/Pmax (par jour)
    f3: (moy(P|T>=T_CHAUDE)) / moy(P semaine)
    f4: (moy(P|T<=T_FROIDE)) / moy(P semaine)
    f5: (P75-P25)/(P95-P05)  -> élevé = ni trop plat, ni pics exagérés
    """
    rows = []

    for cid, g in df_week.groupby("dataid", sort=False):
        if len(g) < MIN_POINTS_WEEK:
            continue

        g = g.sort_values("dt")
        P = g["grid"].to_numpy(dtype=float)
        T = g["temp"].to_numpy(dtype=float)

        # f1
        if np.nanstd(P) > 0 and np.nanstd(T) > 0:
            f1 = float(np.corrcoef(P, T)[0, 1])
            if np.isnan(f1):
                f1 = 0.0
        else:
            f1 = 0.0

        # f2
        day_stats = g.groupby("date")["grid"].agg(["max", "min"])
        valid_days = day_stats["max"] > 0
        if valid_days.any():
            r = (day_stats.loc[valid_days, "max"] - day_stats.loc[valid_days, "min"]) / day_stats.loc[valid_days, "max"]
            f2 = float(np.median(r.to_numpy()))
        else:
            f2 = 0.0

        # moyenne totale semaine
        mu_tot = float(np.nanmean(P))
        if np.isnan(mu_tot) or mu_tot <= 0:
            mu_tot = 0.0

        # f3
        hot = g["temp"] >= T_CHAUDE
        mu_hot = float(g.loc[hot, "grid"].mean()) if hot.any() else 0.0
        if np.isnan(mu_hot):
            mu_hot = 0.0
        f3 = (mu_hot / mu_tot) if mu_tot > 0 else 0.0

        # f4
        cold = g["temp"] <= T_FROIDE
        mu_cold = float(g.loc[cold, "grid"].mean()) if cold.any() else 0.0
        if np.isnan(mu_cold):
            mu_cold = 0.0
        f4 = (mu_cold / mu_tot) if mu_tot > 0 else 0.0

        # f5
        P_clean = P[np.isfinite(P)]
        if P_clean.size >= 10:
            p05, p25, p75, p95 = np.percentile(P_clean, [5, 25, 75, 95])
            den = (p95 - p05)
            f5 = float((p75 - p25) / den) if den > 1e-9 else 0.0
            if np.isnan(f5):
                f5 = 0.0
        else:
            f5 = 0.0

        rows.append([cid, f1, f2, f3, f4, f5])

    return pd.DataFrame(rows, columns=["dataid", "f1", "f2", "f3", "f4", "f5_mod"])

# ------------------------------------------------------------------
# 1) LECTURE + TIMESTAMP + ISO WEEK
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

needed = {"dataid", "year", "month", "day", "hour", "minute", "grid", "temp"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Colonnes manquantes: {missing}\nColonnes dispo: {list(df.columns)}")

df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df = df.dropna(subset=["dataid", "year", "month", "day", "hour", "minute", "grid", "temp"])

df["dt"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]].astype(int), errors="coerce")
df = df.dropna(subset=["dt"])

iso = df["dt"].dt.isocalendar()
df["iso_year"] = iso["year"].astype(int)
df["iso_week"] = iso["week"].astype(int)
df["date"] = df["dt"].dt.floor("D")

# ------------------------------------------------------------------
# 2) FILTRAGE: garder seulement les semaines demandées
# ------------------------------------------------------------------
mask_weeks = False
for (yy, ww) in WEEKS_SELECTED:
    mask_weeks |= ((df["iso_year"] == yy) & (df["iso_week"] == ww))
df_sel = df.loc[mask_weeks].copy()

# ------------------------------------------------------------------
# 3) SÉLECTION CLIENTS (N_CLIENTS)
# ------------------------------------------------------------------
clients_per_week = {}
for (yy, ww) in WEEKS_SELECTED:
    wdf = df_sel[(df_sel["iso_year"] == yy) & (df_sel["iso_week"] == ww)]
    counts = wdf.groupby("dataid").size()
    ok = set(counts[counts >= MIN_POINTS_WEEK].index)
    clients_per_week[(yy, ww)] = ok

common_clients = set.intersection(*clients_per_week.values()) if clients_per_week else set()
common_clients = sorted(list(common_clients))

print(f"Semaines sélectionnées: {WEEKS_SELECTED}")
print(f"Clients communs admissibles: {len(common_clients)}")

rng = np.random.default_rng(RANDOM_SEED)
if len(common_clients) > N_CLIENTS:
    selected_clients = sorted(rng.choice(common_clients, size=N_CLIENTS, replace=False).tolist())
else:
    selected_clients = common_clients

print(f"Clients sélectionnés: {len(selected_clients)} / {len(common_clients)}")

tmp = df_sel[df_sel["dataid"].isin(selected_clients)]
print("Clients par semaine (après filtrage):")
print(tmp.groupby(["iso_year", "iso_week"])["dataid"].nunique())

# ------------------------------------------------------------------
# 4) DBSCAN par semaine + collecte des outliers
# ------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# >>> NOUVEAU: ensemble global des outliers (un client outlier une semaine => retiré du dataset)
global_outlier_clients = set()

for (yy, ww) in WEEKS_SELECTED:
    print(f"\nDBSCAN semaine ({yy}, {ww})")

    wdf = df_sel[(df_sel["iso_year"] == yy) & (df_sel["iso_week"] == ww)].copy()
    wdf = wdf[wdf["dataid"].isin(selected_clients)].copy()

    X = compute_features_week(wdf)

    if len(X) < MINPTS:
        print(f"   ⚠ Pas assez de points (clients) pour DBSCAN: {len(X)}")
        continue

    features = ["f1", "f2", "f3", "f4", "f5_mod"]
    Xs = RobustScaler().fit_transform(X[features].to_numpy())

    eps0 = eps_from_kdist_percentile(Xs, k=MINPTS, q=EPS_PERCENTILE)
    eps = float(eps0 * EPS_BOOST)

    labels = DBSCAN(eps=eps, min_samples=MINPTS).fit_predict(Xs)

    tries = 0
    while (labels == -1).mean() > 0.35 and tries < 3:
        eps *= 1.15
        labels = DBSCAN(eps=eps, min_samples=MINPTS).fit_predict(Xs)
        tries += 1

    X["label"] = labels

    outliers = X.loc[X["label"] == -1, "dataid"].tolist()
    global_outlier_clients.update(outliers)

    n_out = len(outliers)
    print(f"   eps≈{eps:.3f} (base p{EPS_PERCENTILE}={eps0:.3f}, boost={EPS_BOOST}, tries={tries})")
    print(f"   outliers={n_out}/{len(X)}")
    print("   Outliers (-1):", outliers if outliers else "aucun")

    print("\nMatrice des caractéristiques (arrondie)")
    print(X.round(3))

    if DO_PLOT:
        Z = PCA(n_components=2, random_state=0).fit_transform(Xs)

        fig1, ax_db = plt.subplots(figsize=(8, 6))
        noise = (labels == -1)

        ax_db.scatter(Z[noise, 0], Z[noise, 1],
                      c="k", marker="x", s=90, linewidths=2, label="noise (-1)")

        cluster_ids = sorted([l for l in set(labels) if l != -1])
        cmap = plt.get_cmap("tab10", max(1, len(cluster_ids)))

        for i, cid in enumerate(cluster_ids):
            m = labels == cid
            ax_db.scatter(Z[m, 0], Z[m, 1], s=80, c=[cmap(i)], label=f"cluster {cid}")

        for i, cid_txt in enumerate(X["dataid"].astype(str).tolist()):
            ax_db.annotate(cid_txt, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.85)

        ax_db.set_title(f"DBSCAN semaine ({yy}, {ww}) — eps≈{eps:.3f}, minPts={MINPTS}")
        ax_db.set_xlabel("PC1")
        ax_db.set_ylabel("PC2")
        ax_db.grid(True, alpha=0.3)
        ax_db.legend(loc="best", fontsize=8)

        plt.tight_layout()
        plt.show()

# # ------------------------------------------------------------------
# # 5) EXPORT CSV FILTRÉ: on enlève du dataset tous les clients outliers
# #    (si un client est outlier dans au moins une semaine sélectionnée)
# #    -> le CSV de sortie garde les mêmes colonnes que le fichier d'origine
# # ------------------------------------------------------------------
# print("\n==============================")
# print("Filtrage final DBSCAN (union des outliers)")
# print("==============================")
# print(f"Nombre total de clients outliers retirés: {len(global_outlier_clients)}")
# print("Exemples:", sorted(list(global_outlier_clients))[:20], "..." if len(global_outlier_clients) > 20 else "")

# df_filtered = df.copy()
# df_filtered = df_filtered[~df_filtered["dataid"].isin(global_outlier_clients)].copy()

# print(f"Lignes avant: {len(df)} | Lignes après: {len(df_filtered)}")

# # Important: garder EXACTEMENT les mêmes colonnes que le fichier initial
# # (df a déjà des colonnes ajoutées iso_year/iso_week/date/dt; on écrit seulement les colonnes originales)
# df_original = pd.read_csv(DATA_PATH, nrows=0)
# orig_cols = df_original.columns.tolist()

# df_out = df_filtered[orig_cols].copy()
# df_out.to_csv(OUT_PATH_FILTERED, index=False)

# print(f"CSV filtré écrit: {OUT_PATH_FILTERED}")