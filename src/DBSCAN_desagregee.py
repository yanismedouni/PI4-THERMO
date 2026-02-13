# ============================================================
# PHASE 1 (désagrégée) — Features p=7 + DBSCAN outliers
# + 3 semaines (Austin, 2018) + 100% clients + DBSCAN par semaine
# (sans exports, plots seulement)
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# -------------------------
# CHEMIN (Windows)
# -------------------------
DATA_PATH = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\austin\15minute_data_austin_modified.csv"

# -------------------------
# Paramètres
# -------------------------
# blocs horaires (Austin local)
B1 = ("06:00", "10:00")
B2 = ("13:00", "17:00")
B3 = ("17:00", "22:00")

# seuil ON/OFF sur signal HVAC désagrégé (kWh / 15 min)
TAU = 0.30

# seuils chaud/froid (°C)
T_CHAUDE = 20.0
T_FROIDE = 10.0

# règle NaN: interpolation seulement si <=2 NaN consécutifs
MAX_CONSEC_NAN = 2

# DBSCAN
MINPTS = 4  # typiquement 3..5

# protocole
RANDOM_SEED = 42
FRAC_CLIENTS = 1
N_WEEKS = 3

# qualité minimale semaine-client
MIN_POINTS_WEEK = int(7 * 96 * 0.75)

# affichage
DO_PLOT = True

X_cols = ["f1","f2","f3","f4","f5","f6","f7"]

# ============================================================
# Utils
# ============================================================

def longest_nan_run(x: pd.Series) -> int:
    """Longueur max d'une séquence de NaN consécutifs."""
    is_na = x.isna().to_numpy(dtype=bool)
    if not is_na.any():
        return 0
    run = 0
    best = 0
    for v in is_na:
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best

def interp_if_ok(x: pd.Series, max_consec_nan: int) -> tuple[pd.Series, int]:
    """
    Interpole si les runs de NaN sont <= max_consec_nan.
    Retourne (série_interpolée, flag_bad=1 si run > max_consec_nan).
    """
    bad = 1 if longest_nan_run(x) > max_consec_nan else 0
    if bad:
        return x, bad
    x2 = x.interpolate(method="time", limit=max_consec_nan, limit_direction="both")
    x2 = x2.ffill().bfill()
    return x2, 0

def ensure_96_points(day_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Force une grille 15 min complète sur la journée locale (tz-aware)."""
    t0 = day_df[time_col].iloc[0]
    day_start = t0.floor("D")
    idx = pd.date_range(day_start, periods=96, freq="15min", tz=t0.tz)
    out = day_df.set_index(time_col).reindex(idx)
    out.index.name = time_col
    return out.reset_index()

def knee_point(sorted_vals: np.ndarray) -> float:
    """Coude via distance max à la droite reliant le premier et le dernier point."""
    y = np.asarray(sorted_vals, dtype=float)
    n = y.size
    if n == 0:
        return 0.0
    if n < 3:
        return float(np.max(y))

    x = np.arange(n)
    x1, y1 = 0, y[0]
    x2, y2 = n - 1, y[-1]

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    dist = np.abs(a * x + b * y + c) / np.sqrt(a * a + b * b)
    idx = int(np.argmax(dist))
    return float(y[idx])

def k_distance_eps(Xs: np.ndarray, k: int) -> float:
    """k-distance triée + coude."""
    nn = NearestNeighbors(n_neighbors=max(2, k + 1), metric="euclidean")
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    kd = dists[:, k]  # index k car 0 = soi-même
    kd_sorted = np.sort(kd)
    return knee_point(kd_sorted)

def parse_hhmm(hhmm: str):
    return pd.to_datetime(hhmm).time()

# ============================================================
# 1) Load data (désagrégée)
# ============================================================

df = pd.read_csv(DATA_PATH)

# --- déterminer automatiquement la colonne temps
if "local_15min" in df.columns:
    time_col = "local_15min"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
else:
    needed_time = {"year","month","day","hour","minute"}
    if not needed_time.issubset(df.columns):
        raise ValueError("Aucune colonne temps utilisable: il faut local_15min OU year/month/day/hour/minute.")
    df["dt"] = pd.to_datetime(df[["year","month","day","hour","minute"]].astype(int), errors="coerce", utc=True)
    time_col = "dt"

df = df.dropna(subset=[time_col, "dataid"]).copy()
df["dataid"] = pd.to_numeric(df["dataid"], errors="coerce").astype("Int64")
df = df.dropna(subset=["dataid"]).copy()
df["dataid"] = df["dataid"].astype(int)

# --- colonnes désagrégées (air/heat)
air_cols  = [c for c in ["air1","air2","air3"] if c in df.columns]
heat_cols = [c for c in ["furnace1","furnace2","heater1","heater2","heater3"] if c in df.columns]

if len(air_cols) == 0 and len(heat_cols) == 0:
    raise ValueError("Aucune colonne désagrégée trouvée (air*/furnace*/heater*). Ton fichier ne contient pas la donnée désagrégée.")

# --- P_total (grid+solar)
if "grid" not in df.columns:
    raise ValueError("Colonne 'grid' manquante (nécessaire à P_total).")
if "solar" not in df.columns:
    df["solar"] = 0.0

# --- temp
has_temp = ("temp" in df.columns)
if has_temp:
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

# --- numeric + négatifs -> 0
for c in air_cols + heat_cols + ["grid","solar"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df[c] < 0, c] = 0

# --- construit signaux
df["P_air"]   = df[air_cols].sum(axis=1, min_count=1) if len(air_cols) else 0.0
df["P_heat"]  = df[heat_cols].sum(axis=1, min_count=1) if len(heat_cols) else 0.0
df["P_hvac"]  = df[["P_air","P_heat"]].sum(axis=1, min_count=1)
df["P_total"] = df[["grid","solar"]].sum(axis=1, min_count=1)

# ============================================================
# 2) ISO week + Austin local time for blocks
# ============================================================

iso = df[time_col].dt.isocalendar()
df["iso_year"] = iso["year"].astype(int)
df["iso_week"] = iso["week"].astype(int)

df["t_local"] = df[time_col].dt.tz_convert("America/Chicago")
df["date_local"] = df["t_local"].dt.floor("D")

df_2018 = df[df["iso_year"] == 2018].copy()
if df_2018.empty:
    raise RuntimeError("Aucune donnée en 2018 dans ce fichier.")

# ============================================================
# 3) Sélection 3 semaines (2018) les plus riches en clients
# ============================================================

week_counts = (
    df_2018.groupby(["iso_year", "iso_week"])["dataid"]
           .nunique()
           .sort_values(ascending=False)
)

selected_weeks = week_counts.head(N_WEEKS).index.tolist()
print("Semaines sélectionnées (2018, top clients uniques):", selected_weeks)

df_sel = df_2018[df_2018.set_index(["iso_year","iso_week"]).index.isin(selected_weeks)].copy()

# ============================================================
# 4) Sélection des clients (FRAC_CLIENTS)
# ============================================================

all_clients = pd.Series(df_sel["dataid"].dropna().unique())
selected_clients = all_clients.sample(frac=FRAC_CLIENTS, random_state=RANDOM_SEED).to_numpy()
print(f"Clients sélectionnés: {len(selected_clients)} / {len(all_clients)}")

df_sel = df_sel[df_sel["dataid"].isin(selected_clients)].copy()

print("Clients par semaine (après filtrage):")
print(df_sel.groupby(["iso_year","iso_week"])["dataid"].nunique())

# ============================================================
# 5) Features p=7 — f1-4 sur P_total ; f5-7 sur P_hvac (désagrégé)
# ============================================================

def compute_features_for_client_week(g: pd.DataFrame) -> dict | None:
    g = g.sort_values("t_local").copy()

    day_feats = []
    hot_vals = []
    cold_vals = []

    start1, end1 = parse_hhmm(B1[0]), parse_hhmm(B1[1])
    start2, end2 = parse_hhmm(B2[0]), parse_hhmm(B2[1])
    start3, end3 = parse_hhmm(B3[0]), parse_hhmm(B3[1])

    for day, gd in g.groupby("date_local", sort=False):
        cols = ["t_local", "P_total", "P_hvac"]
        if has_temp:
            cols.append("temp")
        gd2 = gd[cols].copy()

        gd2 = ensure_96_points(gd2, "t_local").set_index("t_local")

        Ptot, bad1 = interp_if_ok(gd2["P_total"], MAX_CONSEC_NAN)
        Phvac, bad2 = interp_if_ok(gd2["P_hvac"], MAX_CONSEC_NAN)
        if bad1 or bad2:
            continue

        if has_temp:
            temp = pd.to_numeric(gd2["temp"], errors="coerce")
            temp_i = temp if temp.isna().all() else temp.interpolate(method="time", limit_direction="both").ffill().bfill()
        else:
            temp_i = None

        idx = Ptot.index

        def mean_in_block(start_t, end_t) -> float:
            m = (idx.time >= start_t) & (idx.time < end_t)
            return float(np.nanmean(Ptot.to_numpy()[m])) if m.any() else np.nan

        m1 = mean_in_block(start1, end1)
        m2 = mean_in_block(start2, end2)
        m3 = mean_in_block(start3, end3)

        Pmax = float(np.nanmax(Ptot))
        Pmin = float(np.nanmin(Ptot))
        r = 0.0 if (not np.isfinite(Pmax) or Pmax <= 0) else (Pmax - Pmin) / Pmax

        s = (Phvac > TAU).astype(int).to_numpy()
        n_sw = int(np.sum(np.abs(s[1:] - s[:-1])))

        if has_temp and (temp_i is not None) and (not temp_i.isna().all()):
            th = temp_i.to_numpy(dtype=float)
            hot_mask = th >= T_CHAUDE
            cold_mask = th <= T_FROIDE
            if hot_mask.any():
                hot_vals.extend(s[hot_mask].tolist())
            if cold_mask.any():
                cold_vals.extend(s[cold_mask].tolist())

        day_feats.append((m1, m2, m3, r, n_sw))

    if len(day_feats) < 3:
        return None

    day_feats = np.array(day_feats, dtype=float)

    f1 = float(np.nanmean(day_feats[:, 0]))
    f2 = float(np.nanmean(day_feats[:, 1]))
    f3 = float(np.nanmean(day_feats[:, 2]))
    f4 = float(np.nanmedian(day_feats[:, 3]))
    f5 = float(np.nanmedian(day_feats[:, 4]))
    f6 = float(np.mean(hot_vals)) if len(hot_vals) > 0 else 0.0
    f7 = float(np.mean(cold_vals)) if len(cold_vals) > 0 else 0.0

    return {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6, "f7": f7}

# ============================================================
# 6) Features client–semaine (sur les 3 semaines)
# ============================================================

rows = []
for (cid, y, w), g in df_sel.groupby(["dataid", "iso_year", "iso_week"], sort=False):
    if len(g) < MIN_POINTS_WEEK:
        continue
    feats = compute_features_for_client_week(g)
    if feats is None:
        continue
    feats.update({"dataid": int(cid), "iso_year": int(y), "iso_week": int(w)})
    rows.append(feats)

feat_all = pd.DataFrame(rows)
if feat_all.empty:
    raise RuntimeError("Aucune semaine exploitable après filtrage (jours valides insuffisants).")

# ============================================================
# 7) DBSCAN — un par semaine + plots PCA
# ============================================================

for (y, w) in selected_weeks:
    wk = feat_all[(feat_all["iso_year"] == y) & (feat_all["iso_week"] == w)].copy()

    if len(wk) < MINPTS + 1:
        print(f"Semaine {(y,w)}: pas assez d'observations ({len(wk)}) pour DBSCAN fiable.")
        continue

    X = wk[X_cols].to_numpy(dtype=float)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    eps = k_distance_eps(Xs, k=MINPTS)
    labels = DBSCAN(eps=eps, min_samples=MINPTS, metric="euclidean").fit_predict(Xs)

    wk["label_dbscan"] = labels
    n_out = int((labels == -1).sum())

    print(f"\nDBSCAN semaine ISO {(y,w)}")
    print(f"  eps≈{eps:.3f}, outliers={n_out}/{len(labels)}")
    print("  clusters:", sorted([l for l in set(labels) if l != -1]))

    if DO_PLOT:
        Z = PCA(n_components=2, random_state=0).fit_transform(Xs)

        fig, ax = plt.subplots(figsize=(8, 6))
        noise = (labels == -1)
        ax.scatter(Z[noise, 0], Z[noise, 1],
                   c="k", marker="x", s=90, linewidths=2, label="noise (-1)")

        cluster_ids = sorted([l for l in set(labels) if l != -1])
        cmap = plt.get_cmap("tab10", max(1, len(cluster_ids)))

        for i, lab in enumerate(cluster_ids):
            m = labels == lab
            ax.scatter(Z[m, 0], Z[m, 1], s=80, c=[cmap(i)], label=f"cluster {lab}")

        for i, cid_txt in enumerate(wk["dataid"].astype(str).tolist()):
            ax.annotate(cid_txt, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.85)

        ax.set_title(f"DBSCAN PCA — semaine ISO {(y,w)} (Austin 2018)\n eps≈{eps:.3f}, MinPts={MINPTS}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()
