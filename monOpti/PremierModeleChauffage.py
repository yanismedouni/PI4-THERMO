# ============================================================
# MIQP (CVXPY + MOSEK) — 1 client, 1 semaine (début 2018)
# 1 seul appareil chauffant: HEAT ∈ {0, 0.10, 0.20, 0.45}
# Autorisé quand temp < 5°C (interdit sinon)
# d_min = 2 (30 minutes)
# Plot final: furnace1 (réel) vs p_heat_est (estimé)
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import mosek  # requis pour solver=cp.MOSEK

# -------------------------
# PATHS
# -------------------------
CSV_PATH = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\austin\15minute_data_austin_modified.csv"

# -------------------------
# COLONNES (TON fichier)
# -------------------------
COL_TIME   = "datetime"
COL_CLIENT = "dataid"
COL_PTOTAL = "grid"
COL_TEXT   = "temp"
COL_TRUE   = "furnace1"

# -------------------------
# SELECTION
# -------------------------
CLIENT_ID = None  # mets un id précis (ex: 12345). Si None -> prend le 1er client dispo
WEEK_START = "2018-01-01 00:00:00"
FREQ = "15min"

# -------------------------
# Paramètres modèle
# -------------------------
LEVELS_HEAT = np.array([0.0, 0.10, 0.20, 0.45], dtype=float)  # kW
TEMP_THRESHOLD = 10
LAMBDA_1 = 0.05
D_MIN = 4  

# ============================================================
# Helpers
# ============================================================
def enforce_week_grid(g: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    t1 = t0 + pd.Timedelta(days=7)
    expected = pd.date_range(t0, t1 - pd.Timedelta(minutes=15), freq=FREQ)

    g = g.copy()
    g[COL_TIME] = pd.to_datetime(g[COL_TIME])
    g = g[(g[COL_TIME] >= t0) & (g[COL_TIME] < t1)].sort_values(COL_TIME)

    g = g.set_index(COL_TIME).reindex(expected).reset_index().rename(columns={"index": COL_TIME})
    return g

def clean_numeric(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = g.copy()
    for c in cols:
        g[c] = pd.to_numeric(g[c], errors="coerce")
        g[c] = g[c].interpolate(limit=4).ffill().bfill()
    return g

# ============================================================
# Load
# ============================================================
df = pd.read_csv(CSV_PATH)

needed = [COL_TIME, COL_CLIENT, COL_PTOTAL, COL_TEXT, COL_TRUE]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes: {missing}\nColonnes dispo: {df.columns.tolist()}")

df[COL_TIME] = pd.to_datetime(df[COL_TIME])

clients = sorted(df[COL_CLIENT].dropna().unique().tolist())
if not clients:
    raise ValueError("Aucun client trouvé dans la colonne dataid.")
if CLIENT_ID is None:
    CLIENT_ID = clients[0]

g = df[df[COL_CLIENT] == CLIENT_ID].copy()
if g.empty:
    raise ValueError(f"CLIENT_ID={CLIENT_ID} introuvable.")

t0 = pd.Timestamp(WEEK_START)
g = enforce_week_grid(g, t0)
if g[COL_PTOTAL].isna().all():
    raise ValueError(f"Aucune donnée pour CLIENT_ID={CLIENT_ID} sur la semaine commençant {t0}.")

g = clean_numeric(g, [COL_PTOTAL, COL_TEXT, COL_TRUE])

# ============================================================
# Données semaine
# ============================================================
ts      = pd.to_datetime(g[COL_TIME]).to_numpy()
p_total = g[COL_PTOTAL].to_numpy(float)
t_ext   = g[COL_TEXT].to_numpy(float)
p_true  = g[COL_TRUE].to_numpy(float)

T = len(g)
T_set = range(T)

# chauffage autorisé si temp < 5°C
u_param = (t_ext < TEMP_THRESHOLD).astype(int)

# ============================================================
# Variables
# ============================================================
p_base = cp.Variable(T, nonneg=True)
B = cp.Variable(T)

# x_level[t,k] = 1 si niveau k actif à t (one-hot)
x_level = cp.Variable((T, len(LEVELS_HEAT)), boolean=True)

# ON/OFF pour gérer cycles + autorisation (ON = niveau non-zero)
x_on = cp.Variable(T, boolean=True)

# cycles
s = cp.Variable(T, boolean=True)
f = cp.Variable(T, boolean=True)

# ============================================================
# Contraintes
# ============================================================
constraints = []

# baseload diff
constraints += [B[0] == 0]
for t in range(1, T):
    constraints += [B[t] == p_base[t] - p_base[t-1]]

# one-hot niveaux
for t in T_set:
    constraints += [cp.sum(x_level[t, :]) == 1]

# ON = somme niveaux non-zero
idx_nonzero = [k for k, val in enumerate(LEVELS_HEAT) if abs(val) > 1e-12]
for t in T_set:
    constraints += [x_on[t] == cp.sum(x_level[t, idx_nonzero])]

# autorisation température
for t in T_set:
    constraints += [x_on[t] <= u_param[t]]

# dynamique cycles
for t in T_set:
    if t == 0:
        constraints += [x_on[0] == s[0] - f[0]]  # x(-1)=0
    else:
        constraints += [x_on[t] - x_on[t-1] == s[t] - f[t]]
    constraints += [s[t] + f[t] <= 1]

# anti-switching: max 1 transition dans une fenêtre D_MIN
if D_MIN and D_MIN > 0:
    for t in range(0, T - (D_MIN - 1)):
        constraints += [cp.sum(s[t:t+D_MIN] + f[t:t+D_MIN]) <= 1]

# ============================================================
# Objectif (MIQP)
# p_heat[t] = sum_k x_level[t,k] * LEVELS_HEAT[k]
# ============================================================
p_heat = x_level @ LEVELS_HEAT
residual = p_total - p_base - p_heat

cost = cp.sum_squares(residual) + float(LAMBDA_1) * cp.norm1(B)

prob = cp.Problem(cp.Minimize(cost), constraints)

# ============================================================
# Solve
# ============================================================
obj = prob.solve(
    solver=cp.MOSEK,
    verbose=True,
    mosek_params={
        "MSK_DPAR_MIO_TOL_REL_GAP": 1e-2,
        "MSK_DPAR_MIO_MAX_TIME": 120, # à regarder
    }
)

print("client:", CLIENT_ID)
print("status:", prob.status)
print("obj:", obj)

# ============================================================
# Plot demandé: furnace1 (réel) vs chauffage estimé (puissance)
# ============================================================
p_heat_est = (x_level.value @ LEVELS_HEAT).astype(float)

plt.figure()
plt.plot(ts, p_true, label="furnace1 (réel)")
plt.plot(ts, p_heat_est, label="chauffage estimé (kW)")
plt.title(f"Client {CLIENT_ID} — 1 semaine dès {t0.date()} — niveaux {LEVELS_HEAT.tolist()} kW, autorisé si T<5°C")
plt.xlabel("datetime")
plt.ylabel("power (kW)")
plt.legend()
plt.tight_layout()
plt.show()