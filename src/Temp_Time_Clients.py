import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CHEMIN
# ------------------------------------------------------------------
DATA_PATH = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\austin\15minute_data_austin_modified.csv"

# ------------------------------------------------------------------
# Paramètres
# ------------------------------------------------------------------
ISO_YEAR = 2018
ISO_WEEK = 3
N_CLIENTS = 25
NROWS, NCOLS = 5, 5
FIGSIZE = (20, 14)

# ------------------------------------------------------------------
# Lecture + préparation
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

df["dt"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]].astype(int), errors="coerce")
df = df.dropna(subset=["dt", "grid", "temp", "dataid"])

iso = df["dt"].dt.isocalendar()
df["iso_year"] = iso["year"].astype(int)
df["iso_week"] = iso["week"].astype(int)

# Filtre semaine
df_week = df[(df["iso_year"] == ISO_YEAR) & (df["iso_week"] == ISO_WEEK)].copy()
df_week = df_week.sort_values("dt")

if df_week.empty:
    raise RuntimeError(f"Aucune donnée trouvée pour la semaine ISO {ISO_WEEK} ({ISO_YEAR}).")

# ------------------------------------------------------------------
# Sélection automatique des 25 clients les plus complets sur la semaine
# ------------------------------------------------------------------
counts = df_week.groupby("dataid").size().sort_values(ascending=False)
selected_clients = counts.head(N_CLIENTS).index.tolist()

print(f"Semaine ISO {ISO_WEEK} ({ISO_YEAR})")
print(f"Clients disponibles: {counts.size}")
print(f"Clients sélectionnés (top {N_CLIENTS} par nb de points): {selected_clients}")

# ------------------------------------------------------------------
# Plot : 25 subplots (consommation gauche + temp droite)
# ------------------------------------------------------------------
fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, sharex=False, sharey=False)
axes = np.array(axes).reshape(-1)

for i, client_id in enumerate(selected_clients[:NROWS * NCOLS]):
    ax1 = axes[i]

    g = df_week[df_week["dataid"] == client_id].copy()
    g = g.sort_values("dt")

    if g.empty:
        ax1.axis("off")
        continue

    # Consommation (gauche)
    ax1.plot(g["dt"], g["grid"], linewidth=1.0)
    ax1.set_title(f"{client_id}", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax1.tick_params(axis="y", labelsize=7)

    # Température (droite)
    ax2 = ax1.twinx()
    ax2.plot(g["dt"], g["temp"], linewidth=1.0, alpha=0.7)
    ax2.tick_params(axis="y", labelsize=7)

# Cacher axes inutilisés (si jamais <25)
for j in range(len(selected_clients), NROWS * NCOLS):
    axes[j].axis("off")

fig.suptitle(
    f"{len(selected_clients)} clients — Semaine ISO {ISO_WEEK} ({ISO_YEAR})\n"
    f"Consommation (gauche) & Température (droite)",
    fontsize=14
)
plt.tight_layout()
plt.show()
