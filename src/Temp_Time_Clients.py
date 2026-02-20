# # Un seul client, toutes les semaines, pour visualiser les patterns temporels de consommation et température.

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # ------------------------------------------------------------------
# # CHEMIN
# # ------------------------------------------------------------------
# DATA_PATH = r"C:\Users\Samia\OneDrive - polymtlus\Bureau\DataThermo\austin\15minute_data_austin_modified.csv"

# # ------------------------------------------------------------------
# # Paramètres
# # ------------------------------------------------------------------
# ISO_YEAR = 2018
# WEEKS = [1, 2, 3, 4, 5, 6, 7]
# CLIENT_ID = 7951

# # ------------------------------------------------------------------
# # Lecture + préparation
# # ------------------------------------------------------------------
# df = pd.read_csv(DATA_PATH)

# df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
# df["furnace1"] = pd.to_numeric(df["furnace1"], errors="coerce")
# df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

# df["dt"] = pd.to_datetime(
#     df[["year", "month", "day", "hour", "minute"]].astype(int),
#     errors="coerce"
# )

# df = df.dropna(subset=["dt", "grid", "furnace1", "temp", "dataid"])

# iso = df["dt"].dt.isocalendar()
# df["iso_year"] = iso["year"].astype(int)
# df["iso_week"] = iso["week"].astype(int)

# # ------------------------------------------------------------------
# # Filtre client + semaines
# # ------------------------------------------------------------------
# df_client = df[
#     (df["dataid"] == CLIENT_ID) &
#     (df["iso_year"] == ISO_YEAR) &
#     (df["iso_week"].isin(WEEKS))
# ].copy()

# df_client = df_client.sort_values("dt")

# if df_client.empty:
#     raise RuntimeError("Aucune donnée trouvée pour ce client et ces semaines.")

# # ------------------------------------------------------------------
# # Plot : 1 figure, 7 sous-graphes (un par semaine)
# # ------------------------------------------------------------------
# n = len(WEEKS)
# fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, 3.2 * n), sharex=False)

# if n == 1:
#     axes = [axes]

# for ax1, ww in zip(axes, WEEKS):
#     g = df_client[df_client["iso_week"] == ww].copy().sort_values("dt")

#     if g.empty:
#         ax1.set_title(f"Semaine ISO {ww} ({ISO_YEAR}) — aucune donnée")
#         ax1.axis("off")
#         continue

#     # Grid + Furnace1 (axe gauche)
#     ax1.plot(g["dt"], g["grid"], linewidth=1.2, label="Grid")
#     ax1.plot(g["dt"], g["furnace1"], linewidth=1.2, label="Furnace1")
#     ax1.set_ylabel("Puissance (kW)")
#     ax1.grid(True, alpha=0.3)
#     ax1.tick_params(axis="x", rotation=30, labelsize=8)
#     ax1.tick_params(axis="y", labelsize=8)

#     # Température (axe droit)
#     ax2 = ax1.twinx()
#     ax2.plot(g["dt"], g["temp"], linewidth=1.2, alpha=0.8, label="Température")
#     ax2.set_ylabel("Temp (°C)")
#     ax2.tick_params(axis="y", labelsize=8)

#     # Légende combinée
#     l1, lab1 = ax1.get_legend_handles_labels()
#     l2, lab2 = ax2.get_legend_handles_labels()
#     ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=8)

#     ax1.set_title(f"Client {CLIENT_ID} — Semaine ISO {ww} ({ISO_YEAR})", fontsize=10)

# fig.suptitle(
#     f"Client {CLIENT_ID} — Semaines ISO {WEEKS} ({ISO_YEAR})\nGrid, Furnace1, Température",
#     fontsize=14
# )
# plt.tight_layout()
# plt.show()

# tous les clients, sur 3 semaines, pour visualiser les patterns temporels de consommation et température.

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
WEEKS = [1, 2, 3]

N_CLIENTS = 25
NROWS, NCOLS = 5, 5
FIGSIZE = (20, 14)

# ------------------------------------------------------------------
# Lecture + préparation
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

cols_needed = ["grid", "furnace1", "furnace2", "temp"]
for c in cols_needed:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["dt"] = pd.to_datetime(
    df[["year", "month", "day", "hour", "minute"]].astype(int),
    errors="coerce"
)

df = df.dropna(subset=["dt", "dataid"])

iso = df["dt"].dt.isocalendar()
df["iso_year"] = iso["year"].astype(int)
df["iso_week"] = iso["week"].astype(int)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
for ISO_WEEK in WEEKS:

    df_week = df[
        (df["iso_year"] == ISO_YEAR) &
        (df["iso_week"] == ISO_WEEK)
    ].copy()

    df_week = df_week.sort_values("dt")

    if df_week.empty:
        print(f"Aucune donnée semaine {ISO_WEEK}")
        continue

    # sélectionner top 25 clients les plus complets
    counts = df_week.groupby("dataid").size().sort_values(ascending=False)
    selected_clients = counts.head(N_CLIENTS).index.tolist()

    print(f"Semaine {ISO_WEEK} — Clients sélectionnés: {len(selected_clients)}")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, client_id in enumerate(selected_clients):
        ax1 = axes[i]

        g = df_week[df_week["dataid"] == client_id].copy()
        g = g.sort_values("dt")

        if g.empty:
            ax1.axis("off")
            continue

        # Axe gauche
        ax1.plot(g["dt"], g["grid"], linewidth=1.0, label="grid")
        ax1.plot(g["dt"], g["furnace1"], linewidth=1.0, label="furnace1")
        ax1.plot(g["dt"], g["furnace2"], linewidth=1.0, label="furnace2")

        ax1.set_title(str(client_id), fontsize=9)
        ax1.grid(True, alpha=0.25)
        ax1.tick_params(axis="x", rotation=30, labelsize=7)
        ax1.tick_params(axis="y", labelsize=7)

        # Axe droit (température)
        ax2 = ax1.twinx()
        ax2.plot(g["dt"], g["temp"], linewidth=1.0, alpha=0.6, label="temp")
        ax2.tick_params(axis="y", labelsize=7)

    # cacher axes inutilisés
    for j in range(len(selected_clients), NROWS * NCOLS):
        axes[j].axis("off")

    # légende globale
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[0].twinx().get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2,
               loc="upper right", fontsize=9)

    fig.suptitle(
        f"Semaine ISO {ISO_WEEK} ({ISO_YEAR})\nGrid, Furnace1, Furnace2 & Température",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()