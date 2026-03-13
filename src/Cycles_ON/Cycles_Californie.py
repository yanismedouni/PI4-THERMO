"""
Projet THERMO — Analyse des transitions ON/OFF (Climatisation California)
=========================================================================
Ce script :
1) Charge le CSV prétraité : output/processed_energy_data_california.csv
2) Filtre les clients California (liste fournie) et une période large (2015→2018)
3) Utilise la colonne `clim` (déjà prétraitée upstream)
4) Remplace les NaN de clim par 0 (interprété comme OFF)
5) Binarise ON/OFF via un seuil (kW)
6) Extrait les runs ON/OFF par client
7) Stats durées ON/OFF par saison
8) Cycles OFF→ON par jour et par client + stats saisonnières
9) Graphes :
   - Durée moyenne ON (heures) par mois
   - Duty cycle par saison
   - Profil journalier moyen par saison
   - Durée ON vs heure de démarrage
10) Sauvegarde les tables dans output/

Remarque :
- Tes clients viennent de 2015, 2016, 2018. On garde une plage large.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PARAMÈTRES
# =============================================================================

CHEMIN_CSV = "output/processed_energy_data_california.csv"
COL_DATE = "local_15min"
COL_CLIENT = "dataid"

COL_CLIM = "clim"

# Liste clients California (ceux fournis)
CLIENTS_CA = [
    1731, 4495, 8342,      # 2018
    3938, 4934, 5938, 8061, 9775,  # 2016
    203, 1450, 1524        # 2015
]

# Période large cohérente avec tes années
DATE_DEBUT = "2015-01-01 00:00:00"
DATE_FIN   = "2018-12-31 23:59:59"

PAS_TEMPS_MIN = 15

# Seuil ON/OFF — à ajuster selon tes niveaux de puissance California
SEUIL_ON_KW = 0.67

SAISONS = {
    "Hiver": [12, 1, 2],
    "Printemps": [3, 4, 5],
    "Été": [6, 7, 8],
    "Automne": [9, 10, 11],
}
ORDRE_SAISONS = ["Hiver", "Printemps", "Été", "Automne"]

MOIS_NOMS = {
    1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
    7: "Juil", 8: "Aoû", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
}


# =============================================================================
# OUTILS
# =============================================================================

def mois_vers_saison(mois: int) -> str:
    for saison, mois_list in SAISONS.items():
        if mois in mois_list:
            return saison
    return "Inconnu"


def extraire_runs_on_off(df_client: pd.DataFrame) -> pd.DataFrame:
    tcl_on = df_client["tcl_on"].to_numpy()

    change = np.empty_like(tcl_on)
    change[0] = 1
    change[1:] = (tcl_on[1:] != tcl_on[:-1]).astype(int)

    run_id = np.cumsum(change)

    df_tmp = df_client.copy()
    df_tmp["run_id"] = run_id

    runs = (
        df_tmp.groupby(["run_id", "tcl_on"], as_index=False)
              .agg(
                  run_start=(COL_DATE, "min"),
                  run_end=(COL_DATE, "max"),
                  n_steps=(COL_DATE, "size")
              )
    )

    runs["duration_min"] = runs["n_steps"] * PAS_TEMPS_MIN
    runs["run_state"] = runs["tcl_on"].map({0: "OFF", 1: "ON"})
    runs["season"] = runs["run_start"].dt.month.map(mois_vers_saison)
    runs[COL_CLIENT] = df_client[COL_CLIENT].iloc[0]

    return runs[[COL_CLIENT, "run_state", "run_start", "run_end", "n_steps", "duration_min", "season"]]


# =============================================================================
# MAIN
# =============================================================================

def main():

    # 1) Charger données
    df = pd.read_csv(CHEMIN_CSV, parse_dates=[COL_DATE])

    if COL_CLIM not in df.columns:
        raise ValueError(f"Colonne '{COL_CLIM}' introuvable dans {CHEMIN_CSV}")

    # 2) Filtrer clients + période
    df = df[df[COL_CLIENT].isin(CLIENTS_CA)].copy()
    df = df[(df[COL_DATE] >= DATE_DEBUT) & (df[COL_DATE] <= DATE_FIN)].copy()
    df = df.sort_values([COL_CLIENT, COL_DATE]).reset_index(drop=True)

    print(f"[INFO] Lignes après filtre clients+période: {len(df):,}")

    # 3) Préparer clim_total
    df["clim_total"] = pd.to_numeric(df[COL_CLIM], errors="coerce")
    df.loc[df["clim_total"] < 0, "clim_total"] = 0
    df["clim_total"] = df["clim_total"].fillna(0)

    # 4) ON/OFF
    df["tcl_on"] = (df["clim_total"] > SEUIL_ON_KW).astype(int)
    df["season"] = df[COL_DATE].dt.month.map(mois_vers_saison)
    df["hour"] = df[COL_DATE].dt.hour

    print(f"[INFO] Seuil ON/OFF = {SEUIL_ON_KW:.3f} kW (sur clim_total)")
    print(df["tcl_on"].value_counts().rename({0: "OFF", 1: "ON"}))

    # 5) Runs
    runs_list = []
    for cid, df_c in df.groupby(COL_CLIENT, sort=False):
        df_c = df_c[[COL_CLIENT, COL_DATE, "clim_total", "tcl_on"]].copy()
        df_c = df_c.sort_values(COL_DATE)
        df_c["tcl_on"] = df_c["tcl_on"].astype(int)
        runs_list.append(extraire_runs_on_off(df_c))

    runs_all = pd.concat(runs_list, ignore_index=True)
    runs_all["season"] = pd.Categorical(runs_all["season"], categories=ORDRE_SAISONS, ordered=True)

    print(f"\n[INFO] Nombre total de runs: {len(runs_all):,}")
    print(runs_all.head())

    # 6) Stats durées ON/OFF par saison
    stats_durations = (
        runs_all.groupby(["season", "run_state"], observed=False)["duration_min"]
                .agg(
                    n_runs="count",
                    mean_min="mean",
                    median_min="median",
                    min_min="min",
                    max_min="max"
                )
                .reset_index()
                .sort_values(["season", "run_state"])
    )

    # 7) Cycles OFF→ON par jour & client
    runs_all = runs_all.sort_values([COL_CLIENT, "run_start"]).reset_index(drop=True)
    runs_all["prev_state"] = runs_all.groupby(COL_CLIENT)["run_state"].shift(1)
    runs_all["is_cycle"] = (
        (runs_all["run_state"] == "ON") &
        (runs_all["prev_state"] == "OFF")
    ).astype(int)

    runs_all["date"] = runs_all["run_start"].dt.date

    daily_cycles_client = (
        runs_all[runs_all["is_cycle"] == 1]
        .groupby(["season", COL_CLIENT, "date"], observed=False)
        .size()
        .reset_index(name="n_cycles")
    )

    daily_cycles_client["season"] = pd.Categorical(
        daily_cycles_client["season"], categories=ORDRE_SAISONS, ordered=True
    )

    stats_cycles = (
        daily_cycles_client.groupby("season", observed=False)["n_cycles"]
        .agg(
            n_obs="count",
            mean_cycles="mean",
            median_cycles="median",
            min_cycles="min",
            max_cycles="max"
        )
        .reset_index()
        .sort_values("season")
    )

    # 8) Graphe : durée moyenne ON par mois
    runs_on = runs_all[runs_all["run_state"] == "ON"].copy()
    runs_on["month"] = runs_on["run_start"].dt.month

    stats_cycle_mois = (
        runs_on.groupby("month")["duration_min"]
               .agg(
                   n_cycles="count",
                   mean_duration_min="mean",
                   median_duration_min="median",
                   min_duration_min="min",
                   max_duration_min="max"
               )
               .reset_index()
               .sort_values("month")
    )

    stats_cycle_mois["mean_duration_h"] = stats_cycle_mois["mean_duration_min"] / 60.0
    mois_x = stats_cycle_mois["month"].tolist()
    mois_labels = [MOIS_NOMS[m] for m in mois_x]

    plt.figure(figsize=(10, 5))
    plt.plot(mois_x, stats_cycle_mois["mean_duration_h"], marker="o")
    plt.xticks(mois_x, mois_labels)
    plt.xlabel("Mois")
    plt.ylabel("Durée moyenne d'un cycle ON (heures)")
    plt.title("California — Durée moyenne d'un cycle ON (clim) par mois")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 9) Graphe : Duty cycle par saison
    duty_saison = (
        df.groupby("season", observed=False)["tcl_on"]
          .mean()
          .reindex(ORDRE_SAISONS)
    )

    plt.figure(figsize=(8, 5))
    duty_saison.plot(kind="bar")
    plt.ylabel("Duty cycle moyen (fraction du temps ON)")
    plt.title("California — Duty cycle (clim ON/total) par saison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # 10) Graphe : Profil journalier moyen par saison
    profil = (
        df.groupby(["season", "hour"], observed=False)["clim_total"]
          .mean()
          .reset_index()
    )

    plt.figure(figsize=(10, 6))
    for saison in ORDRE_SAISONS:
        data_s = profil[profil["season"] == saison]
        if len(data_s) > 0:
            plt.plot(data_s["hour"], data_s["clim_total"], label=saison)

    plt.xlabel("Heure de la journée")
    plt.ylabel("Puissance moyenne clim (kW)")
    plt.title("California — Profil journalier moyen de la clim par saison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 11) Graphe : Durée ON vs heure de démarrage
    runs_on = runs_all[runs_all["run_state"] == "ON"].copy()
    runs_on["start_hour"] = runs_on["run_start"].dt.hour
    runs_on["duration_h"] = runs_on["duration_min"] / 60.0

    duree_par_heure = (
        runs_on.groupby("start_hour")["duration_h"]
               .mean()
               .reset_index()
               .sort_values("start_hour")
    )

    plt.figure(figsize=(10, 5))
    plt.plot(duree_par_heure["start_hour"], duree_par_heure["duration_h"], marker="o")
    plt.xlabel("Heure de démarrage du cycle ON")
    plt.ylabel("Durée moyenne ON (heures)")
    plt.title("California — Durée ON vs heure de démarrage (clim)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 12) Sauvegarde CSV + commentaires
    # (A) Runs ON/OFF : séquences consécutives ON/OFF par client
    runs_all.to_csv("output/california_runs_on_off.csv", index=False)

    # (B) Stats durées ON/OFF par saison : n_runs + durées mean/median/min/max
    stats_durations.to_csv("output/california_stats_durees_on_off_par_saison.csv", index=False)

    # (C) Cycles OFF→ON par client/jour : n_cycles par (saison, client, date)
    daily_cycles_client.to_csv("output/california_daily_cycles_par_client.csv", index=False)

    # (D) Stats cycles par saison : résumé de n_cycles/jour
    stats_cycles.to_csv("output/california_stats_cycles_par_saison.csv", index=False)

    # (E) Durée ON par mois : résumé mensuel des runs ON
    stats_cycle_mois.to_csv("output/california_duree_cycle_on_par_mois.csv", index=False)

    print("\n[OK] Fichiers générés :")
    print(" - output/california_runs_on_off.csv")
    print(" - output/california_stats_durees_on_off_par_saison.csv")
    print(" - output/california_daily_cycles_par_client.csv")
    print(" - output/california_stats_cycles_par_saison.csv")
    print(" - output/california_duree_cycle_on_par_mois.csv")


if __name__ == "__main__":
    main()