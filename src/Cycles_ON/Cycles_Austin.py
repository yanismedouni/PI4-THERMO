"""
Projet THERMO — Analyse des transitions ON/OFF (Climatisation Austin 2018)
=========================================================================
Objectif
--------
Analyser les cycles ON/OFF de la climatisation (TCL) à partir du CSV prétraité
`output/processed_energy_data_austin.csv`, en se basant sur la colonne `clim`.

Le script :
1) Charge le CSV prétraité
2) Filtre les clients Austin (liste fournie) et la période 2018
3) Utilise directement la colonne `clim` (déjà prétraitée)
4) Remplace les NaN de `clim` par 0 (interprété comme OFF)
5) Binarise ON/OFF via un seuil (kW)
6) Extrait les runs (séquences consécutives) ON et OFF par client
7) Calcule des stats des durées ON/OFF par saison
8) Compte les transitions OFF→ON par jour et par client + stats saisonnières
9) Graphe : durée moyenne d’un cycle ON (heures) en fonction du mois
10) Graphe : duty cycle par saison
11) Graphe : profil journalier moyen par saison (moyenne tous clients)
12) Graphe : durée ON vs heure de démarrage
13) Sauvegarde plusieurs CSV d’analyse dans output/


----------------------
- Le pas de temps est de 15 min.
- `clim` est en kW  et représente un canal agrégé.
- Les NaN de clim sont remplacés par 0 : on considère que donnée absente = OFF (après spline cubique).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PARAMÈTRES
# =============================================================================

# CSV prétraité (Austin)
CHEMIN_CSV = "output/processed_energy_data_austin.csv"

# Noms de colonnes dans le CSV
COL_DATE = "local_15min"
COL_CLIENT = "dataid"

# Colonne TCL analysée (climatisation agrégée)
COL_CLIM = "clim"

# Clients Austin retenus
CLIENTS_AUSTIN = [
    661, 1642, 2335, 2361, 2818, 3039,
    3456, 3538, 4031, 4373, 4767, 5746
]

# Période Austin 2018
DATE_DEBUT = "2018-01-01 00:00:00"
DATE_FIN   = "2018-12-31 23:59:59"

# Pas de temps (minutes)
PAS_TEMPS_MIN = 15

# Seuil ON/OFF (kW)
# Ici : basé sur le cluster ON_bas (0.480 kW)
SEUIL_ON_KW = 0.480

# Définition saisons (par mois)
SAISONS = {
    "Hiver": [12, 1, 2],
    "Printemps": [3, 4, 5],
    "Été": [6, 7, 8],
    "Automne": [9, 10, 11],
}
ORDRE_SAISONS = ["Hiver", "Printemps", "Été", "Automne"]

# Labels mois (affichage)
MOIS_NOMS = {
    1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
    7: "Juil", 8: "Aoû", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
}


# =============================================================================
# OUTILS
# =============================================================================

def mois_vers_saison(mois: int) -> str:
    """
    Convertit un mois (1..12) en saison selon le mapping SAISONS.
    """
    for saison, mois_list in SAISONS.items():
        if mois in mois_list:
            return saison
    return "Inconnu"


def extraire_runs_on_off(df_client: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les runs ON/OFF pour un seul client.

    Définition d’un "run"
    ---------------------
    Un run est une séquence consécutive d’états identiques (ON ou OFF).
    Exemple (air_on = 0/1):
        0 0 0 1 1 1 0 0
         run OFF, run ON, run OFF

    Entrée attendue
    ---------------
    df_client :
        - trié par date
        - colonnes : [dataid, local_15min, air_on] avec air_on SANS NaN
        - air_on de type int (0/1)

    Sortie
    ------
    DataFrame runs :
        dataid, run_state, run_start, run_end, n_steps, duration_min, season
    """
    air_on = df_client["air_on"].to_numpy()

    # -------------------------------------------------------------------------
    # Repérer les changements d’état :
    # change[i] = 1 si :
    # - i == 0 (début série)
    # - air_on[i] != air_on[i-1] (transition ON<->OFF)
    # -------------------------------------------------------------------------
    change = np.empty_like(air_on)
    change[0] = 1
    change[1:] = (air_on[1:] != air_on[:-1]).astype(int)

    # run_id = 1,2,3,... identifie chaque run
    run_id = np.cumsum(change)

    df_tmp = df_client.copy()
    df_tmp["run_id"] = run_id

    # -------------------------------------------------------------------------
    # On agrège par run_id et air_on :
    # - run_start = premier timestamp du run
    # - run_end   = dernier timestamp du run
    # - n_steps   = nombre de points (pas 15 min) dans le run
    # -------------------------------------------------------------------------
    runs = (
        df_tmp.groupby(["run_id", "air_on"], as_index=False)
              .agg(
                  run_start=(COL_DATE, "min"),
                  run_end=(COL_DATE, "max"),
                  n_steps=(COL_DATE, "size")
              )
    )

    # Durée (minutes) = nb points * 15 min
    runs["duration_min"] = runs["n_steps"] * PAS_TEMPS_MIN

    # Étiquettes ON/OFF lisibles
    runs["run_state"] = runs["air_on"].map({0: "OFF", 1: "ON"})

    # Saison associée au début du run
    runs["season"] = runs["run_start"].dt.month.map(mois_vers_saison)

    # ID client (constant dans df_client)
    runs[COL_CLIENT] = df_client[COL_CLIENT].iloc[0]

    # Ordre final des colonnes
    return runs[[COL_CLIENT, "run_state", "run_start", "run_end", "n_steps", "duration_min", "season"]]


# =============================================================================
# MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 1) Charger données
    # -------------------------------------------------------------------------
    df = pd.read_csv(CHEMIN_CSV, parse_dates=[COL_DATE])

    # Vérification colonne clim
    if COL_CLIM not in df.columns:
        raise ValueError(f"Colonne '{COL_CLIM}' introuvable dans {CHEMIN_CSV}")

    # -------------------------------------------------------------------------
    # 2) Filtrer clients + période
    # -------------------------------------------------------------------------
    df = df[df[COL_CLIENT].isin(CLIENTS_AUSTIN)].copy()
    df = df[(df[COL_DATE] >= DATE_DEBUT) & (df[COL_DATE] <= DATE_FIN)].copy()

    # Tri pour garantir la cohérence temporelle dans chaque client
    df = df.sort_values([COL_CLIENT, COL_DATE]).reset_index(drop=True)

    print(f"[INFO] Lignes après filtre clients+période: {len(df):,}")

    # -------------------------------------------------------------------------
    # 3) Préparer clim_total (signal analysé)
    # -------------------------------------------------------------------------
    # Conversion numérique robuste + clamp des valeurs négatives
    df["clim_total"] = pd.to_numeric(df[COL_CLIM], errors="coerce")
    df.loc[df["clim_total"] < 0, "clim_total"] = 0

    # IMPORTANT : on remplace les NaN par 0
    # -> on interprète "donnée absente" comme "OFF"
    df["clim_total"] = df["clim_total"].fillna(0)

    # -------------------------------------------------------------------------
    # 4) Binarisation ON/OFF (air_on = état de la clim)
    # -------------------------------------------------------------------------
    # Note : on conserve le nom air_on (hérité de ta version initiale),
    # mais c’est bien l’état ON/OFF de clim_total.
    df["air_on"] = (df["clim_total"] > SEUIL_ON_KW).astype(int)

    # Saison et heure (utiles pour les graphes)
    df["season"] = df[COL_DATE].dt.month.map(mois_vers_saison)
    df["hour"] = df[COL_DATE].dt.hour

    print(f"[INFO] Seuil ON/OFF = {SEUIL_ON_KW:.3f} kW (sur clim_total)")
    print(df["air_on"].value_counts().rename({0: "OFF", 1: "ON"}))

    # -------------------------------------------------------------------------
    # 5) Extraction des runs ON/OFF par client
    # -------------------------------------------------------------------------
    runs_list = []

    for cid, df_c in df.groupby(COL_CLIENT, sort=False):
        # On ne garde que les colonnes utiles pour l’extraction des runs
        df_c = df_c[[COL_CLIENT, COL_DATE, "clim_total", "air_on"]].copy()

        # Sécurité : tri
        df_c = df_c.sort_values(COL_DATE)

        # air_on en int (0/1) garanti
        df_c["air_on"] = df_c["air_on"].astype(int)

        # Extraire runs
        runs_list.append(extraire_runs_on_off(df_c))

    # Concaténation runs de tous les clients
    runs_all = pd.concat(runs_list, ignore_index=True)

    # Saison en catégorie ordonnée (pour beaux tris + graphes)
    runs_all["season"] = pd.Categorical(runs_all["season"], categories=ORDRE_SAISONS, ordered=True)

    print(f"\n[INFO] Nombre total de runs: {len(runs_all):,}")
    print(runs_all.head())

    # -------------------------------------------------------------------------
    # 6) Stats durées ON/OFF par saison
    # -------------------------------------------------------------------------
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

    print("\n=== Statistiques des durées ON/OFF (min) par saison ===")
    print(stats_durations)

    # -------------------------------------------------------------------------
    # 7) Cycles OFF→ON par jour et par client + stats saisonnières
    # -------------------------------------------------------------------------
    runs_all = runs_all.sort_values([COL_CLIENT, "run_start"]).reset_index(drop=True)

    # État précédent (par client)
    runs_all["prev_state"] = runs_all.groupby(COL_CLIENT)["run_state"].shift(1)

    # Un "cycle" est défini ici comme une transition OFF -> ON
    runs_all["is_cycle"] = (
        (runs_all["run_state"] == "ON") &
        (runs_all["prev_state"] == "OFF")
    ).astype(int)

    # Date (jour) associée au démarrage du run ON
    runs_all["date"] = runs_all["run_start"].dt.date

    # Nombre de cycles par (saison, client, jour)
    daily_cycles_client = (
        runs_all[runs_all["is_cycle"] == 1]
        .groupby(["season", COL_CLIENT, "date"], observed=False)
        .size()
        .reset_index(name="n_cycles")
    )

    daily_cycles_client["season"] = pd.Categorical(
        daily_cycles_client["season"], categories=ORDRE_SAISONS, ordered=True
    )

    # Stats du nb de cycles/jour (tous clients & jours)
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

    print("\n=== Statistiques cycles OFF→ON par jour et par client (par saison) ===")
    print(stats_cycles)

    # -------------------------------------------------------------------------
    # 8) Graphe : durée moyenne d'un cycle ON (heures) par mois
    # -------------------------------------------------------------------------
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
    plt.title("Austin 2018 — Durée moyenne d'un cycle ON (clim) par mois")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 9) Graphe : Duty cycle par saison
    # -------------------------------------------------------------------------
    # Duty cycle = fraction de pas 15 min où la clim est ON
    duty_saison = (
        df.groupby("season", observed=False)["air_on"]
          .mean()
          .reindex(ORDRE_SAISONS)
    )

    plt.figure(figsize=(8, 5))
    duty_saison.plot(kind="bar")
    plt.ylabel("Duty cycle moyen (fraction du temps ON)")
    plt.title("Austin 2018 — Duty cycle (clim ON/total) par saison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    print("\n=== Duty cycle par saison ===")
    print(duty_saison)

    # -------------------------------------------------------------------------
    # 10) Graphe : Profil journalier moyen par saison (moyenne tous clients)
    # -------------------------------------------------------------------------
    # On moyenne clim_total pour chaque heure (0..23) et chaque saison
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
    plt.title("Austin 2018 — Profil journalier moyen de la clim par saison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 11) Graphe : Durée ON vs heure de démarrage
    # -------------------------------------------------------------------------
    # On calcule la durée moyenne des runs ON en fonction de l'heure de run_start
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
    plt.title("Austin 2018 — Durée ON vs heure de démarrage")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n=== Durée moyenne ON par heure de démarrage ===")
    print(duree_par_heure)

    # -------------------------------------------------------------------------
    # 12) Sauvegarde résultats 
    # -------------------------------------------------------------------------

    # (A) Table brute des runs ON/OFF
    # Chaque ligne = un run consécutif (ON ou OFF) pour un client.
    # Utile pour :
    # - analyser les durées des cycles
    # - détecter cycles longs / courts
    # - calculer des statistiques par sous-groupe (client, saison, mois, etc.)
    runs_all.to_csv("output/austin_runs_on_off.csv", index=False)

    # (B) Statistiques des durées ON/OFF par saison
    # Résumé global (tous clients confondus) des durées des runs ON et OFF.
    # Colonnes :
    # - n_runs : nb de runs observés
    # - mean/median/min/max : durées en minutes
    stats_durations.to_csv("output/austin_stats_durees_on_off_par_saison.csv", index=False)

    # (C) Cycles OFF→ON par jour et par client
    # Chaque ligne = (saison, client, date) avec :
    # - n_cycles : nb de démarrages OFF→ON observés ce jour-là.
    # Utile pour quantifier la fréquence d'activation au jour le jour.
    daily_cycles_client.to_csv("output/austin_daily_cycles_par_client.csv", index=False)

    # (D) Statistiques des cycles OFF→ON par saison
    # Résumé global des n_cycles par saison :
    # - mean_cycles : nb moyen de démarrages/jour
    # - median/min/max : robustesse et extrêmes
    stats_cycles.to_csv("output/austin_stats_cycles_par_saison.csv", index=False)

    # (E) Durée moyenne des cycles ON par mois
    # Résumé mensuel des runs ON :
    # - n_cycles : nb de cycles ON observés dans le mois
    # - mean_duration_h : durée moyenne ON (heures)
    # - median/min/max : durées en minutes
    stats_cycle_mois.to_csv("output/austin_duree_cycle_on_par_mois.csv", index=False)

    print("\n[OK] Fichiers générés :")
    print(" - output/austin_runs_on_off.csv")
    print(" - output/austin_stats_durees_on_off_par_saison.csv")
    print(" - output/austin_daily_cycles_par_client.csv")
    print(" - output/austin_stats_cycles_par_saison.csv")
    print(" - output/austin_duree_cycle_on_par_mois.csv")


if __name__ == "__main__":
    main()