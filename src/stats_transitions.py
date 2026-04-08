import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_loader import (
    BASE_DIR, SEUIL_ON, PAS_MINUTES, ORDRE_SAISONS, COULEURS_REGIONS,
    _save, get_couleur, charger_sources,
)


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# "region"        → données régionales (processed_energy_data_*.csv)
# "desagregation" → résultats de désagrégation (resultats_desagregation_*.csv)
MODE = "region"

# Laisser vide [] pour charger tous les fichiers de désagrégation disponibles.
FICHIERS_DESAGREGATION = [
    # "resultats_desagregation_1417_2019-08-01_7jours.csv",
]

PALETTE = {
    "on"  : "#2E75B6",
    "off" : "#E06C2E",
    "gris": "#888888",
    "dark": "#1F3864",
}


# ══════════════════════════════════════════════════════════════════════
# EXTRACTION DES RUNS ON / OFF
# ══════════════════════════════════════════════════════════════════════

def extraire_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifie chaque séquence consécutive ON ou OFF.
    Retourne un DataFrame avec une ligne par run :
      etat, debut, fin, duree_pas, duree_min, saison, dataid
    """
    records = []

    for dataid, grp in df.groupby("dataid"):
        grp = grp.sort_index()
        on  = (grp["clim"] > SEUIL_ON).astype(int)

        changement = on.diff().fillna(0).ne(0)
        grp_id     = changement.cumsum()

        for run_id, run in grp.groupby(grp_id):
            etat      = "ON" if run["clim"].mean() > SEUIL_ON else "OFF"
            debut     = run.index[0]
            fin       = run.index[-1]
            duree_pas = len(run)
            duree_min = duree_pas * PAS_MINUTES
            saison    = run["saison"].iloc[0]
            date      = debut.normalize()

            records.append({
                "dataid"   : dataid,
                "etat"     : etat,
                "debut"    : debut,
                "fin"      : fin,
                "duree_pas": duree_pas,
                "duree_min": duree_min,
                "saison"   : saison,
                "date"     : date,
            })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════
# STATISTIQUES DE TRANSITIONS
# ══════════════════════════════════════════════════════════════════════

def _duty_cycle_saison(df: pd.DataFrame, saison: str) -> float:
    """Duty cycle (% temps ON) pour une saison donnée."""
    mask = df["saison"] == saison
    if mask.sum() == 0:
        return float("nan")
    return (df.loc[mask, "clim"] > SEUIL_ON).mean() * 100

def calculer_stats_transitions(
    df: pd.DataFrame,
    runs: pd.DataFrame,
    label: str,
) -> pd.Series:
    runs_on  = runs[runs["etat"] == "ON"]
    runs_off = runs[runs["etat"] == "OFF"]
    n_jours  = df["date"].nunique()
    n_cycles = len(runs_on)

    pct_cycles_courts = (
        (runs_on["duree_min"] < 30).sum() / len(runs_on) * 100
        if len(runs_on) > 0 else float("nan")
    )

    return pd.Series({
        "Transitions totales"           : int(df.groupby("dataid")["clim"]
                                             .apply(lambda x: (x > SEUIL_ON)
                                             .astype(int).diff().abs()
                                             .sum()).sum()),
        "Cycles (démarrages OFF→ON)"    : n_cycles,
        "Cycles / jour (moyenne)"       : round(n_cycles / n_jours, 2) if n_jours > 0 else float("nan"),
        "Durée ON - moyenne (min)"      : runs_on["duree_min"].mean(),
        "Durée ON - médiane (min)"      : runs_on["duree_min"].median(),
        "Durée ON - min observée (min)" : runs_on["duree_min"].min(),
        "Durée ON - max observée (min)" : runs_on["duree_min"].max(),
        "Durée ON - écart-type (min)"   : runs_on["duree_min"].std(),
        "Durée OFF - moyenne (min)"     : runs_off["duree_min"].mean(),
        "Durée OFF - médiane (min)"     : runs_off["duree_min"].median(),
        "Durée OFF - min observée (min)": runs_off["duree_min"].min(),
        "Durée OFF - max observée (min)": runs_off["duree_min"].max(),
        "Durée OFF - écart-type (min)"  : runs_off["duree_min"].std(),
        "% cycles ON < 30 min"          : pct_cycles_courts,
        # Duty cycle
        "Duty cycle global (%)"         : (df["clim"] > SEUIL_ON).mean() * 100,
        "Duty cycle - Hiver (%)"        : _duty_cycle_saison(df, "Hiver"),
        "Duty cycle - Printemps (%)"    : _duty_cycle_saison(df, "Printemps"),
        "Duty cycle - Été (%)"          : _duty_cycle_saison(df, "Été"),
        "Duty cycle - Automne (%)"      : _duty_cycle_saison(df, "Automne"),
    }, name=label).round(3)


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES PAR SOURCE
# ══════════════════════════════════════════════════════════════════════

def graphiques_transitions(
    df: pd.DataFrame,
    runs: pd.DataFrame,
    label: str,
    out_dir: Path,
) -> None:

    runs_on  = runs[runs["etat"] == "ON"]
    runs_off = runs[runs["etat"] == "OFF"]
    couleur  = get_couleur(label)

    # T1 - Histogrammes durées ON et OFF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, etat, col in [
        (axes[0], runs_on["duree_min"],  "ON",  PALETTE["on"]),
        (axes[1], runs_off["duree_min"], "OFF", PALETTE["off"]),
    ]:
        ax.hist(data.clip(upper=data.quantile(0.95)),
                bins=30, color=col, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="black",       ls="--", lw=1.5,
                   label=f"Moyenne = {data.mean():.0f} min")
        ax.axvline(data.median(), color=PALETTE["gris"], ls=":",  lw=1.5,
                   label=f"Médiane = {data.median():.0f} min")
        ax.axvline(PAS_MINUTES,   color="red",         ls="--", lw=1.2,
                   label=f"1 pas = {PAS_MINUTES} min")
        ax.set_xlabel("Durée (min)")
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Distribution des durées {etat} ({label})")
        ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "T1_durees_on_off.png")

    # T2 - Durées ON/OFF par saison (boxplot)
    runs_saison = runs[runs["saison"].isin(ORDRE_SAISONS)]
    fig, axes   = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, etat, col in [
        (axes[0], "ON",  PALETTE["on"]),
        (axes[1], "OFF", PALETTE["off"]),
    ]:
        data_etat = runs_saison[runs_saison["etat"] == etat]
        sns.boxplot(data=data_etat, x="saison", y="duree_min",
                    order=ORDRE_SAISONS, color=col, ax=ax)
        ax.set_xlabel("Saison")
        ax.set_ylabel("Durée (min)")
        ax.set_title(f"Durées {etat} par saison ({label})")
        ax.axhline(PAS_MINUTES, color="red", ls="--", lw=1.2,
                   label=f"1 pas = {PAS_MINUTES} min")
        ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "T2_durees_saison.png")

    # T3 - Cycles par jour par saison
    cycles_jour = (
        runs_on.groupby(["saison", "date"])
        .size()
        .reset_index(name="n_cycles")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=cycles_jour, x="saison", y="n_cycles",
                order=ORDRE_SAISONS, color=couleur, ax=ax)
    ax.set_xlabel("Saison")
    ax.set_ylabel("Nombre de cycles par jour")
    ax.set_title(f"Cycles de climatisation par jour et par saison ({label})")
    plt.tight_layout()
    _save(fig, out_dir, "T3_cycles_par_jour_saison.png")

    # T4 - Transitions par heure de la journée
    on_binary = (df["clim"] > SEUIL_ON).astype(int)
    demarrages = on_binary[(on_binary.diff() == 1)]
    demarrages_par_heure = (demarrages.groupby(demarrages.index.hour)
                             .size()
                             .reindex(range(24), fill_value=0))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(demarrages_par_heure.index, demarrages_par_heure.values,
           color=couleur, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Nombre de démarrages (OFF→ON)")
    ax.set_title(f"Démarrages de climatisation par heure ({label})")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    _save(fig, out_dir, "T4_demarrages_par_heure.png")

    # T5 - Duty cycle journalier par saison (boxplot)
    duty_jour = (
        df.groupby(["saison", "date"])["clim"]
        .apply(lambda x: (x > SEUIL_ON).mean() * 100)
        .reset_index(name="duty_cycle")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=duty_jour, x="saison", y="duty_cycle",
                order=ORDRE_SAISONS, color=couleur, ax=ax)
    ax.set_xlabel("Saison")
    ax.set_ylabel("Duty cycle journalier (%)")
    ax.set_title(f"Duty cycle journalier par saison ({label})")
    ax.axhline(duty_jour["duty_cycle"].mean(), color="red", ls="--", lw=1.2,
               label=f"Moyenne globale = {duty_jour['duty_cycle'].mean():.1f}%")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "T5_duty_cycle_saison.png")

    # T6 - Profil horaire du duty cycle
    duty_heure = (
        df.groupby("hour")["clim"]
        .apply(lambda x: (x > SEUIL_ON).mean() * 100)
        .reindex(range(24), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(duty_heure.index, duty_heure.values,
           color=couleur, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Duty cycle (%)")
    ax.set_title(f"Profil horaire du duty cycle ({label})")
    ax.set_xticks(range(0, 24))
    ax.axhline(duty_heure.mean(), color="red", ls="--", lw=1.2,
               label=f"Moyenne = {duty_heure.mean():.1f}%")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "T6_duty_cycle_horaire.png")


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES COMPARATIFS TOUTES SOURCES
# ══════════════════════════════════════════════════════════════════════

def graphiques_comparatifs_transitions(
    all_runs: dict,
    all_dfs: dict,
    out_dir: Path,
) -> None:

    couleurs = [get_couleur(lbl) for lbl in all_runs]

    # TC1 - Durées ON comparées
    fig, ax = plt.subplots(figsize=(12, 5))
    data_plot = pd.concat([
        r[r["etat"] == "ON"][["duree_min"]].assign(region=lbl)
        for lbl, r in all_runs.items()
    ])
    sns.boxplot(data=data_plot, x="region", y="duree_min",
                order=list(all_runs.keys()),
                palette=couleurs, ax=ax)
    ax.axhline(PAS_MINUTES, color="red", ls="--", lw=1.2,
               label=f"1 pas = {PAS_MINUTES} min")
    ax.set_xlabel("Source")
    ax.set_ylabel("Durée ON (min)")
    ax.set_title("Distribution des durées ON - Comparaison")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "TC1_durees_on_comparatif.png")

    # TC2 - Cycles par jour
    fig, ax = plt.subplots(figsize=(12, 5))
    data_cycles = pd.concat([
        r[r["etat"] == "ON"]
        .groupby("date").size()
        .reset_index(name="n_cycles")
        .assign(region=lbl)
        for lbl, r in all_runs.items()
    ])
    sns.boxplot(data=data_cycles, x="region", y="n_cycles",
                order=list(all_runs.keys()),
                palette=couleurs, ax=ax)
    ax.set_xlabel("Source")
    ax.set_ylabel("Cycles par jour")
    ax.set_title("Cycles de climatisation par jour - Comparaison")
    plt.tight_layout()
    _save(fig, out_dir, "TC2_cycles_jour_comparatif.png")

    # TC3 - Démarrages par heure superposés
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, df in all_dfs.items():
        on_binary  = (df["clim"] > SEUIL_ON).astype(int)
        demarrages = on_binary[(on_binary.diff() == 1)]
        par_heure  = (demarrages.groupby(demarrages.index.hour)
                      .size()
                      .reindex(range(24), fill_value=0))
        ax.plot(par_heure.index, par_heure.values, lw=2,
                color=get_couleur(label), label=label, marker="o", markersize=4)
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Nombre de démarrages (OFF→ON)")
    ax.set_title("Démarrages de climatisation par heure - Comparaison")
    ax.set_xticks(range(0, 24))
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "TC3_demarrages_heure_comparatif.png")


# ══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("ANALYSE DES TRANSITIONS ON/OFF - THERMO NILM")
    print(f"Mode : {MODE}")
    print("=" * 60)

    all_dfs   = charger_sources(MODE, FICHIERS_DESAGREGATION)
    all_stats = {}
    all_runs  = {}

    for label, df in all_dfs.items():
        print(f"\n── {label} ──")

        print(f"  Extraction des runs ON/OFF...")
        runs = extraire_runs(df)
        all_runs[label] = runs
        print(f"    Runs ON  : {len(runs[runs['etat']=='ON']):,}")
        print(f"    Runs OFF : {len(runs[runs['etat']=='OFF']):,}")

        stats = calculer_stats_transitions(df, runs, label)
        all_stats[label] = stats
        print(f"\n  Statistiques de transitions :")
        print(stats.to_string())

        slug    = label.replace(" ", "_").replace("/", "-")
        out_dir = BASE_DIR / "output" / "transitions" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques → {out_dir.name}/")
        graphiques_transitions(df, runs, label, out_dir)

    if all_stats:
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF")
        print("=" * 60)
        comparatif = pd.DataFrame(all_stats)
        print(comparatif.to_string())

        out_csv = BASE_DIR / "output" / "transitions" / "transitions_comparatif.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparatif.to_csv(out_csv)
        print(f"\n  Tableau sauvegardé : {out_csv.name}")

    if len(all_dfs) > 1:
        out_comp = BASE_DIR / "output" / "transitions" / "comparatif"
        out_comp.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques comparatifs → comparatif/")
        graphiques_comparatifs_transitions(all_runs, all_dfs, out_comp)

    print("\nAnalyse des transitions terminée.")


if __name__ == "__main__":
    main()
