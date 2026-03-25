import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, Sequence
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════

def load_results_csv(
    csv_path: str,
    encoding: str = "utf-8",
    sep: Optional[str] = None,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        encoding=encoding,
        sep=sep or None,
        engine="python",
        usecols=usecols,
    )


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")


def _save(fig: plt.Figure, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}")


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent

REGIONS = {
    "Austin"    : "processed_energy_data_austin.csv",
    "California": "processed_energy_data_california.csv",
    "New York"  : "processed_energy_data_newyork.csv",
}

DATAID       = None    # None = tous les clients, int = un seul client
SEUIL_ON     = 0.05   # kW - seuil de détection état ON
PAS_MINUTES  = 15     # résolution temporelle des données

ORDRE_SAISONS = ["Hiver", "Printemps", "Été", "Automne"]
MAP_SAISON = {
    12: "Hiver",     1: "Hiver",     2: "Hiver",
     3: "Printemps", 4: "Printemps", 5: "Printemps",
     6: "Été",       7: "Été",       8: "Été",
     9: "Automne",  10: "Automne",  11: "Automne",
}

COULEURS_REGIONS = {
    "Austin"    : "#2E75B6",
    "California": "#E06C2E",
    "New York"  : "#2E8B57",
}

PALETTE = {
    "on"    : "#2E75B6",
    "off"   : "#E06C2E",
    "gris"  : "#888888",
    "dark"  : "#1F3864",
}


# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT D'UNE RÉGION
# ══════════════════════════════════════════════════════════════════════

def charger_region(nom_region: str, nom_fichier: str) -> pd.DataFrame:
    csv_path = BASE_DIR / "data" / nom_fichier
    if not csv_path.exists():
        print(f"   Fichier introuvable : {csv_path} - région ignorée.")
        return pd.DataFrame()

    cols = ["dataid", "year", "month", "day", "hour", "minute",
            "temp", "grid", "clim"]

    df = load_results_csv(str(csv_path), usecols=cols)

    try:
        _require_cols(df, cols)
    except ValueError as e:
        print(f"   {nom_region} : {e} - région ignorée.")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime({
        "year": df["year"], "month": df["month"], "day": df["day"],
        "hour": df["hour"], "minute": df["minute"],
    }, errors="coerce")

    n_nat = df["datetime"].isna().sum()
    if n_nat > 0:
        print(f"   {nom_region} : {n_nat} horodatages invalides → exclus")
        df = df.dropna(subset=["datetime"])

    df["clim"] = pd.to_numeric(df["clim"], errors="coerce").fillna(0)
    df["grid"]  = pd.to_numeric(df["grid"],  errors="coerce")
    df = df.dropna(subset=["grid"])

    if DATAID is not None:
        df = df[df["dataid"] == DATAID]

    df = df.set_index("datetime").sort_index()
    df["saison"] = df["month"].map(MAP_SAISON)
    df["region"] = nom_region
    df["date"]   = df.index.normalize()

    print(f"  {nom_region} : {len(df):,} observations chargées")
    return df


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

    # Traiter client par client pour éviter les faux runs inter-clients
    for dataid, grp in df.groupby("dataid"):
        grp = grp.sort_index()
        on  = (grp["clim"] > SEUIL_ON).astype(int)

        # Détecter les changements d'état
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

def calculer_stats_transitions(
    df: pd.DataFrame,
    runs: pd.DataFrame,
    region: str,
) -> pd.Series:
    """Retourne un pd.Series de statistiques de transitions pour une région."""

    runs_on  = runs[runs["etat"] == "ON"]
    runs_off = runs[runs["etat"] == "OFF"]

    # Nombre de jours uniques
    n_jours = df["date"].nunique()

    # Cycles = nombre de démarrages OFF→ON = nombre de runs ON
    n_cycles = len(runs_on)

    # % cycles ON dont durée < 1 pas (15 min) - impossible ici car min = 1 pas
    # On cherche ceux < 2 pas (< 30 min) pour identifier les cycles très courts
    pct_cycles_courts = (
        (runs_on["duree_min"] < 30).sum() / len(runs_on) * 100
        if len(runs_on) > 0 else float("nan")
    )

    return pd.Series({
        # Transitions globales
        "Transitions totales"          : int(df.groupby("dataid")["clim"]
                                            .apply(lambda x: (x > SEUIL_ON)
                                            .astype(int).diff().abs()
                                            .sum()).sum()),
        "Cycles (démarrages OFF→ON)"   : n_cycles,
        "Cycles / jour (moyenne)"      : round(n_cycles / n_jours, 2) if n_jours > 0 else float("nan"),
        # Durées ON
        "Durée ON - moyenne (min)"     : runs_on["duree_min"].mean(),
        "Durée ON - médiane (min)"     : runs_on["duree_min"].median(),
        "Durée ON - min observée (min)": runs_on["duree_min"].min(),
        "Durée ON - max observée (min)": runs_on["duree_min"].max(),
        "Durée ON - écart-type (min)"  : runs_on["duree_min"].std(),
        # Durées OFF
        "Durée OFF - moyenne (min)"    : runs_off["duree_min"].mean(),
        "Durée OFF - médiane (min)"    : runs_off["duree_min"].median(),
        "Durée OFF - min observée (min)": runs_off["duree_min"].min(),
        "Durée OFF - max observée (min)": runs_off["duree_min"].max(),
        "Durée OFF - écart-type (min)" : runs_off["duree_min"].std(),
        # Duty cycle
        "Duty cycle global (%)"        : (df["clim"] > SEUIL_ON).mean() * 100,
        "% cycles ON < 30 min"         : pct_cycles_courts,
    }, name=region).round(3)


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES PAR RÉGION
# ══════════════════════════════════════════════════════════════════════

def graphiques_transitions(
    df: pd.DataFrame,
    runs: pd.DataFrame,
    region: str,
    out_dir: Path,
) -> None:

    runs_on  = runs[runs["etat"] == "ON"]
    runs_off = runs[runs["etat"] == "OFF"]
    couleur  = COULEURS_REGIONS.get(region, "#2E75B6")

    # ── T1 : Histogrammes durées ON et OFF ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, etat, col in [
        (axes[0], runs_on["duree_min"],  "ON",  PALETTE["on"]),
        (axes[1], runs_off["duree_min"], "OFF", PALETTE["off"]),
    ]:
        ax.hist(data.clip(upper=data.quantile(0.95)),
                bins=30, color=col, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="black",  ls="--", lw=1.5,
                   label=f"Moyenne = {data.mean():.0f} min")
        ax.axvline(data.median(), color=PALETTE["gris"], ls=":",  lw=1.5,
                   label=f"Médiane = {data.median():.0f} min")
        ax.axvline(PAS_MINUTES,   color="red",    ls="--", lw=1.2,
                   label=f"1 pas = {PAS_MINUTES} min")
        ax.set_xlabel("Durée (min)")
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Distribution des durées {etat} ({region})")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "T1_durees_on_off.png")

    # ── T2 : Durées ON/OFF par saison (boxplot) ───────────────────
    runs_saison = runs[runs["saison"].isin(ORDRE_SAISONS)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, etat, col in [
        (axes[0], "ON",  PALETTE["on"]),
        (axes[1], "OFF", PALETTE["off"]),
    ]:
        data_etat = runs_saison[runs_saison["etat"] == etat]
        sns.boxplot(data=data_etat, x="saison", y="duree_min",
                    order=ORDRE_SAISONS,
                    color=col, ax=ax)
        ax.set_xlabel("Saison")
        ax.set_ylabel("Durée (min)")
        ax.set_title(f"Durées {etat} par saison ({region})")
        ax.axhline(PAS_MINUTES, color="red", ls="--", lw=1.2,
                   label=f"1 pas = {PAS_MINUTES} min")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, out_dir, "T2_durees_saison.png")

    # ── T3 : Cycles par jour par saison ──────────────────────────
    cycles_jour = (
        runs_on.groupby(["saison", "date"])
        .size()
        .reset_index(name="n_cycles")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=cycles_jour, x="saison", y="n_cycles",
                order=ORDRE_SAISONS,
                color=couleur, ax=ax)
    ax.set_xlabel("Saison")
    ax.set_ylabel("Nombre de cycles par jour")
    ax.set_title(f"Cycles de climatisation par jour et par saison ({region})")
    plt.tight_layout()
    _save(fig, out_dir, "T3_cycles_par_jour_saison.png")

    # ── T4 : Transitions par heure de la journée ─────────────────
    # Compter les démarrages OFF→ON par heure
    on_binary = (df["clim"] > SEUIL_ON).astype(int)
    demarrages = on_binary[(on_binary.diff() == 1)]
    demarrages_par_heure = demarrages.groupby(demarrages.index.hour).size()
    demarrages_par_heure = demarrages_par_heure.reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(demarrages_par_heure.index, demarrages_par_heure.values,
           color=couleur, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Nombre de démarrages (OFF→ON)")
    ax.set_title(f"Démarrages de climatisation par heure ({region})")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    _save(fig, out_dir, "T4_demarrages_par_heure.png")

    # ── T5 : Série temporelle d'une semaine type ──────────────────
    # Choisir la semaine avec le plus de cycles ON (été)
    ete = df[df["saison"] == "Été"]
    if ete.empty:
        ete = df  # fallback si pas d'été dans les données

    cycles_par_semaine = (
        runs_on[runs_on["saison"] == "Été"]
        .groupby(runs_on["debut"].dt.isocalendar().week)
        .size()
    ) if not runs_on[runs_on["saison"] == "Été"].empty else pd.Series()

    if not cycles_par_semaine.empty:
        semaine_cible = cycles_par_semaine.idxmax()
        mask = ete.index.isocalendar().week == semaine_cible
        semaine_df = ete[mask].iloc[:672]   # max 7 jours × 96 pas
    else:
        semaine_df = df.iloc[:672]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Sous-graphique 1 : puissance + état ON/OFF coloré
    ax1 = axes[0]
    ax1.plot(semaine_df.index, semaine_df["clim"],
             color=PALETTE["dark"], lw=1, alpha=0.8, label="Puissance (kW)")
    ax1.axhline(SEUIL_ON, color="red", ls=":", lw=1,
                label=f"Seuil ON = {SEUIL_ON} kW")
    # Zones ON en bleu clair
    on_mask = semaine_df["clim"] > SEUIL_ON
    ax1.fill_between(semaine_df.index, 0, semaine_df["clim"],
                     where=on_mask, alpha=0.25,
                     color=PALETTE["on"], label="État ON")
    ax1.set_ylabel("Puissance (kW)")
    ax1.set_title(f"Série temporelle - Semaine type été ({region})")
    ax1.legend(fontsize=9, loc="upper right")

    # Sous-graphique 2 : état binaire ON/OFF
    ax2 = axes[1]
    ax2.fill_between(semaine_df.index,
                     (semaine_df["clim"] > SEUIL_ON).astype(int),
                     step="post",
                     color=PALETTE["on"], alpha=0.7, label="ON")
    ax2.fill_between(semaine_df.index,
                     (semaine_df["clim"] <= SEUIL_ON).astype(int),
                     step="post",
                     color=PALETTE["off"], alpha=0.3, label="OFF")
    ax2.set_ylabel("État (1=ON, 0=OFF)")
    ax2.set_xlabel("Date / Heure")
    ax2.set_yticks([0, 1])
    patch_on  = mpatches.Patch(color=PALETTE["on"],  alpha=0.7, label="ON")
    patch_off = mpatches.Patch(color=PALETTE["off"], alpha=0.3, label="OFF")
    ax2.legend(handles=[patch_on, patch_off], fontsize=9, loc="upper right")

    plt.tight_layout()
    _save(fig, out_dir, "T5_serie_temporelle_semaine.png")


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES COMPARATIFS TOUTES RÉGIONS
# ══════════════════════════════════════════════════════════════════════

def graphiques_comparatifs_transitions(
    all_runs: dict,
    all_dfs: dict,
    out_dir: Path,
) -> None:

    # TC1 - Durées ON comparées par région
    fig, ax = plt.subplots(figsize=(12, 5))
    data_plot = pd.concat([
        r[r["etat"] == "ON"][["duree_min"]].assign(region=reg)
        for reg, r in all_runs.items()
    ])
    sns.boxplot(data=data_plot, x="region", y="duree_min",
                order=list(all_runs.keys()),
                palette=list(COULEURS_REGIONS.values()),
                ax=ax)
    ax.axhline(PAS_MINUTES, color="red", ls="--", lw=1.2,
               label=f"1 pas = {PAS_MINUTES} min")
    ax.set_xlabel("Région")
    ax.set_ylabel("Durée ON (min)")
    ax.set_title("Distribution des durées ON - Comparaison régions")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "TC1_durees_on_comparatif.png")

    # TC2 - Cycles par jour par région
    fig, ax = plt.subplots(figsize=(12, 5))
    data_cycles = pd.concat([
        r[r["etat"] == "ON"]
        .groupby("date").size()
        .reset_index(name="n_cycles")
        .assign(region=reg)
        for reg, r in all_runs.items()
    ])
    sns.boxplot(data=data_cycles, x="region", y="n_cycles",
                order=list(all_runs.keys()),
                palette=list(COULEURS_REGIONS.values()),
                ax=ax)
    ax.set_xlabel("Région")
    ax.set_ylabel("Cycles par jour")
    ax.set_title("Cycles de climatisation par jour - Comparaison régions")
    plt.tight_layout()
    _save(fig, out_dir, "TC2_cycles_jour_comparatif.png")

    # TC3 - Démarrages par heure superposés
    fig, ax = plt.subplots(figsize=(12, 5))
    for region, df in all_dfs.items():
        on_binary  = (df["clim"] > SEUIL_ON).astype(int)
        demarrages = on_binary[(on_binary.diff() == 1)]
        par_heure  = (demarrages.groupby(demarrages.index.hour)
                      .size()
                      .reindex(range(24), fill_value=0))
        ax.plot(par_heure.index, par_heure.values, lw=2,
                color=COULEURS_REGIONS[region], label=region, marker="o",
                markersize=4)
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Nombre de démarrages (OFF→ON)")
    ax.set_title("Démarrages de climatisation par heure - Comparaison régions")
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
    print("=" * 60)

    all_stats = {}
    all_runs  = {}
    all_dfs   = {}

    for region, fichier in REGIONS.items():
        print(f"\n── {region} ──")

        df = charger_region(region, fichier)
        if df.empty:
            continue
        all_dfs[region] = df

        # Extraction des runs
        print(f"  Extraction des runs ON/OFF...")
        runs = extraire_runs(df)
        all_runs[region] = runs
        print(f"    Runs ON  : {len(runs[runs['etat']=='ON']):,}")
        print(f"    Runs OFF : {len(runs[runs['etat']=='OFF']):,}")

        # Statistiques
        stats = calculer_stats_transitions(df, runs, region)
        all_stats[region] = stats
        print(f"\n  Statistiques de transitions :")
        print(stats.to_string())

        # Graphiques individuels
        out_region = BASE_DIR / "output" / "transitions" / region.replace(" ", "_")
        out_region.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques → {out_region.name}/")
        graphiques_transitions(df, runs, region, out_region)

    # Tableau comparatif
    if all_stats:
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF - TOUTES RÉGIONS")
        print("=" * 60)
        comparatif = pd.DataFrame(all_stats)
        print(comparatif.to_string())

        out_csv = BASE_DIR / "output" / "transitions" / "transitions_comparatif.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparatif.to_csv(out_csv)
        print(f"\n  Tableau sauvegardé : {out_csv.name}")

    # Graphiques comparatifs
    if len(all_dfs) > 1:
        out_comp = BASE_DIR / "output" / "transitions" / "comparatif"
        out_comp.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques comparatifs → comparatif/")
        graphiques_comparatifs_transitions(all_runs, all_dfs, out_comp)

    print("\nAnalyse des transitions terminée.")


if __name__ == "__main__":
    main()