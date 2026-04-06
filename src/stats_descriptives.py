import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_loader import (
    BASE_DIR, SEUIL_ON, ORDRE_SAISONS, COULEURS_REGIONS,
    _save, get_couleur, charger_sources,
)


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# "region"       → données régionales (processed_energy_data_*.csv)
# "desagregation" → résultats de désagrégation (resultats_desagregation_*.csv)
MODE = "desagregation"

# En mode "desagregation" : liste de fichiers à analyser (dans data/).
# Laisser vide [] pour charger tous les fichiers de désagrégation disponibles.
FICHIERS_DESAGREGATION = [
    # "resultats_desagregation_1417_2019-08-01_7jours.csv",
]

BINS_PUISSANCE = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
LABELS_BINS    = ["0–0.5", "0.5–1.0", "1.0–1.5",
                  "1.5–2.0", "2.0–2.5", "2.5–3.0"]

PALETTE = {
    "principal" : "#2E75B6",
    "secondaire": "#1F3864",
    "accent1"   : "red",
    "accent2"   : "orange",
    "gris"      : "#888888",
}

PALETTE_SAISONS = ["#D6E4F0", "#2E75B6", "#1F3864", "#0A1628"]


# ══════════════════════════════════════════════════════════════════════
# STATISTIQUES DESCRIPTIVES D'UNE RÉGION / SOURCE
# ══════════════════════════════════════════════════════════════════════

def calculer_stats(df: pd.DataFrame, label: str) -> pd.Series:
    P     = df["clim"]
    P_on  = P[P > SEUIL_ON]
    mode_on = P_on.mode()

    return pd.Series({
        "Moyenne (kW)"               : P.mean(),
        "Médiane (kW)"               : P.median(),
        "Mode - états ON (kW)"       : mode_on.iloc[0] if not mode_on.empty else float("nan"),
        "Écart-type (kW)"            : P.std(),
        "Variance (kW²)"             : P.var(),
        "Min (kW)"                   : P.min(),
        "Max (kW)"                   : P.max(),
        "Coeff. de variation (%)"    : (P.std() / P.mean()) * 100,
        "P5  (kW)"                   : P.quantile(0.05),
        "P10 (kW)"                   : P.quantile(0.10),
        "Q1  (kW)"                   : P.quantile(0.25),
        "Q2  (kW)"                   : P.quantile(0.50),
        "Q3  (kW)"                   : P.quantile(0.75),
        "P90 (kW)"                   : P.quantile(0.90),
        "P95 (kW)"                   : P.quantile(0.95),
        "IQR (kW)"                   : P.quantile(0.75) - P.quantile(0.25),
        "% temps ON"                 : (P > SEUIL_ON).mean() * 100,
    }, name=label).round(4)


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES PAR SOURCE
# ══════════════════════════════════════════════════════════════════════

def graphiques_region(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    P       = df["clim"]
    P_on    = P[P > SEUIL_ON]
    couleur = get_couleur(label)

    # G1 - Histogramme
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(P_on, bins=40, color=couleur, edgecolor="white", alpha=0.85)
    ax.axvline(P.mean(),   color="red",    ls="--", lw=1.8,
               label=f"Moyenne = {P.mean():.3f} kW")
    ax.axvline(P.median(), color="orange", ls="--", lw=1.8,
               label=f"Médiane = {P.median():.3f} kW")
    ax.set_xlabel("Puissance (kW)")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution des puissances - États ON uniquement ({label})")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "G1_histogramme.png")

    # G2 - Fréquences par intervalles
    df = df.copy()
    df["intervalle"] = pd.cut(P, bins=BINS_PUISSANCE,
                               labels=LABELS_BINS, right=True)
    freq_rel = (df["intervalle"].value_counts(normalize=True)
                .sort_index() * 100)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(freq_rel.index, freq_rel.values,
                  color=couleur, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, freq_rel.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Intervalle de puissance (kW)")
    ax.set_ylabel("Fréquence relative (%)")
    ax.set_title(f"Distribution des fréquences par intervalles ({label})")
    plt.tight_layout()
    _save(fig, out_dir, "G2_freq_intervalles.png")

    # G3 - Boxplot par saison
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="saison", y="clim",
                order=ORDRE_SAISONS, palette=PALETTE_SAISONS, ax=ax)
    ax.set_xlabel("Saison")
    ax.set_ylabel("Puissance climatisation (kW)")
    ax.set_title(f"Distribution par saison ({label})")
    plt.tight_layout()
    _save(fig, out_dir, "G3_boxplot_saison.png")

    # G4 - Profil horaire moyen
    profil = df.groupby("hour")["clim"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(profil.index, profil["mean"], color=couleur, lw=2, label="Moyenne")
    ax.fill_between(profil.index,
                    (profil["mean"] - profil["std"]).clip(lower=0),
                    profil["mean"] + profil["std"],
                    alpha=0.2, color=couleur, label="±1 écart-type")
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Puissance (kW)")
    ax.set_title(f"Profil horaire moyen - Climatisation ({label})")
    ax.set_xticks(range(0, 24))
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "G4_profil_horaire.png")

    # G5 - Heatmap heure × mois
    pivot = df.pivot_table(values="clim", index="hour",
                           columns="month", aggfunc="mean")
    noms_mois = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Jun",
                 7:"Jul", 8:"Aoû", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"}
    pivot.columns = [noms_mois.get(m, m) for m in pivot.columns]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3,
                linecolor="#eeeeee",
                cbar_kws={"label": "Puissance moyenne (kW)"})
    ax.set_title(f"Consommation moyenne - Heure × Mois ({label})")
    ax.set_ylabel("Heure")
    ax.set_xlabel("Mois")
    plt.tight_layout()
    _save(fig, out_dir, "G5_heatmap_heure_mois.png")


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES COMPARATIFS TOUTES SOURCES
# ══════════════════════════════════════════════════════════════════════

def graphiques_comparatifs(dfs: dict, out_dir: Path) -> None:

    couleurs = [get_couleur(lbl) for lbl in dfs]

    # GC1 - Profils horaires superposés
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, df in dfs.items():
        profil = df.groupby("hour")["clim"].mean()
        ax.plot(profil.index, profil.values, lw=2,
                color=get_couleur(label), label=label)
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Puissance moyenne (kW)")
    ax.set_title("Profils horaires moyens - Comparaison")
    ax.set_xticks(range(0, 24))
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "GC1_profils_horaires_comparatif.png")

    # GC2 - Boxplots côte à côte
    df_all = pd.concat(dfs.values())
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df_all, x="region", y="clim",
                order=list(dfs.keys()),
                palette=couleurs, ax=ax)
    ax.set_xlabel("Source")
    ax.set_ylabel("Puissance climatisation (kW)")
    ax.set_title("Distribution de la climatisation par source")
    plt.tight_layout()
    _save(fig, out_dir, "GC2_boxplot_regions.png")

    # GC3 - % temps ON par source et saison
    pct_on = (df_all.groupby(["region", "saison"])["clim"]
              .apply(lambda x: (x > SEUIL_ON).mean() * 100)
              .unstack("saison")
              .reindex(columns=ORDRE_SAISONS))

    fig, ax = plt.subplots(figsize=(11, 5))
    pct_on.plot(kind="bar", ax=ax,
                color=["#D6E4F0", "#2E75B6", "#1F3864", "#0A1628"],
                edgecolor="white")
    ax.set_xlabel("Source")
    ax.set_ylabel("% temps ON")
    ax.set_title("Pourcentage de temps ON - Par source et saison")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(title="Saison")
    plt.tight_layout()
    _save(fig, out_dir, "GC3_pct_on_region_saison.png")


# ══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("ANALYSE DESCRIPTIVE - THERMO NILM")
    print(f"Mode : {MODE}")
    print("=" * 60)

    all_dfs   = charger_sources(MODE, FICHIERS_DESAGREGATION)
    all_stats = {}

    for label, df in all_dfs.items():
        print(f"\n── {label} ──")

        stats = calculer_stats(df, label)
        all_stats[label] = stats

        # Distribution des fréquences
        df_tmp = df.copy()
        df_tmp["intervalle"] = pd.cut(df_tmp["clim"], bins=BINS_PUISSANCE,
                                       labels=LABELS_BINS, right=True)
        freq_abs = df_tmp["intervalle"].value_counts().sort_index()
        freq_rel = (freq_abs / freq_abs.sum() * 100).round(2)
        print(f"\n  Distribution des fréquences :")
        print(pd.DataFrame({
            "Fréquence absolue"    : freq_abs,
            "Fréquence relative (%)": freq_rel,
        }).to_string())

        # Graphiques individuels
        slug    = label.replace(" ", "_").replace("/", "-")
        out_dir = BASE_DIR / "output" / "stats" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques → {out_dir.name}/")
        graphiques_region(df, label, out_dir)

    if all_stats:
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF")
        print("=" * 60)
        comparatif = pd.DataFrame(all_stats)
        print(comparatif.to_string())

        out_csv = BASE_DIR / "output" / "stats" / "stats_comparatif.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparatif.to_csv(out_csv)
        print(f"\n  Tableau sauvegardé : {out_csv.name}")

    if len(all_dfs) > 1:
        out_comp = BASE_DIR / "output" / "stats" / "comparatif"
        out_comp.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques comparatifs → comparatif/")
        graphiques_comparatifs(all_dfs, out_comp)

    print("\nAnalyse terminée.")


if __name__ == "__main__":
    main()
