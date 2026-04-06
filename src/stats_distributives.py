import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, jarque_bera, probplot
from pathlib import Path

from data_loader import (
    BASE_DIR, SEUIL_ON, ORDRE_SAISONS,
    _save, get_couleur, charger_sources,
)


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# "region"        → données régionales (processed_energy_data_*.csv)
# "desagregation" → résultats de désagrégation (resultats_desagregation_*.csv)
MODE = "desagregation"

# En mode "desagregation" : liste de fichiers à analyser (dans data/).
# Laisser vide [] pour charger tous les fichiers de désagrégation disponibles.
FICHIERS_DESAGREGATION = [
    # "resultats_desagregation_1417_2019-08-01_7jours.csv",
]


# ══════════════════════════════════════════════════════════════════════
# ANALYSE DISTRIBUTIVE D'UNE SOURCE
# ══════════════════════════════════════════════════════════════════════

def analyser_distribution(df: pd.DataFrame, label: str, out_dir: Path) -> pd.Series:
    """Calcule les statistiques distributives et génère les graphiques."""

    couleur = get_couleur(label)
    data    = df["clim"].dropna()

    print(f"\n  Nombre de points : {len(data):,}")

    skewness_val         = skew(data)
    kurtosis_val         = kurtosis(data)
    jb_stat, jb_pvalue   = jarque_bera(data)

    print(f"  Skewness    : {skewness_val:.4f}")
    print(f"  Kurtosis    : {kurtosis_val:.4f}")
    print(f"  Jarque-Bera : stat={jb_stat:.4f}, p-value={jb_pvalue:.6f}")

    if skewness_val > 0:
        interp_skew = "asymétrique à droite"
    elif skewness_val < 0:
        interp_skew = "asymétrique à gauche"
    else:
        interp_skew = "symétrique"
    interp_kurt   = "queues épaisses (valeurs extrêmes)" if kurtosis_val > 0 else "distribution plus uniforme"
    interp_normal = "non normale (p < 0.05)" if jb_pvalue < 0.05 else "compatible avec une loi normale"
    print(f"  → Distribution {interp_skew}, {interp_kurt}, {interp_normal}.")

    # ── D1 : Histogramme + KDE ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data, bins=50, kde=True, color=couleur, ax=ax)
    ax.set_title(f"Distribution complète de la clim - avec KDE ({label})")
    ax.set_xlabel("clim (kW)")
    ax.set_ylabel("Fréquence")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, out_dir, "D1_histogramme_kde.png")

    # ── D2 : QQ-plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    probplot(data, plot=ax)
    ax.set_title(f"QQ-plot - clim ({label})")
    plt.tight_layout()
    _save(fig, out_dir, "D2_qqplot.png")

    return pd.Series({
        "Skewness"          : round(skewness_val, 4),
        "Kurtosis"          : round(kurtosis_val,  4),
        "Jarque-Bera stat"  : round(jb_stat,        4),
        "Jarque-Bera p-val" : round(jb_pvalue,       6),
        "N points"          : len(data),
        "Normale ?"         : "Non" if jb_pvalue < 0.05 else "Oui",
    }, name=label)


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES COMPARATIFS TOUTES SOURCES
# ══════════════════════════════════════════════════════════════════════

def graphiques_comparatifs_distributifs(dfs: dict, out_dir: Path) -> None:

    # DC1 : KDE superposées
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in dfs.items():
        sns.kdeplot(df["clim"].dropna(), ax=ax,
                    color=get_couleur(label), label=label, lw=2)
    ax.set_xlabel("clim (kW)")
    ax.set_ylabel("Densité")
    ax.set_title("Distributions comparées - Climatisation")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, out_dir, "DC1_kde_comparatif.png")

    # DC2 : Skewness et Kurtosis comparés
    stats_list = []
    for label, df in dfs.items():
        d = df["clim"].dropna()
        stats_list.append({
            "region"  : label,
            "Skewness": skew(d),
            "Kurtosis": kurtosis(d),
        })
    df_stats = pd.DataFrame(stats_list).set_index("region")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, ["Skewness", "Kurtosis"]):
        bars = ax.bar(df_stats.index, df_stats[col],
                      color=[get_couleur(lbl) for lbl in df_stats.index],
                      edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"{col} par source")
        ax.set_ylabel(col)
        ax.set_xticklabels(df_stats.index, rotation=15, ha="right")
        for bar, val in zip(bars, df_stats[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * np.sign(val + 1e-9),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "DC2_skewness_kurtosis_comparatif.png")


# ══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("ANALYSE DISTRIBUTIVE - THERMO NILM")
    print(f"Mode : {MODE}")
    print("=" * 60)

    all_dfs   = charger_sources(MODE, FICHIERS_DESAGREGATION)
    all_stats = {}

    for label, df in all_dfs.items():
        print(f"\n── {label} ──")

        slug    = label.replace(" ", "_").replace("/", "-")
        out_dir = BASE_DIR / "output" / "distributives" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Graphiques → {out_dir.name}/")

        stats = analyser_distribution(df, label, out_dir)
        all_stats[label] = stats

    if all_stats:
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF")
        print("=" * 60)
        comparatif = pd.DataFrame(all_stats)
        print(comparatif.to_string())

        out_csv = BASE_DIR / "output" / "distributives" / "distributives_comparatif.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparatif.to_csv(out_csv)
        print(f"\n  Tableau sauvegardé : {out_csv.name}")

    if len(all_dfs) > 1:
        out_comp = BASE_DIR / "output" / "distributives" / "comparatif"
        out_comp.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques comparatifs → comparatif/")
        graphiques_comparatifs_distributifs(all_dfs, out_comp)

    print("\nAnalyse distributive terminée.")


if __name__ == "__main__":
    main()
