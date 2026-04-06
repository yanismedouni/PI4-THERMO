import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, jarque_bera
from pathlib import Path
from typing import Optional, Sequence


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

DATAID   = None   # None = tous les clients, int = un seul client
SEUIL_ON = 0.05   # kW - seuil de détection état ON

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


# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT D'UNE RÉGION
# ══════════════════════════════════════════════════════════════════════

def charger_region(nom_region: str, nom_fichier: str) -> pd.DataFrame:
    csv_path = BASE_DIR / "data" / nom_fichier
    if not csv_path.exists():
        print(f"    Fichier introuvable : {csv_path} - région ignorée.")
        return pd.DataFrame()

    cols = ["dataid", "year", "month", "day", "hour", "minute",
            "temp", "grid", "clim", "chauffage"]

    df = load_results_csv(str(csv_path), usecols=cols)

    try:
        _require_cols(df, cols)
    except ValueError as e:
        print(f"    {nom_region} : {e} - région ignorée.")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime({
        "year": df["year"], "month": df["month"], "day": df["day"],
        "hour": df["hour"], "minute": df["minute"],
    }, errors="coerce")

    n_nat = df["datetime"].isna().sum()
    if n_nat > 0:
        print(f"   {nom_region} : {n_nat} horodatages invalides → exclus")
        df = df.dropna(subset=["datetime"])

    for col in ["grid", "clim", "chauffage", "temp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["clim"]      = df["clim"].fillna(0)
    df["chauffage"] = df["chauffage"].fillna(0)
    df = df.dropna(subset=["grid"])

    if DATAID is not None:
        df = df[df["dataid"] == DATAID]

    df = df.set_index("datetime").sort_index()
    df["saison"] = df["month"].map(MAP_SAISON)
    df["region"] = nom_region

    print(f"  {nom_region} : {len(df):,} observations chargées")
    return df


# ══════════════════════════════════════════════════════════════════════
# ANALYSE DISTRIBUTIVE D'UNE RÉGION
# ══════════════════════════════════════════════════════════════════════

def analyser_distribution(df: pd.DataFrame, region: str, out_dir: Path) -> pd.Series:
    """Calcule les statistiques distributives et génère les graphiques pour une région."""

    couleur = COULEURS_REGIONS.get(region, "#2E75B6")
    data    = df["clim"].dropna()

    print(f"\n  Nombre de points : {len(data):,}")

    # Calcul des indicateurs
    skewness_val = skew(data)
    kurtosis_val = kurtosis(data)
    jb_stat, jb_pvalue = jarque_bera(data)

    print(f"  Skewness  : {skewness_val:.4f}")
    print(f"  Kurtosis  : {kurtosis_val:.4f}")
    print(f"  Jarque-Bera : stat={jb_stat:.4f}, p-value={jb_pvalue:.6f}")

    # Interprétation
    if skewness_val > 0:
        interp_skew = "asymétrique à droite"
    elif skewness_val < 0:
        interp_skew = "asymétrique à gauche"
    else:
        interp_skew = "symétrique"

    interp_kurt  = "queues épaisses (valeurs extrêmes)" if kurtosis_val > 0 else "distribution plus uniforme"
    interp_normal = "non normale (p < 0.05)" if jb_pvalue < 0.05 else "compatible avec une loi normale"

    print(f"  → Distribution {interp_skew}, {interp_kurt}, {interp_normal}.")

    # ── D1 : Histogramme + KDE ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data, bins=50, kde=True, color=couleur, ax=ax)
    ax.set_title(f"Distribution de la clim ({region})")
    ax.set_xlabel("clim (kW)")
    ax.set_ylabel("Fréquence")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, out_dir, "D1_histogramme_kde.png")

    # ── D2 : QQ-plot (normalité) ──────────────────────────────────────
    from scipy.stats import probplot
    fig, ax = plt.subplots(figsize=(6, 6))
    probplot(data, plot=ax)
    ax.set_title(f"QQ-plot - clim ({region})")
    plt.tight_layout()
    _save(fig, out_dir, "D2_qqplot.png")

    return pd.Series({
        "Skewness"          : round(skewness_val, 4),
        "Kurtosis"          : round(kurtosis_val,  4),
        "Jarque-Bera stat"  : round(jb_stat,        4),
        "Jarque-Bera p-val" : round(jb_pvalue,       6),
        "N points"          : len(data),
        "Normale ?"         : "Non" if jb_pvalue < 0.05 else "Oui",
    }, name=region)


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES COMPARATIFS TOUTES RÉGIONS
# ══════════════════════════════════════════════════════════════════════

def graphiques_comparatifs_distributifs(dfs: dict, out_dir: Path) -> None:

    # DC1 : KDE superposées
    fig, ax = plt.subplots(figsize=(10, 5))
    for region, df in dfs.items():
        sns.kdeplot(df["clim"].dropna(), ax=ax,
                    color=COULEURS_REGIONS[region], label=region, lw=2)
    ax.set_xlabel("clim (kW)")
    ax.set_ylabel("Densité")
    ax.set_title("Distributions comparées - Climatisation (toutes régions)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, out_dir, "DC1_kde_comparatif.png")

    # DC2 : Skewness et Kurtosis comparés
    stats_list = []
    for region, df in dfs.items():
        d = df["clim"].dropna()
        stats_list.append({
            "region"  : region,
            "Skewness": skew(d),
            "Kurtosis": kurtosis(d),
        })
    df_stats = pd.DataFrame(stats_list).set_index("region")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col in zip(axes, ["Skewness", "Kurtosis"]):
        bars = ax.bar(df_stats.index, df_stats[col],
                      color=[COULEURS_REGIONS[r] for r in df_stats.index],
                      edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"{col} par région")
        ax.set_ylabel(col)
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
    print("=" * 60)

    all_stats = {}
    all_dfs   = {}

    for region, fichier in REGIONS.items():
        print(f"\n── {region} ──")

        df = charger_region(region, fichier)
        if df.empty:
            continue
        all_dfs[region] = df

        out_dir = BASE_DIR / "output" / "distributives" / region.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Graphiques → {out_dir.name}/")

        stats = analyser_distribution(df, region, out_dir)
        all_stats[region] = stats

    # Tableau comparatif
    if all_stats:
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF - TOUTES RÉGIONS")
        print("=" * 60)
        comparatif = pd.DataFrame(all_stats)
        print(comparatif.to_string())

        out_csv = BASE_DIR / "output" / "distributives" / "distributives_comparatif.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparatif.to_csv(out_csv)
        print(f"\n  Tableau sauvegardé : {out_csv.name}")

    # Graphiques comparatifs
    if len(all_dfs) > 1:
        out_comp = BASE_DIR / "output" / "distributives" / "comparatif"
        out_comp.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques comparatifs → comparatif/")
        graphiques_comparatifs_distributifs(all_dfs, out_comp)

    print("\nAnalyse distributive terminée.")


if __name__ == "__main__":
    main()
