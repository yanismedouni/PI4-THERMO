import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Régions à analyser : nom affiché → nom du fichier CSV
REGIONS = {
    "Austin"    : "processed_energy_data_austin.csv",
    "California": "processed_energy_data_california.csv",
    "New York"  : "processed_energy_data_newyork.csv",
}

DATAID         = None    # None = tous les clients, int = un seul client
SEUIL_ON       = 0.05    # kW - seuil de détection état ON
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

ORDRE_SAISONS   = ["Hiver", "Printemps", "Été", "Automne"]
PALETTE_SAISONS = ["#D6E4F0", "#2E75B6", "#1F3864", "#0A1628"]

MAP_SAISON = {
    12: "Hiver",     1: "Hiver",     2: "Hiver",
     3: "Printemps", 4: "Printemps", 5: "Printemps",
     6: "Été",       7: "Été",       8: "Été",
     9: "Automne",  10: "Automne",  11: "Automne",
}

# Couleur distincte par région pour les graphiques comparatifs
COULEURS_REGIONS = {
    "Austin"    : "#2E75B6",
    "California": "#E06C2E",
    "New York"  : "#2E8B57",
}


# ══════════════════════════════════════════════════════════════════════
# CHARGEMENT D'UNE RÉGION
# ══════════════════════════════════════════════════════════════════════

def charger_region(nom_region: str, nom_fichier: str) -> pd.DataFrame:
    """Charge et nettoie le CSV d'une région. Retourne un DataFrame prêt."""
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

    # Reconstruction datetime sans local_15min
    df["datetime"] = pd.to_datetime({
        "year": df["year"], "month": df["month"], "day": df["day"],
        "hour": df["hour"], "minute": df["minute"],
    }, errors="coerce")

    n_nat = df["datetime"].isna().sum()
    if n_nat > 0:
        print(f"   {nom_region} : {n_nat} horodatages invalides → exclus")
        df = df.dropna(subset=["datetime"])

    # Nettoyage numérique
    for col in ["grid", "clim", "chauffage", "temp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["clim"]      = df["clim"].fillna(0)
    df["chauffage"] = df["chauffage"].fillna(0)
    df = df.dropna(subset=["grid"])

    # Filtrage client optionnel
    if DATAID is not None:
        df = df[df["dataid"] == DATAID]

    df = df.set_index("datetime").sort_index()
    df["saison"] = df["month"].map(MAP_SAISON)
    df["region"] = nom_region

    print(f"  {nom_region} : {len(df):,} observations chargées")
    return df


# ══════════════════════════════════════════════════════════════════════
# STATISTIQUES DESCRIPTIVES D'UNE RÉGION
# ══════════════════════════════════════════════════════════════════════

def calculer_stats(df: pd.DataFrame, region: str) -> pd.Series:
    """Retourne un pd.Series de statistiques descriptives pour une région."""
    P = df["clim"]
    P_on = P[P > SEUIL_ON]
    mode_on = P_on.mode()

    return pd.Series({
        # Tendance centrale
        "Moyenne (kW)"               : P.mean(),
        "Médiane (kW)"               : P.median(),
        "Mode - états ON (kW)"       : mode_on.iloc[0] if not mode_on.empty else float("nan"),
        # Dispersion
        "Écart-type (kW)"            : P.std(),
        "Variance (kW²)"             : P.var(),
        "Min (kW)"                   : P.min(),
        "Max (kW)"                   : P.max(),
        "Coeff. de variation (%)"    : (P.std() / P.mean()) * 100,
        # Forme
        "Asymétrie (skewness)"       : P.skew(),
        "Aplatissement (kurtosis)"   : P.kurt(),
        # Percentiles
        "P5  (kW)"                   : P.quantile(0.05),
        "P10 (kW)"                   : P.quantile(0.10),
        "Q1  (kW)"                   : P.quantile(0.25),
        "Q2  (kW)"                   : P.quantile(0.50),
        "Q3  (kW)"                   : P.quantile(0.75),
        "P90 (kW)"                   : P.quantile(0.90),
        "P95 (kW)"                   : P.quantile(0.95),
        "IQR (kW)"                   : P.quantile(0.75) - P.quantile(0.25),
        # Activité
        "% temps ON"                 : (P > SEUIL_ON).mean() * 100,
    }, name=region).round(4)


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES PAR RÉGION
# ══════════════════════════════════════════════════════════════════════

def graphiques_region(df: pd.DataFrame, region: str, out_dir: Path) -> None:
    """Génère les 7 graphiques descriptifs pour une région."""
    P      = df["clim"]
    P_on   = P[P > SEUIL_ON]
    couleur = COULEURS_REGIONS.get(region, PALETTE["principal"])

    # G1 - Histogramme
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(P_on, bins=40, color=couleur, edgecolor="white", alpha=0.85)
    ax.axvline(P.mean(),   color="red",    ls="--", lw=1.8,
               label=f"Moyenne = {P.mean():.3f} kW")
    ax.axvline(P.median(), color="orange", ls="--", lw=1.8,
               label=f"Médiane = {P.median():.3f} kW")
    ax.set_xlabel("Puissance (kW)")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution des puissances - Climatisation ({region})")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "G1_histogramme.png")

    # G2 - Fréquences par intervalles
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
    ax.set_title(f"Distribution des fréquences par intervalles ({region})")
    plt.tight_layout()
    _save(fig, out_dir, "G2_freq_intervalles.png")

    # G3 - Boxplot par saison
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="saison", y="clim",
                order=ORDRE_SAISONS, palette=PALETTE_SAISONS, ax=ax)
    ax.set_xlabel("Saison")
    ax.set_ylabel("Puissance climatisation (kW)")
    ax.set_title(f"Distribution par saison ({region})")
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
    ax.set_title(f"Profil horaire moyen - Climatisation ({region})")
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
    ax.set_title(f"Consommation moyenne - Heure × Mois ({region})")
    ax.set_ylabel("Heure")
    ax.set_xlabel("Mois")
    plt.tight_layout()
    _save(fig, out_dir, "G5_heatmap_heure_mois.png")



# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES COMPARATIFS TOUTES RÉGIONS
# ══════════════════════════════════════════════════════════════════════

def graphiques_comparatifs(dfs: dict, out_dir: Path) -> None:
    """Génère les graphiques comparant les 3 régions simultanément."""

    # GC1 - Profils horaires superposés
    fig, ax = plt.subplots(figsize=(12, 5))
    for region, df in dfs.items():
        profil = df.groupby("hour")["clim"].mean()
        ax.plot(profil.index, profil.values, lw=2,
                color=COULEURS_REGIONS[region], label=region)
    ax.set_xlabel("Heure de la journée")
    ax.set_ylabel("Puissance moyenne (kW)")
    ax.set_title("Profils horaires moyens - Comparaison régions")
    ax.set_xticks(range(0, 24))
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "GC1_profils_horaires_comparatif.png")

    # GC2 - Boxplots côte à côte par région
    df_all = pd.concat(dfs.values())
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df_all, x="region", y="clim",
                order=list(dfs.keys()),
                palette=list(COULEURS_REGIONS.values()),
                ax=ax)
    ax.set_xlabel("Région")
    ax.set_ylabel("Puissance climatisation (kW)")
    ax.set_title("Distribution de la climatisation par région")
    plt.tight_layout()
    _save(fig, out_dir, "GC2_boxplot_regions.png")

    # GC3 - % temps ON par région et saison
    pct_on = (df_all.groupby(["region", "saison"])["clim"]
              .apply(lambda x: (x > SEUIL_ON).mean() * 100)
              .unstack("saison")
              .reindex(columns=ORDRE_SAISONS))

    fig, ax = plt.subplots(figsize=(11, 5))
    pct_on.plot(kind="bar", ax=ax,
                color=["#D6E4F0", "#2E75B6", "#1F3864", "#0A1628"],
                edgecolor="white")
    ax.set_xlabel("Région")
    ax.set_ylabel("% temps ON")
    ax.set_title("Pourcentage de temps ON - Par région et saison")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Saison")
    plt.tight_layout()
    _save(fig, out_dir, "GC3_pct_on_region_saison.png")


# ══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("ANALYSE DESCRIPTIVE - THERMO NILM")
    print("=" * 60)

    all_stats = {}
    all_dfs   = {}

    for region, fichier in REGIONS.items():
        print(f"\n── {region} ──")

        # Chargement
        df = charger_region(region, fichier)
        if df.empty:
            continue
        all_dfs[region] = df

        # Statistiques
        stats = calculer_stats(df, region)
        all_stats[region] = stats

        # Distribution des fréquences
        df["intervalle"] = pd.cut(df["clim"], bins=BINS_PUISSANCE,
                                   labels=LABELS_BINS, right=True)
        freq_abs = df["intervalle"].value_counts().sort_index()
        freq_rel = (freq_abs / freq_abs.sum() * 100).round(2)
        print(f"\n  Distribution des fréquences :")
        print(pd.DataFrame({
            "Fréquence absolue"    : freq_abs,
            "Fréquence relative (%)": freq_rel,
        }).to_string())

        # Graphiques individuels
        out_region = BASE_DIR / "output" / "stats" / region.replace(" ", "_")
        out_region.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques → {out_region.name}/")
        graphiques_region(df, region, out_region)

    # Tableau comparatif toutes régions
    if all_stats:
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF - TOUTES RÉGIONS")
        print("=" * 60)
        comparatif = pd.DataFrame(all_stats)
        print(comparatif.to_string())

        # Sauvegarde CSV du comparatif
        out_csv = BASE_DIR / "output" / "stats" / "stats_comparatif.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparatif.to_csv(out_csv)
        print(f"\n  Tableau sauvegardé : {out_csv.name}")

    # Graphiques comparatifs
    if len(all_dfs) > 1:
        out_comp = BASE_DIR / "output" / "stats" / "comparatif"
        out_comp.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques comparatifs → comparatif/")
        graphiques_comparatifs(all_dfs, out_comp)

    print("\nAnalyse terminée.")


if __name__ == "__main__":
    main()
