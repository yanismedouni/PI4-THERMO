import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

DATAID               = None   # None = tous les clients, int = un seul client
SEUIL_ON             = 0.05   # kW - seuil de détection état ON
SEUIL_TEMPERATURE    = 22     # °C - température minimale pour l'analyse de corrélation
SEUIL_FREQ_ACTIVATION = 0.10  # fréquence d'activation au-delà de laquelle on détecte le seuil thermique

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
# ANALYSE DE CORRÉLATION D'UNE RÉGION
# ══════════════════════════════════════════════════════════════════════

def analyser_correlation(df: pd.DataFrame, region: str, out_dir: Path) -> dict:
    """Analyse la corrélation température ↔ clim pour une région.
    Retourne un dict avec les indicateurs calculés."""

    couleur = COULEURS_REGIONS.get(region, "#2E75B6")
    df_seuil = df[["temp", "clim"]].dropna().copy()
    df_seuil["clim_on"] = (df_seuil["clim"] >= SEUIL_ON).astype(int)

    if len(df_seuil) == 0:
        print(f"  {region} : aucune donnée disponible.")
        return {}

    nb_nan = df["temp"].isna().sum()
    if nb_nan > 0:
        print(f"  Attention : {nb_nan} valeurs de température manquantes ignorées")

    # ── C1 : Fréquence d'activation par intervalle de température ────
    borne_min = int(np.floor(df_seuil["temp"].min()))
    borne_max = int(np.ceil(df_seuil["temp"].max()))
    bins = np.arange(borne_min, borne_max + 1, 1)

    df_seuil["temp_bin"] = pd.cut(df_seuil["temp"], bins=bins,
                                   right=True, include_lowest=True)
    freq_activation = (df_seuil.groupby("temp_bin", observed=False)["clim_on"]
                        .mean().dropna())

    print(f"\n  Fréquence d'activation par intervalle de température :")
    print(freq_activation.to_string())

    seuil_freq = freq_activation[freq_activation > SEUIL_FREQ_ACTIVATION]
    seuil_thermique = seuil_freq.index[0] if len(seuil_freq) > 0 else None
    if seuil_thermique:
        print(f"\n  Température seuil (fréquence > {SEUIL_FREQ_ACTIVATION*100:.0f}%) : "
              f"{seuil_thermique}")
    else:
        print(f"\n  Aucun intervalle n'atteint une fréquence d'activation "
              f"> {SEUIL_FREQ_ACTIVATION*100:.0f}%.")

    fig, ax = plt.subplots(figsize=(10, 5))
    freq_activation.plot(marker="o", color=couleur, ax=ax)
    ax.axhline(SEUIL_FREQ_ACTIVATION, color="red", ls="--", lw=1.2,
               label=f"Seuil {SEUIL_FREQ_ACTIVATION*100:.0f}%")
    ax.set_xlabel("Intervalle de température (°C)")
    ax.set_ylabel("Fréquence d'activation de la clim")
    ax.set_title(f"Fréquence d'activation de la clim selon la température ({region})")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    _save(fig, out_dir, "C1_freq_activation_temperature.png")

    # ── C2 : Corrélation température ↔ clim ──────────────────────────
    df_corr = df_seuil[
        (df_seuil["temp"] >= SEUIL_TEMPERATURE) &
        (df_seuil["clim"] >= SEUIL_ON)
    ].copy()

    print(f"\n  Points utilisés pour la corrélation : {len(df_corr)}")

    resultats = {"region": region, "seuil_thermique": str(seuil_thermique)}

    if len(df_corr) < 10:
        print("  Pas assez de données pour calculer des coefficients fiables.")
        return resultats

    corr_pearson  = df_corr["temp"].corr(df_corr["clim"], method="pearson")
    corr_spearman = df_corr["temp"].corr(df_corr["clim"], method="spearman")
    pente, intercept = np.polyfit(df_corr["temp"], df_corr["clim"], 1)

    print(f"\n  Coefficients de corrélation :")
    print(f"    Pearson  : {corr_pearson:.4f}")
    print(f"    Spearman : {corr_spearman:.4f}")
    print(f"    Gradient thermique : {pente:.6f} kW/°C")
    print(f"    Intercept : {intercept:.6f} kW")

    x_reg = np.linspace(df_corr["temp"].min(), df_corr["temp"].max(), 200)
    y_reg = pente * x_reg + intercept

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_corr["temp"], df_corr["clim"],
               alpha=0.3, color=couleur, label="Observations")
    ax.plot(x_reg, y_reg, color="black", lw=2,
            label=f"Régression (Pearson = {corr_pearson:.3f})")
    ax.set_xlabel("Température extérieure (°C)")
    ax.set_ylabel("Consommation clim (kW)")
    ax.set_title(f"Corrélation température ↔ clim ({region})")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    _save(fig, out_dir, "C2_correlation_temperature_clim.png")

    resultats.update({
        "Pearson"            : round(corr_pearson,  4),
        "Spearman"           : round(corr_spearman, 4),
        "Gradient (kW/°C)"  : round(pente,          6),
        "Intercept (kW)"    : round(intercept,       6),
        "N points corrélation": len(df_corr),
    })
    return resultats


# ══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("ANALYSE DE CORRÉLATION TEMPÉRATURE ↔ CLIM - THERMO NILM")
    print("=" * 60)

    all_resultats = {}

    for region, fichier in REGIONS.items():
        print(f"\n── {region} ──")

        df = charger_region(region, fichier)
        if df.empty:
            continue

        out_dir = BASE_DIR / "output" / "correlation" / region.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques → {out_dir.name}/")

        resultats = analyser_correlation(df, region, out_dir)
        if resultats:
            all_resultats[region] = resultats

    # Tableau comparatif
    if all_resultats:
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF - TOUTES RÉGIONS")
        print("=" * 60)
        comparatif = pd.DataFrame(all_resultats).T
        print(comparatif.to_string())

        out_csv = BASE_DIR / "output" / "correlation" / "correlation_comparatif.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparatif.to_csv(out_csv)
        print(f"\n  Tableau sauvegardé : {out_csv.name}")

    print("\nAnalyse de corrélation terminée.")


if __name__ == "__main__":
    main()
