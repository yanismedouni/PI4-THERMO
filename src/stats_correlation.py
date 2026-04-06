import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import (
    BASE_DIR, SEUIL_ON,
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

SEUIL_TEMPERATURE     = 21    # °C - température minimale pour la corrélation
SEUIL_FREQ_ACTIVATION = 0.10  # seuil de fréquence pour détecter le seuil thermique


# ══════════════════════════════════════════════════════════════════════
# ANALYSE DE CORRÉLATION D'UNE SOURCE
# ══════════════════════════════════════════════════════════════════════

def analyser_correlation(df: pd.DataFrame, label: str, out_dir: Path) -> dict:
    """Analyse la corrélation température ↔ clim. Retourne un dict d'indicateurs."""

    couleur  = get_couleur(label)
    df_seuil = df[["temp", "clim"]].dropna().copy()
    df_seuil["clim_on"] = (df_seuil["clim"] >= SEUIL_ON).astype(int)

    if len(df_seuil) == 0:
        print(f"  {label} : aucune donnée disponible.")
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

    seuil_freq      = freq_activation[freq_activation > SEUIL_FREQ_ACTIVATION]
    seuil_thermique = seuil_freq.index[0] if len(seuil_freq) > 0 else None
    if seuil_thermique:
        print(f"\n  Température seuil (fréquence > {SEUIL_FREQ_ACTIVATION*100:.0f}%) : "
              f"{seuil_thermique}")
    else:
        print(f"\n  Aucun intervalle n'atteint une fréquence > {SEUIL_FREQ_ACTIVATION*100:.0f}%.")

    fig, ax = plt.subplots(figsize=(10, 5))
    freq_activation.plot(marker="o", color=couleur, ax=ax)
    ax.axhline(SEUIL_FREQ_ACTIVATION, color="red", ls="--", lw=1.2,
               label=f"Seuil {SEUIL_FREQ_ACTIVATION*100:.0f}%")
    ax.set_xlabel("Intervalle de température (°C)")
    ax.set_ylabel("Fréquence d'activation de la clim")
    ax.set_title(f"Fréquence d'activation de la clim selon la température ({label})")
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

    resultats = {"region": label, "seuil_thermique": str(seuil_thermique)}

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
    ax.set_title(f"Corrélation température ↔ clim ({label})")
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
    print(f"Mode : {MODE}")
    print("=" * 60)

    all_dfs      = charger_sources(MODE, FICHIERS_DESAGREGATION)
    all_resultats = {}

    for label, df in all_dfs.items():
        print(f"\n── {label} ──")

        slug    = label.replace(" ", "_").replace("/", "-")
        out_dir = BASE_DIR / "output" / "correlation" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Graphiques → {out_dir.name}/")

        resultats = analyser_correlation(df, label, out_dir)
        if resultats:
            all_resultats[label] = resultats

    if all_resultats:
        print("\n" + "=" * 60)
        print("TABLEAU COMPARATIF")
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
