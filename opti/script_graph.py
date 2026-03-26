# =============================================================================
# Module      : script_graph.py
# Auteur      : Équipe THERMO — ELE8080
# Date        : 2025-03
# Description : Génère les graphiques d'analyse de sensibilité OAT à partir
#               du CSV produit par script_param.py.
#
#   Graphiques produits :
#     1. Sensibilité par paramètre  — F1, RMSE, MAE vs valeur (6 PNG)
#     2. Distribution (boxplots)    — un PNG par métrique (5 PNG)
#     3. Amplitude (classement)     — un PNG par métrique (5 PNG)
#     4. Confusion empilée          — TP/FP/FN/TN par paramètre (1 PNG)
#
# Usage :
#   python script_graph.py
#   python script_graph.py --csv chemin/vers/resultats.csv --output dossier/
# =============================================================================

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION PAR DÉFAUT
# ─────────────────────────────────────────────────────────────────────────────
FICHIER_CSV_DEFAUT = Path(__file__).parent / "resultats_sensibilite_juillet.csv"
DOSSIER_SORTIE     = Path(__file__).parent / "output_graphiques"

# Métriques affichées sur les graphiques de sensibilité (3 panneaux côte à côte)
METRIQUES_SENSIBILITE = ["f1", "rmse", "mae"]

# Métriques pour lesquelles on génère boxplots et classement individuellement
METRIQUES_INDIVIDUELLES = ["f1", "rmse", "mae", "precision", "rappel"]

# Sens d'optimisation : True = plus haut est meilleur, False = plus bas est meilleur
METRIQUE_PLUS_HAUT_MEILLEUR = {
    "f1":        True,
    "rmse":      False,
    "mae":       False,
    "precision": True,
    "rappel":    True,
}

LABELS_METRIQUES = {
    "f1":        "F1-score",
    "rmse":      "RMSE (kW)",
    "mae":       "MAE (kW)",
    "precision": "Précision",
    "rappel":    "Rappel",
}

# Paramètres numériques (axe X continu) vs catégoriels (barres groupées)
PARAMS_NUMERIQUES  = ["lambda1", "d_min", "M", "x_DUTY_clim", "x_DUTY_prime_clim"]
PARAMS_CATEGORIELS = ["niveaux_clim"]

# Noms lisibles pour les titres
NOMS_PARAMS = {
    "lambda1":           "λ₁ (régularisation baseload)",
    "d_min":             "d_min (durée minimale ON/OFF)",
    "M":                 "M (Big-M thermique)",
    "x_DUTY_clim":       "x_DUTY clim (duty-cycle pointe)",
    "x_DUTY_prime_clim": "x_DUTY' clim (duty-cycle hors-pointe)",
    "niveaux_clim":      "Niveaux de puissance clim (kW)",
}

# Palette pour distinguer les clients
PALETTE_CLIENTS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

# Couleurs des boîtes pour les boxplots
COULEURS_BOXPLOTS = ["#a6cee3", "#b2df8a", "#fdbf6f", "#fb9a99",
                      "#cab2d6", "#ffff99"]


# =============================================================================
# CHARGEMENT
# =============================================================================

def charger_resultats(chemin_csv: str) -> pd.DataFrame:
    """Charge le CSV de sensibilité et convertit les types.

    Args:
        chemin_csv (str): Chemin du fichier CSV.

    Returns:
        pd.DataFrame: Données nettoyées avec valeur_num ajoutée quand possible.

    Raises:
        FileNotFoundError: Si le CSV est introuvable.

    Example:
        >>> df = charger_resultats("resultats_sensibilite_juillet.csv")
    """
    df = pd.read_csv(chemin_csv)

    # Colonne numérique pour les paramètres qui le permettent
    df["valeur_num"] = pd.to_numeric(df["valeur_testee"], errors="coerce")

    return df


# =============================================================================
# 1. GRAPHIQUES DE SENSIBILITÉ PAR PARAMÈTRE
# =============================================================================

def tracer_sensibilite_numerique(
    df: pd.DataFrame,
    param: str,
    dossier: Path,
) -> Path:
    """Trace F1 / RMSE / MAE vs la valeur d'un paramètre numérique.

    Chaque client est une ligne distincte ; la moyenne inter-clients est
    tracée en trait épais noir.

    Args:
        df      (pd.DataFrame): Sous-ensemble filtré pour ce paramètre.
        param   (str): Nom du paramètre varié (ex. "lambda1").
        dossier (Path): Dossier de sortie pour le PNG.

    Returns:
        Path: Chemin du fichier PNG sauvegardé.

    Example:
        >>> tracer_sensibilite_numerique(df_lambda, "lambda1", Path("output/"))
    """
    clients = sorted(df["dataid"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for idx_m, metrique in enumerate(METRIQUES_SENSIBILITE):
        ax = axes[idx_m]

        for idx_c, cid in enumerate(clients):
            sub = df[df["dataid"] == cid].sort_values("valeur_num")
            ax.plot(
                sub["valeur_num"], sub[metrique],
                marker="o", markersize=4, alpha=0.5,
                color=PALETTE_CLIENTS[idx_c % len(PALETTE_CLIENTS)],
                label=f"Client {cid}",
            )

        # Moyenne inter-clients
        moy = df.groupby("valeur_num")[metrique].mean().sort_index()
        ax.plot(
            moy.index, moy.values,
            "k-", lw=2.5, marker="s", markersize=6,
            label="Moyenne",
        )

        ax.set_xlabel(NOMS_PARAMS.get(param, param), fontsize=10)
        ax.set_ylabel(LABELS_METRIQUES.get(metrique, metrique), fontsize=10)
        ax.set_title(LABELS_METRIQUES.get(metrique, metrique), fontsize=11, weight="bold")
        ax.grid(True, alpha=0.3)

        # Échelle log pour M (plage 100–10 000)
        if param == "M":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(clients) + 1, fontsize=8)

    fig.suptitle(
        f"Sensibilité OAT — {NOMS_PARAMS.get(param, param)}",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    chemin = dossier / f"sensibilite_{param}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


def tracer_sensibilite_categoriel(
    df: pd.DataFrame,
    param: str,
    dossier: Path,
) -> Path:
    """Trace F1 / RMSE / MAE en barres groupées pour un paramètre catégoriel.

    Chaque groupe de barres correspond à une valeur du paramètre ; les barres
    individuelles représentent chaque client, avec la moyenne superposée.

    Args:
        df      (pd.DataFrame): Sous-ensemble filtré pour ce paramètre.
        param   (str): Nom du paramètre varié.
        dossier (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG.

    Example:
        >>> tracer_sensibilite_categoriel(df_niveaux, "niveaux_clim", Path("out/"))
    """
    clients   = sorted(df["dataid"].unique())
    valeurs   = df["valeur_testee"].unique()
    n_vals    = len(valeurs)
    n_clients = len(clients)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Labels raccourcis : afficher seulement les niveaux ON
    labels_courts = []
    for v in valeurs:
        niv = [float(x) for x in v.strip("[]").split(",")]
        labels_courts.append(" / ".join(f"{n:.1f}" for n in niv if n > 0))

    x_pos   = np.arange(n_vals)
    largeur = 0.7 / n_clients

    for idx_m, metrique in enumerate(METRIQUES_SENSIBILITE):
        ax = axes[idx_m]

        for idx_c, cid in enumerate(clients):
            sub = df[df["dataid"] == cid]
            vals_m = []
            for v in valeurs:
                row = sub[sub["valeur_testee"] == v]
                vals_m.append(row[metrique].values[0] if len(row) > 0 else np.nan)

            decalage = (idx_c - n_clients / 2 + 0.5) * largeur
            ax.bar(
                x_pos + decalage, vals_m, largeur,
                color=PALETTE_CLIENTS[idx_c % len(PALETTE_CLIENTS)],
                alpha=0.7, label=f"Client {cid}" if idx_m == 0 else None,
            )

        # Moyenne par valeur
        moy_vals = [df[df["valeur_testee"] == v][metrique].mean() for v in valeurs]
        ax.plot(x_pos, moy_vals, "ks-", markersize=7, lw=2,
                label="Moyenne" if idx_m == 0 else None)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_courts, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(LABELS_METRIQUES.get(metrique, metrique), fontsize=10)
        ax.set_title(LABELS_METRIQUES.get(metrique, metrique), fontsize=11, weight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_clients + 1, fontsize=8)

    fig.suptitle(
        f"Sensibilité OAT — {NOMS_PARAMS.get(param, param)}",
        fontsize=13, weight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])

    chemin = dossier / f"sensibilite_{param}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# 2. DISTRIBUTION (BOXPLOTS) — UN GRAPHIQUE PAR MÉTRIQUE
# =============================================================================

def tracer_boxplots(df: pd.DataFrame, metrique: str, dossier: Path) -> Path:
    """Trace les boxplots d'une métrique pour chaque paramètre varié.

    Permet de comparer la dispersion inter-clients. La baseline est
    affichée comme ligne horizontale de référence.

    Args:
        df       (pd.DataFrame): Résultats complets.
        metrique (str): Nom de la métrique (ex. "f1", "rmse").
        dossier  (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG sauvegardé.

    Example:
        >>> tracer_boxplots(df, "f1", Path("output/"))
    """
    params = [p for p in df["param_varie"].unique() if p != "baseline"]
    label  = LABELS_METRIQUES.get(metrique, metrique)

    fig, ax = plt.subplots(figsize=(12, 5))

    donnees_box = []
    labels_box  = []
    for param in params:
        sub = df[df["param_varie"] == param]
        donnees_box.append(sub[metrique].dropna().values)
        labels_box.append(NOMS_PARAMS.get(param, param))

    bp = ax.boxplot(
        donnees_box,
        tick_labels=labels_box,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
    )

    for patch, couleur in zip(bp["boxes"], COULEURS_BOXPLOTS):
        patch.set_facecolor(couleur)
        patch.set_alpha(0.7)

    # Référence baseline
    baseline_moy = df[df["param_varie"] == "baseline"][metrique].mean()
    ax.axhline(baseline_moy, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"Baseline moy. ({label}={baseline_moy:.3f})")

    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f"Distribution {label} par paramètre varié", fontsize=12, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()

    chemin = dossier / f"boxplots_{metrique}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# 3. AMPLITUDE (CLASSEMENT) — UN GRAPHIQUE PAR MÉTRIQUE
# =============================================================================

def tracer_classement(df: pd.DataFrame, metrique: str, dossier: Path) -> Path:
    """Trace un barplot horizontal de l'amplitude pire-meilleur pour une métrique.

    Le sens « meilleur » dépend de la métrique : F1/précision/rappel = max,
    RMSE/MAE = min. Les paramètres sont triés par amplitude décroissante.

    Args:
        df       (pd.DataFrame): Résultats complets.
        metrique (str): Nom de la métrique.
        dossier  (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG.

    Example:
        >>> tracer_classement(df, "rmse", Path("output/"))
    """
    params       = [p for p in df["param_varie"].unique() if p != "baseline"]
    label        = LABELS_METRIQUES.get(metrique, metrique)
    plus_haut_ok = METRIQUE_PLUS_HAUT_MEILLEUR[metrique]

    resultats = []
    for param in params:
        sub         = df[df["param_varie"] == param]
        moy_par_val = sub.groupby("valeur_testee")[metrique].mean()

        if plus_haut_ok:
            meilleur_val = moy_par_val.idxmax()
            meilleur     = moy_par_val.max()
            pire         = moy_par_val.min()
        else:
            # Pour RMSE/MAE, le meilleur est le plus bas
            meilleur_val = moy_par_val.idxmin()
            meilleur     = moy_par_val.min()
            pire         = moy_par_val.max()

        resultats.append({
            "param":        NOMS_PARAMS.get(param, param),
            "meilleur_val": meilleur_val,
            "meilleur":     meilleur,
            "pire":         pire,
            "amplitude":    abs(meilleur - pire),
        })

    # Trier par amplitude décroissante
    resultats.sort(key=lambda r: r["amplitude"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(resultats))

    # Barres horizontales entre pire et meilleur
    for i, r in enumerate(resultats):
        gauche = min(r["pire"], r["meilleur"])
        ax.barh(i, r["amplitude"], left=gauche, color="#4c72b0", alpha=0.7, height=0.5)

    # Annotations : toujours à droite de la barre pour éviter le chevauchement
    for i, r in enumerate(resultats):
        droite = max(r["pire"], r["meilleur"])
        ax.text(droite + 0.005, i,
                f"{label}={r['meilleur']:.3f}  (val={r['meilleur_val'][:15]})",
                va="center", fontsize=8, weight="bold", ha="left")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["param"] for r in resultats], fontsize=9)
    ax.set_xlabel(label, fontsize=10)

    sens = "↑ meilleur" if plus_haut_ok else "↓ meilleur"
    ax.set_title(
        f"Sensibilité des paramètres — Amplitude {label} (pire → meilleur, {sens})",
        fontsize=12, weight="bold",
    )
    ax.grid(True, alpha=0.2, axis="x")

    # Baseline référence
    baseline_val = df[df["param_varie"] == "baseline"][metrique].mean()
    ax.axvline(baseline_val, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"Baseline ({label}={baseline_val:.3f})")
    ax.legend(fontsize=9)

    fig.tight_layout()
    chemin = dossier / f"classement_{metrique}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# 4. CONFUSION EMPILÉE — TP/FP/FN/TN PAR PARAMÈTRE
# =============================================================================

def tracer_confusion_empilee(df: pd.DataFrame, dossier: Path) -> Path:
    """Trace les TP/FP/FN/TN empilés pour chaque valeur de chaque paramètre.

    Chaque sous-graphique est un paramètre ; les barres empilées montrent
    la répartition des prédictions moyennées sur les clients.

    Args:
        df      (pd.DataFrame): Résultats complets.
        dossier (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG.

    Example:
        >>> tracer_confusion_empilee(df, Path("output/"))
    """
    params = [p for p in df["param_varie"].unique() if p != "baseline"]
    n_cols = 3
    n_rows = int(np.ceil(len(params) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).flatten()

    couleurs_conf = {"tp": "#2ca02c", "fp": "#d62728", "fn": "#ff7f0e", "tn": "#aec7e8"}
    labels_conf   = {"tp": "TP", "fp": "FP", "fn": "FN", "tn": "TN"}

    for idx, param in enumerate(params):
        ax  = axes_flat[idx]
        sub = df[df["param_varie"] == param]

        if param in PARAMS_NUMERIQUES:
            moy = sub.groupby("valeur_num")[["tp", "fp", "fn", "tn"]].mean().sort_index()
            x_labels = [f"{v:g}" for v in moy.index]
        else:
            moy = sub.groupby("valeur_testee")[["tp", "fp", "fn", "tn"]].mean()
            x_labels = [v[:20] for v in moy.index]

        x   = np.arange(len(moy))
        bas = np.zeros(len(moy))

        for composante in ["tp", "fp", "fn", "tn"]:
            vals = moy[composante].values
            ax.bar(x, vals, bottom=bas, color=couleurs_conf[composante],
                   label=labels_conf[composante] if idx == 0 else None,
                   alpha=0.8, width=0.6)
            bas += vals

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(NOMS_PARAMS.get(param, param), fontsize=9, weight="bold")
        ax.set_ylabel("Nombre de pas (moy.)", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    for idx in range(len(params), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=couleurs_conf[c], alpha=0.8)
               for c in ["tp", "fp", "fn", "tn"]]
    fig.legend(handles, ["TP", "FP", "FN", "TN"],
               loc="lower center", ncol=4, fontsize=9)

    fig.suptitle("Matrice de confusion empilée par paramètre (moyenne clients)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    chemin = dossier / "confusion_empilee_parametres.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Point d'entrée : charge le CSV et génère tous les graphiques.

    Produit 17 PNG au total :
      - 6 sensibilité (un par paramètre, 3 métriques chacun)
      - 5 boxplots (un par métrique)
      - 5 classements amplitude (un par métrique)
      - 1 confusion empilée

    Returns:
        None

    Example:
        >>> # python script_graph.py --csv resultats.csv --output output_graphiques/
    """
    parser = argparse.ArgumentParser(
        description="Graphiques de sensibilité OAT — THERMO ELE8080"
    )
    parser.add_argument(
        "--csv", type=str, default=str(FICHIER_CSV_DEFAUT),
        help="Chemin du CSV de résultats (défaut : resultats_sensibilite_juillet.csv)",
    )
    parser.add_argument(
        "--output", type=str, default=str(DOSSIER_SORTIE),
        help="Dossier de sortie des graphiques",
    )
    args = parser.parse_args()

    dossier = Path(args.output)
    dossier.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GÉNÉRATION DES GRAPHIQUES DE SENSIBILITÉ OAT")
    print("=" * 70)

    # ── Chargement ────────────────────────────────────────────────────────────
    print(f"\n[1] Chargement : {args.csv}")
    df = charger_resultats(args.csv)
    print(f"    {len(df)} lignes — {df['dataid'].nunique()} clients — "
          f"{df['param_varie'].nunique()} paramètres")

    # ── 1. Sensibilité par paramètre (F1 / RMSE / MAE) ───────────────────────
    print("\n[2] Graphiques de sensibilité par paramètre...")
    for param in df["param_varie"].unique():
        if param == "baseline":
            continue
        sub = df[df["param_varie"] == param]
        if param in PARAMS_CATEGORIELS:
            tracer_sensibilite_categoriel(sub, param, dossier)
        else:
            tracer_sensibilite_numerique(sub, param, dossier)

    # ── 2. Boxplots — un par métrique ─────────────────────────────────────────
    print("\n[3] Boxplots de distribution par métrique...")
    for metrique in METRIQUES_INDIVIDUELLES:
        tracer_boxplots(df, metrique, dossier)

    # ── 3. Classement amplitude — un par métrique ─────────────────────────────
    print("\n[4] Classement amplitude par métrique...")
    for metrique in METRIQUES_INDIVIDUELLES:
        tracer_classement(df, metrique, dossier)

    # ── 4. Confusion empilée ──────────────────────────────────────────────────
    print("\n[5] Confusion empilée...")
    tracer_confusion_empilee(df, dossier)

    # ── Résumé ────────────────────────────────────────────────────────────────
    n_fichiers = len(list(dossier.glob("*.png")))
    print(f"\n{'=' * 70}")
    print(f"TERMINÉ — {n_fichiers} graphiques sauvegardés dans : {dossier}")
    print("=" * 70)


if __name__ == "__main__":
    main()