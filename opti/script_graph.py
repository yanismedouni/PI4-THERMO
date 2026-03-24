# =============================================================================
# Module      : script_graph.py
# Auteur      : Équipe THERMO — ELE8080
# Date        : 2025-03
# Description : Génère les graphiques d'analyse de sensibilité OAT à partir
#               du CSV produit par script_param.py. Un graphique par paramètre
#               varié (F1, RMSE, MAE vs valeur du paramètre) plus des
#               graphiques synthétiques (heatmap, radar, matrice de confusion,
#               précision–rappel, boxplots).
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

# Métriques affichées sur chaque graphique de sensibilité
METRIQUES_PRINCIPALES = ["f1", "rmse", "mae"]
LABELS_METRIQUES = {
    "f1":        "F1-score",
    "rmse":      "RMSE (kW)",
    "mae":       "MAE (kW)",
    "precision": "Précision",
    "rappel":    "Rappel",
    "diff_moy":  "Biais moyen (kW)",
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

    # Créer une colonne numérique pour les paramètres qui le permettent
    df["valeur_num"] = pd.to_numeric(df["valeur_testee"], errors="coerce")

    return df


# =============================================================================
# GRAPHIQUES PAR PARAMÈTRE — NUMÉRIQUES
# =============================================================================

def tracer_sensibilite_numerique(
    df: pd.DataFrame,
    param: str,
    dossier: Path,
) -> Path:
    """Trace F1 / RMSE / MAE vs la valeur d'un paramètre numérique.

    Chaque client est une ligne distincte ; la moyenne inter-clients est
    tracée en trait épais. Le point baseline est marqué par une ligne
    verticale pointillée.

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

    for idx_m, metrique in enumerate(METRIQUES_PRINCIPALES):
        ax = axes[idx_m]

        # Ligne par client
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

    # Légende commune en bas
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


# =============================================================================
# GRAPHIQUES PAR PARAMÈTRE — CATÉGORIELS (niveaux_clim)
# =============================================================================

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
    clients  = sorted(df["dataid"].unique())
    valeurs  = df["valeur_testee"].unique()
    n_vals   = len(valeurs)
    n_clients = len(clients)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Labels raccourcis pour lisibilité
    labels_courts = []
    for v in valeurs:
        niv = [float(x) for x in v.strip("[]").split(",")]
        # Afficher seulement les niveaux ON (retirer le 0.0)
        labels_courts.append(" / ".join(f"{n:.1f}" for n in niv if n > 0))

    x_pos = np.arange(n_vals)
    largeur = 0.7 / n_clients

    for idx_m, metrique in enumerate(METRIQUES_PRINCIPALES):
        ax = axes[idx_m]

        for idx_c, cid in enumerate(clients):
            sub = df[df["dataid"] == cid]
            # Aligner les valeurs dans le même ordre
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
        moy_vals = []
        for v in valeurs:
            moy_vals.append(df[df["valeur_testee"] == v][metrique].mean())
        ax.plot(x_pos, moy_vals, "ks-", markersize=7, lw=2, label="Moyenne" if idx_m == 0 else None)

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
# HEATMAP — F1 PAR (PARAMÈTRE × VALEUR) MOYENNÉ SUR CLIENTS
# =============================================================================

def tracer_heatmap_f1(df: pd.DataFrame, dossier: Path) -> Path:
    """Trace une heatmap du F1-score moyen par (paramètre, valeur testée).

    Les lignes sont les paramètres variés, les colonnes les valeurs testées.
    Le F1 est moyenné sur tous les clients pour chaque combinaison.

    Args:
        df      (pd.DataFrame): Résultats complets.
        dossier (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG.

    Example:
        >>> tracer_heatmap_f1(df, Path("output/"))
    """
    params_a_tracer = [p for p in df["param_varie"].unique() if p != "baseline"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Construire la matrice : lignes = params, colonnes = valeurs ordonnées
    lignes_labels = []
    matrice = []
    annotations = []

    for param in params_a_tracer:
        sub = df[df["param_varie"] == param]
        moy = sub.groupby("valeur_testee")["f1"].mean()

        # Trier numériquement si possible
        try:
            moy.index = moy.index.astype(float)
            moy = moy.sort_index()
            labels_col = [f"{v:g}" for v in moy.index]
        except (ValueError, TypeError):
            labels_col = list(moy.index)

        lignes_labels.append(NOMS_PARAMS.get(param, param))
        matrice.append(moy.values)
        annotations.append(labels_col)

    # Normaliser en matrice rectangulaire (padding NaN)
    max_cols = max(len(row) for row in matrice)
    mat_rect = np.full((len(matrice), max_cols), np.nan)
    for i, row in enumerate(matrice):
        mat_rect[i, :len(row)] = row

    im = ax.imshow(mat_rect, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)

    # Annotations : F1 + nom de la valeur
    for i in range(len(matrice)):
        for j in range(len(matrice[i])):
            val = matrice[i][j]
            label = annotations[i][j]
            ax.text(j, i, f"{val:.2f}\n({label})", ha="center", va="center",
                    fontsize=7, weight="bold")

    ax.set_yticks(range(len(lignes_labels)))
    ax.set_yticklabels(lignes_labels, fontsize=9)
    ax.set_xticks([])
    ax.set_title("Heatmap F1-score moyen par paramètre et valeur testée", fontsize=12, weight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("F1-score", fontsize=10)

    fig.tight_layout()
    chemin = dossier / "heatmap_f1_parametres.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# RADAR — COMPARAISON BASELINE PAR CLIENT
# =============================================================================

def tracer_radar_baseline(df: pd.DataFrame, dossier: Path) -> Path:
    """Trace un radar (spider chart) des métriques baseline par client.

    Les axes du radar sont : F1, Précision, Rappel, 1−RMSE, 1−MAE.
    Chaque client est un polygone distinct.

    Args:
        df      (pd.DataFrame): Résultats complets.
        dossier (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG.

    Example:
        >>> tracer_radar_baseline(df, Path("output/"))
    """
    df_base = df[df["param_varie"] == "baseline"].copy()
    if df_base.empty:
        print("  [SKIP] Pas de baseline dans les données.")
        return None

    # Métriques normalisées entre 0 et 1 (plus haut = meilleur)
    axes_radar = ["F1", "Précision", "Rappel", "1 − RMSE", "1 − MAE"]
    clients = sorted(df_base["dataid"].unique())

    angles = np.linspace(0, 2 * np.pi, len(axes_radar), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le polygone

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for idx_c, cid in enumerate(clients):
        row = df_base[df_base["dataid"] == cid].iloc[0]
        valeurs = [
            row["f1"],
            row["precision"],
            row["rappel"],
            max(0, 1 - row["rmse"]),
            max(0, 1 - row["mae"]),
        ]
        valeurs += valeurs[:1]

        ax.plot(angles, valeurs, "o-", lw=2, markersize=5,
                color=PALETTE_CLIENTS[idx_c % len(PALETTE_CLIENTS)],
                label=f"Client {cid}")
        ax.fill(angles, valeurs,
                color=PALETTE_CLIENTS[idx_c % len(PALETTE_CLIENTS)],
                alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_radar, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Profil baseline par client", fontsize=12, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)

    fig.tight_layout()
    chemin = dossier / "radar_baseline_clients.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# PRÉCISION VS RAPPEL — PAR PARAMÈTRE
# =============================================================================

def tracer_precision_rappel(df: pd.DataFrame, dossier: Path) -> Path:
    """Trace le compromis précision–rappel pour chaque paramètre varié.

    Chaque sous-graphique correspond à un paramètre. Les points sont
    colorés selon la valeur du paramètre ; la taille reflète le F1-score.

    Args:
        df      (pd.DataFrame): Résultats complets.
        dossier (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG.

    Example:
        >>> tracer_precision_rappel(df, Path("output/"))
    """
    params = [p for p in df["param_varie"].unique() if p != "baseline"]
    n_cols = 3
    n_rows = int(np.ceil(len(params) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, param in enumerate(params):
        ax = axes_flat[idx]
        sub = df[df["param_varie"] == param]

        # Colorer par valeur numérique si possible
        if param in PARAMS_NUMERIQUES:
            scatter = ax.scatter(
                sub["rappel"], sub["precision"],
                c=sub["valeur_num"], cmap="viridis",
                s=sub["f1"] * 200, alpha=0.7, edgecolors="k", linewidth=0.5,
            )
            fig.colorbar(scatter, ax=ax, label=param, shrink=0.7)
        else:
            # Catégoriel : couleur par catégorie
            for idx_v, val in enumerate(sub["valeur_testee"].unique()):
                s = sub[sub["valeur_testee"] == val]
                ax.scatter(
                    s["rappel"], s["precision"],
                    s=s["f1"] * 200, alpha=0.7, edgecolors="k", linewidth=0.5,
                    label=val[:20],
                )
            ax.legend(fontsize=6, loc="lower left")

        # Courbes iso-F1 pour référence visuelle
        for f1_val in [0.6, 0.7, 0.8, 0.9]:
            r_range = np.linspace(0.01, 1, 100)
            denom = 2 * r_range - f1_val
            # Éviter la division par zéro aux points singuliers
            with np.errstate(divide="ignore", invalid="ignore"):
                p_iso = np.where(np.abs(denom) > 1e-10,
                                 f1_val * r_range / denom, np.nan)
            mask = (p_iso > 0) & (p_iso <= 1)
            ax.plot(r_range[mask], p_iso[mask], "--", color="gray", alpha=0.3, lw=0.8)
            # Annoter la courbe iso
            idx_mid = np.argmin(np.abs(r_range[mask] - 0.5))
            if idx_mid < len(r_range[mask]):
                ax.text(r_range[mask][idx_mid], p_iso[mask][idx_mid] + 0.02,
                        f"F1={f1_val}", fontsize=6, color="gray", alpha=0.6)

        ax.set_xlabel("Rappel", fontsize=9)
        ax.set_ylabel("Précision", fontsize=9)
        ax.set_title(NOMS_PARAMS.get(param, param), fontsize=10, weight="bold")
        ax.set_xlim(0.3, 1.02)
        ax.set_ylim(0.55, 1.05)
        ax.grid(True, alpha=0.2)

    # Masquer les axes inutilisés
    for idx in range(len(params), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Compromis précision – rappel par paramètre", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    chemin = dossier / "precision_rappel_parametres.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# BOXPLOTS — DISTRIBUTION F1 PAR PARAMÈTRE
# =============================================================================

def tracer_boxplots_f1(df: pd.DataFrame, dossier: Path) -> Path:
    """Trace les boxplots du F1-score pour chaque paramètre varié.

    Permet de comparer rapidement la dispersion inter-clients de chaque
    paramètre. Le baseline est affiché comme référence horizontale.

    Args:
        df      (pd.DataFrame): Résultats complets.
        dossier (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG.

    Example:
        >>> tracer_boxplots_f1(df, Path("output/"))
    """
    params = [p for p in df["param_varie"].unique() if p != "baseline"]

    fig, ax = plt.subplots(figsize=(12, 5))

    donnees_box = []
    labels_box = []
    for param in params:
        sub = df[df["param_varie"] == param]
        donnees_box.append(sub["f1"].values)
        labels_box.append(NOMS_PARAMS.get(param, param))

    bp = ax.boxplot(
        donnees_box,
        tick_labels=labels_box,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
    )

    # Colorer les boîtes
    couleurs_box = ["#a6cee3", "#b2df8a", "#fdbf6f", "#fb9a99", "#cab2d6", "#ffff99"]
    for patch, couleur in zip(bp["boxes"], couleurs_box):
        patch.set_facecolor(couleur)
        patch.set_alpha(0.7)

    # Référence baseline
    f1_baseline_moy = df[df["param_varie"] == "baseline"]["f1"].mean()
    ax.axhline(f1_baseline_moy, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"Baseline moy. (F1={f1_baseline_moy:.3f})")

    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_title("Distribution du F1-score par paramètre varié", fontsize=12, weight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()

    chemin = dossier / "boxplots_f1_parametres.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# MATRICE DE CONFUSION EMPILÉE — PAR PARAMÈTRE
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
        ax = axes_flat[idx]
        sub = df[df["param_varie"] == param]

        # Ordonner les valeurs
        if param in PARAMS_NUMERIQUES:
            moy = sub.groupby("valeur_num")[["tp", "fp", "fn", "tn"]].mean().sort_index()
            x_labels = [f"{v:g}" for v in moy.index]
        else:
            moy = sub.groupby("valeur_testee")[["tp", "fp", "fn", "tn"]].mean()
            x_labels = [v[:20] for v in moy.index]

        x = np.arange(len(moy))
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

    # Légende commune
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
# CLASSEMENT — MEILLEURE VALEUR PAR PARAMÈTRE
# =============================================================================

def tracer_classement_meilleures(df: pd.DataFrame, dossier: Path) -> Path:
    """Trace un barplot horizontal du F1 moyen pour la meilleure valeur de chaque paramètre.

    Permet d'identifier rapidement quel paramètre offre le plus grand
    potentiel d'amélioration et quelle est sa valeur optimale.

    Args:
        df      (pd.DataFrame): Résultats complets.
        dossier (Path): Dossier de sortie.

    Returns:
        Path: Chemin du PNG.

    Example:
        >>> tracer_classement_meilleures(df, Path("output/"))
    """
    params = [p for p in df["param_varie"].unique() if p != "baseline"]

    resultats = []
    for param in params:
        sub = df[df["param_varie"] == param]
        moy_par_val = sub.groupby("valeur_testee")["f1"].mean()
        meilleur_val = moy_par_val.idxmax()
        meilleur_f1  = moy_par_val.max()
        pire_f1      = moy_par_val.min()
        resultats.append({
            "param":        NOMS_PARAMS.get(param, param),
            "meilleur_val": meilleur_val,
            "meilleur_f1":  meilleur_f1,
            "pire_f1":      pire_f1,
            "amplitude":    meilleur_f1 - pire_f1,
        })

    # Trier par amplitude décroissante (plus sensible en premier)
    resultats.sort(key=lambda r: r["amplitude"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(resultats))

    # Barres : amplitude (meilleur − pire)
    ax.barh(
        y_pos,
        [r["amplitude"] for r in resultats],
        left=[r["pire_f1"] for r in resultats],
        color="#4c72b0", alpha=0.7, height=0.5,
    )

    # Annotations
    for i, r in enumerate(resultats):
        ax.text(r["meilleur_f1"] + 0.005, i,
                f"F1={r['meilleur_f1']:.3f}  (val={r['meilleur_val'][:15]})",
                va="center", fontsize=8, weight="bold")
        ax.text(r["pire_f1"] - 0.005, i,
                f"{r['pire_f1']:.3f}", va="center", ha="right",
                fontsize=8, color="gray")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["param"] for r in resultats], fontsize=9)
    ax.set_xlabel("F1-score", fontsize=10)
    ax.set_title(
        "Sensibilité des paramètres — Amplitude du F1 (pire → meilleur)",
        fontsize=12, weight="bold",
    )
    ax.grid(True, alpha=0.2, axis="x")

    # Baseline référence
    f1_base = df[df["param_varie"] == "baseline"]["f1"].mean()
    ax.axvline(f1_base, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"Baseline (F1={f1_base:.3f})")
    ax.legend(fontsize=9)

    fig.tight_layout()
    chemin = dossier / "classement_sensibilite.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {chemin.name}")
    return chemin


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Point d'entrée : charge le CSV et génère tous les graphiques.

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

    # ── Graphiques par paramètre ──────────────────────────────────────────────
    print("\n[2] Graphiques de sensibilité par paramètre...")
    for param in df["param_varie"].unique():
        if param == "baseline":
            continue

        sub = df[df["param_varie"] == param]

        if param in PARAMS_CATEGORIELS:
            tracer_sensibilite_categoriel(sub, param, dossier)
        else:
            tracer_sensibilite_numerique(sub, param, dossier)

    # ── Graphiques synthétiques ───────────────────────────────────────────────
    print("\n[3] Graphiques synthétiques...")
    tracer_heatmap_f1(df, dossier)
    tracer_radar_baseline(df, dossier)
    tracer_precision_rappel(df, dossier)
    tracer_boxplots_f1(df, dossier)
    tracer_confusion_empilee(df, dossier)
    tracer_classement_meilleures(df, dossier)

    # ── Résumé ────────────────────────────────────────────────────────────────
    n_fichiers = len(list(dossier.glob("*.png")))
    print(f"\n{'=' * 70}")
    print(f"TERMINÉ — {n_fichiers} graphiques sauvegardés dans : {dossier}")
    print("=" * 70)


if __name__ == "__main__":
    main()
