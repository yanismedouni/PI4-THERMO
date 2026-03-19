# =============================================================================
# Module      : script_param.py
# Auteur      : Équipe THERMO — ELE8080
# Date        : 2025-01
# Description : Analyse de sensibilité OAT (One-At-a-Time) sur les paramètres
#               du modèle MIQP. Pour chaque paramètre, on fait varier sa valeur
#               sur une grille prédéfinie en maintenant tous les autres à leur
#               valeur par défaut. Chaque run produit une ligne dans le CSV.
#
# Usage :
#   python script_param.py --dataid 661 --date 2018-07-15
# =============================================================================

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from parametres import obtenir_parametres_defaut, afficher_parametres
from modele_opti import creer_modele_optimisation, resoudre_optimisation
from main_opti   import charger_journee, construire_donnees_modele
from graph       import tracer_desagregation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = Path(__file__).parent.parent / "output"

# =============================================================================
# GRILLE OAT
# Chaque entrée : nom du paramètre varié, valeurs à tester, fonction qui
# applique la valeur sur une copie du dict params.
# =============================================================================
GRILLE_OAT = [
    # ── baseline ──────────────────────────────────────────────────────────────
    {
        "param_varie":   "baseline",
        "valeur_testee": "defaut",
        "appliquer":     lambda p, v: None,
        "valeurs":       ["defaut"],
    },
    # ── lambda1 ───────────────────────────────────────────────────────────────
    {
        "param_varie":   "lambda1",
        "appliquer":     lambda p, v: p.update({"lambda1": v}),
        "valeurs":       [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    },
    # ── d_min ─────────────────────────────────────────────────────────────────
    {
        "param_varie":   "d_min",
        "appliquer":     lambda p, v: p.update({"d_min": int(v)}),
        "valeurs":       [1, 2, 3, 4, 5, 6],
    },
    # ── M ─────────────────────────────────────────────────────────────────────
    {
        "param_varie":   "M",
        "appliquer":     lambda p, v: p.update({"M": v}),
        "valeurs":       [100, 200, 500, 1000, 10000],
    },
    # ── x_DUTY_clim ───────────────────────────────────────────────────────────
    {
        "param_varie":   "x_DUTY_clim",
        "appliquer":     lambda p, v: p["duty_cycle"]["climatisation"].update({"x_DUTY": v}),
        "valeurs":       [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    },
    # ── x_DUTY_prime_clim ─────────────────────────────────────────────────────
    {
        "param_varie":   "x_DUTY_prime_clim",
        "appliquer":     lambda p, v: p["duty_cycle"]["climatisation"].update({"x_DUTY_prime": v}),
        "valeurs":       [1.5, 2.0, 3.0, 4.0, 6.0],
    },
    # ── niveaux_clim ──────────────────────────────────────────────────────────
    {
        "param_varie":   "niveaux_clim",
        "appliquer":     lambda p, v: p["niveaux_puissance"].update({"climatisation": v}),
        "valeurs":       [
            [0.0, 2.5],
            [0.0, 1.5, 2.5],
            [0.0, 0.5, 1.5, 2.5],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.5, 1.0, 2.0, 2.5],
        ],
    },
]


# =============================================================================
# VÉRITÉ TERRAIN
# =============================================================================

def extraire_verite_terrain(df_jour: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extrait le vecteur ON/OFF réel et la puissance réelle de la clim.

    Args:
        df_jour (pd.DataFrame): Journée chargée par charger_journee().

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - o_reel     : binaire ON/OFF shape (T,), 1 si P_clim > 0.05 kW
            - p_clim_reel: puissance agrégée réelle shape (T,)

    Raises:
        KeyError: Si 'P_clim_reel' est absent du DataFrame.

    Example:
        >>> o_reel, p_reel = extraire_verite_terrain(df_jour)
    """
    if "P_clim_reel" not in df_jour.columns:
        raise KeyError("Colonne 'P_clim_reel' absente — vérifier charger_journee().")

    p_clim_reel = df_jour["P_clim_reel"].fillna(0.0).values

    # Seuil 0.05 kW pour absorber le bruit de mesure résiduel des capteurs
    o_reel = (p_clim_reel > 0.05).astype(int)

    return o_reel, p_clim_reel


# =============================================================================
# MÉTRIQUES
# =============================================================================

def calculer_metriques(
    o_estime: np.ndarray,
    o_reel: np.ndarray,
    p_estimee: np.ndarray,
    p_reelle: np.ndarray,
) -> dict:
    """Calcule toutes les métriques de performance pour un run.

    Args:
        o_estime  (np.ndarray): ON/OFF estimé par le solveur, shape (T,).
        o_reel    (np.ndarray): ON/OFF réel (vérité terrain), shape (T,).
        p_estimee (np.ndarray): Puissance estimée (kW), shape (T,).
        p_reelle  (np.ndarray): Puissance réelle (kW), shape (T,).

    Returns:
        dict: tp, fp, fn, tn, precision, rappel, f1, rmse, mae.
            Retourne NaN pour precision/rappel/f1 si dénominateur nul.

    Example:
        >>> m = calculer_metriques(o_est, o_ref, p_est, p_ref)
        >>> print(m['f1'])
    """
    o_bin = np.round(o_estime).astype(int)
    o_ref = o_reel.astype(int)

    tp = int(np.sum((o_bin == 1) & (o_ref == 1)))
    fp = int(np.sum((o_bin == 1) & (o_ref == 0)))
    fn = int(np.sum((o_bin == 0) & (o_ref == 1)))
    tn = int(np.sum((o_bin == 0) & (o_ref == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rappel    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    if not (np.isnan(precision) or np.isnan(rappel)) and (precision + rappel) > 0:
        f1 = 2 * precision * rappel / (precision + rappel)
    else:
        f1 = float("nan")

    rmse = float(np.sqrt(np.mean((p_reelle - p_estimee) ** 2)))
    mae  = float(np.mean(np.abs(p_reelle - p_estimee)))

    return {
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "tn":        tn,
        "precision": round(precision, 4) if not np.isnan(precision) else float("nan"),
        "rappel":    round(rappel,    4) if not np.isnan(rappel)    else float("nan"),
        "f1":        round(f1,        4) if not np.isnan(f1)        else float("nan"),
        "rmse":      round(rmse, 4),
        "mae":       round(mae,  4),
    }


# =============================================================================
# EXPORT CSV
# =============================================================================

def ecrire_ligne_csv(chemin_csv: str, ligne: dict) -> None:
    """Ajoute une ligne de résultats dans le CSV (crée le fichier si absent).

    Args:
        chemin_csv (str): Chemin du CSV de sortie.
        ligne      (dict): Dictionnaire {colonne: valeur} pour une ligne.

    Returns:
        None

    Example:
        >>> ecrire_ligne_csv("output/resultats.csv", {"f1": 0.82, ...})
    """
    Path(chemin_csv).parent.mkdir(parents=True, exist_ok=True)
    df_ligne = pd.DataFrame([ligne])

    if Path(chemin_csv).exists():
        df_ligne.to_csv(chemin_csv, mode="a", header=False, index=False)
    else:
        df_ligne.to_csv(chemin_csv, mode="w", header=True, index=False)

    print(f"  Ligne ecrite dans : {chemin_csv}")


# =============================================================================
# UN SEUL RUN
# =============================================================================

def executer_run(
    df_jour: pd.DataFrame,
    donnees_base: dict,
    o_reel: np.ndarray,
    p_clim_reel: np.ndarray,
    dataid: int,
    date: str,
    params: dict,
    param_varie: str,
    valeur_testee,
    chemin_csv: str,
    run_id: int,
    sauvegarder_graph: bool,
) -> dict:
    """Exécute un run MIQP avec les params donnés et écrit une ligne CSV.

    Args:
        df_jour        (pd.DataFrame): Journée chargée.
        donnees_base   (dict): Données modèle (P_total, T_ext, heures).
        o_reel         (np.ndarray): Vérité terrain ON/OFF.
        p_clim_reel    (np.ndarray): Puissance réelle clim.
        dataid         (int): ID client.
        date           (str): Date YYYY-MM-DD.
        params         (dict): Paramètres du run (deep copy modifiée).
        param_varie    (str): Nom du paramètre varié pour le CSV.
        valeur_testee  (any): Valeur testée pour ce run.
        chemin_csv     (str): Chemin du CSV de sortie.
        run_id         (int): Numéro du run pour affichage.
        sauvegarder_graph (bool): Sauvegarder le graphique PNG.

    Returns:
        dict: Ligne de résultats.

    Example:
        >>> executer_run(df_jour, donnees, o_reel, p_reel, 661, "2018-07-15",
        ...              params, "lambda1", 2.0, "output/res.csv", 3, False)
    """
    # Reconstruire les données modèle avec les nouveaux params
    # (nécessaire car construire_donnees_modele injecte heures dans params)
    donnees = copy.deepcopy(donnees_base)
    donnees['heures'] = np.array(params.get('heures', donnees_base['heures']))

    modele    = creer_modele_optimisation(donnees, params)
    resultats = resoudre_optimisation(modele, verbose=False)

    # Métriques NaN si le solveur échoue
    if resultats is None:
        duty_clim = params["duty_cycle"]["climatisation"]
        ligne = {
            "run_id":            run_id,
            "param_varie":       param_varie,
            "valeur_testee":     str(valeur_testee),
            "dataid":            dataid,
            "date":              date,
            "lambda1":           params["lambda1"],
            "d_min":             params["d_min"],
            "M":                 params["M"],
            "x_DUTY_clim":       duty_clim["x_DUTY"],
            "x_DUTY_prime_clim": duty_clim["x_DUTY_prime"],
            "niveaux_clim":      str(params["niveaux_puissance"]["climatisation"]),
            "T_ext_MIN":         params["thermique"]["climatisation"]["T_ext_MIN"],
            "T_ext_MAX":         params["thermique"]["climatisation"]["T_ext_MAX"],
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "precision": float("nan"), "rappel": float("nan"),
            "f1": float("nan"), "rmse": float("nan"),
            "mae": float("nan"), "diff_moy": float("nan"),
            "statut_solveur":  "echec",
            "valeur_objectif": float("nan"),
            "gap_relatif":     float("nan"),
        }
        ecrire_ligne_csv(chemin_csv, ligne)
        return ligne

    res_clim  = resultats["appareils"]["climatisation"]
    o_estime  = res_clim["o"]
    p_estimee = res_clim["P_estimee"]

    metriques = calculer_metriques(o_estime, o_reel, p_estimee, p_clim_reel)
    diff_moy  = round(float(np.mean(p_clim_reel - p_estimee)), 4)
    gap       = resultats.get("gap_relatif", float("nan"))

    duty_clim = params["duty_cycle"]["climatisation"]
    ligne = {
        "run_id":            run_id,
        "param_varie":       param_varie,
        "valeur_testee":     str(valeur_testee),
        "dataid":            dataid,
        "date":              date,
        "lambda1":           params["lambda1"],
        "d_min":             params["d_min"],
        "M":                 params["M"],
        "x_DUTY_clim":       duty_clim["x_DUTY"],
        "x_DUTY_prime_clim": duty_clim["x_DUTY_prime"],
        "niveaux_clim":      str(params["niveaux_puissance"]["climatisation"]),
        "T_ext_MIN":         params["thermique"]["climatisation"]["T_ext_MIN"],
        "T_ext_MAX":         params["thermique"]["climatisation"]["T_ext_MAX"],
        "tp":                metriques["tp"],
        "fp":                metriques["fp"],
        "fn":                metriques["fn"],
        "tn":                metriques["tn"],
        "precision":         metriques["precision"],
        "rappel":            metriques["rappel"],
        "f1":                metriques["f1"],
        "rmse":              metriques["rmse"],
        "mae":               metriques["mae"],
        "diff_moy":          diff_moy,
        "statut_solveur":    resultats["statut"],
        "valeur_objectif":   round(resultats["valeur_optimale"], 4),
        "gap_relatif":       gap,
    }

    ecrire_ligne_csv(chemin_csv, ligne)

    # Graphique uniquement pour la baseline et sur demande
    if sauvegarder_graph:
        tracer_desagregation(donnees, resultats, params, df_jour,
                             dataid, f"{date}_run{run_id}",
                             afficher=False)

    return ligne


# =============================================================================
# BOUCLE OAT PRINCIPALE
# =============================================================================

def run_oat(dataid: int | None, date: str | None, chemin_csv: str) -> None:
    """Lance la boucle OAT complète et écrit une ligne par run dans le CSV.

    Pour chaque paramètre de GRILLE_OAT, itère sur ses valeurs en gardant
    tous les autres paramètres à leur valeur par défaut.

    Args:
        dataid     (int | None): ID client.
        date       (str | None): Date YYYY-MM-DD.
        chemin_csv (str): Chemin du CSV de sortie.

    Returns:
        None

    Example:
        >>> run_oat(661, "2018-07-15", "output/resultats_sensibilite.csv")
    """
    print("=" * 70)
    print("ANALYSE DE SENSIBILITÉ OAT — DÉSAGRÉGATION MIQP")
    print("=" * 70)

    # ── Chargement unique des données ─────────────────────────────────────────
    print("\n[1] Chargement de la journée...")
    df_jour, dataid_reel, date_reel = charger_journee(dataid=dataid, date=date)

    print("\n[2] Extraction de la vérité terrain...")
    o_reel, p_clim_reel = extraire_verite_terrain(df_jour)
    nb_on = int(o_reel.sum())
    print(f"  Pas ON  réel : {nb_on}/{len(o_reel)} ({100*nb_on/len(o_reel):.1f}%)")

    # ── Données modèle de base — communes à tous les runs ─────────────────────
    print("\n[3] Construction des données modèle de base...")
    params_defaut = obtenir_parametres_defaut()
    donnees_base  = construire_donnees_modele(df_jour, params_defaut)

    # Compter le nombre total de runs
    nb_runs = sum(len(cfg["valeurs"]) for cfg in GRILLE_OAT)
    print(f"\n  Total runs : {nb_runs}")
    print("=" * 70)

    # ── Suppression du CSV existant pour repartir propre ─────────────────────
    # On recrée le fichier à chaque lancement pour garantir la cohérence
    # des colonnes — pas d'accumulation de lignes avec des ordres différents.
    if Path(chemin_csv).exists():
        Path(chemin_csv).unlink()
        print(f"  CSV existant supprimé : {chemin_csv}")

    # ── Boucle OAT ────────────────────────────────────────────────────────────
    run_id = 0
    for cfg in GRILLE_OAT:
        param_varie = cfg["param_varie"]

        for valeur in cfg["valeurs"]:
            run_id += 1

            # Deep copy pour ne pas polluer le run suivant
            params = copy.deepcopy(params_defaut)
            cfg["appliquer"](params, valeur)

            # Réinjecter heures (effacé par deepcopy si déjà présent)
            params['heures'] = donnees_base['heures'].tolist()

            print(f"\n[{run_id:>2}/{nb_runs}] param={param_varie}  valeur={valeur}")

            executer_run(
                df_jour          = df_jour,
                donnees_base     = donnees_base,
                o_reel           = o_reel,
                p_clim_reel      = p_clim_reel,
                dataid           = dataid_reel,
                date             = date_reel,
                params           = params,
                param_varie      = param_varie,
                valeur_testee    = valeur,
                chemin_csv       = chemin_csv,
                run_id           = run_id,
                # Sauvegarder le graphique uniquement pour la baseline
                sauvegarder_graph = (param_varie == "baseline"),
            )

    print("\n" + "=" * 70)
    print(f"OAT TERMINÉ — {run_id} runs — CSV : {chemin_csv}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Point d'entrée CLI du script script_param.

    Returns:
        None

    Example:
        >>> # python script_param.py --dataid 661 --date 2018-07-15
    """
    parser = argparse.ArgumentParser(
        description="Analyse de sensibilité OAT — MIQP désagrégation TCL"
    )
    parser.add_argument("--dataid", type=int, default=None)
    parser.add_argument("--date",   type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "resultats_sensibilite.csv"),
                        help="Chemin du CSV de sortie")
    args = parser.parse_args()

    run_oat(
        dataid     = args.dataid,
        date       = args.date,
        chemin_csv = args.output,
    )


if __name__ == "__main__":
    main()