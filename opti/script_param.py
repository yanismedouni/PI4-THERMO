# =============================================================================
# Module      : run_miqp.py
# Auteur      : Équipe THERMO — ELE8080
# Date        : 2025-01
# Description : Exécute UN seul run de désagrégation MIQP et calcule les
#               métriques de performance (F1, précision, rappel, RMSE, MAE).
#               Écrit une ligne dans un CSV de résultats.
#               Point de départ avant la boucle multi-runs.
#
# Usage :
#   python run_un_scenario.py --dataid 661 --date 2018-07-15
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from parametres import obtenir_parametres_defaut, afficher_parametres
from modele_opti import creer_modele_optimisation, resoudre_optimisation
from main_opti   import charger_journee, construire_donnees_modele

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = Path(__file__).parent.parent / "output"


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
    # Binarisation robuste — MOSEK retourne parfois 0.9999 au lieu de 1
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

    Si le fichier n'existe pas encore, écrit l'en-tête puis la ligne.
    Si le fichier existe déjà, appende sans réécrire l'en-tête — pratique
    pour accumuler les runs un par un.

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

    # Mode append sans en-tête si le fichier existe déjà
    if Path(chemin_csv).exists():
        df_ligne.to_csv(chemin_csv, mode="a", header=False, index=False)
    else:
        df_ligne.to_csv(chemin_csv, mode="w", header=True,  index=False)

    print(f"\n  Ligne écrite dans : {chemin_csv}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_scenario(dataid: int | None, date: str | None, chemin_csv: str) -> dict:
    """Exécute un run complet et retourne la ligne de résultats.

    Pipeline :
      1. Charge la journée et extrait la vérité terrain
      2. Charge les paramètres par défaut
      3. Construit les données modèle
      4. Crée et résout le modèle MIQP
      5. Calcule les métriques
      6. Écrit une ligne dans le CSV

    Args:
        dataid     (int | None): ID client. None = premier client disponible.
        date       (str | None): Date YYYY-MM-DD. None = première date.
        chemin_csv (str): Chemin du CSV de sortie.

    Returns:
        dict: Ligne complète de résultats (paramètres + métriques).

    Raises:
        SystemExit: Si le chargement des données échoue.

    Example:
        >>> ligne = run_scenario(661, "2018-07-15", "output/resultats.csv")
    """
    print("=" * 70)
    print("RUN UNIQUE — DÉSAGRÉGATION MIQP + MÉTRIQUES")
    print("=" * 70)

    # ── 1. Données ────────────────────────────────────────────────────────────
    print("\n[1] Chargement de la journée...")
    df_jour, dataid_reel, date_reel = charger_journee(dataid=dataid, date=date)

    print("\n[2] Extraction de la vérité terrain...")
    o_reel, p_clim_reel = extraire_verite_terrain(df_jour)
    nb_on = int(o_reel.sum())
    print(f"  Pas ON  réel : {nb_on}/{len(o_reel)} ({100*nb_on/len(o_reel):.1f}%)")
    print(f"  Pas OFF réel : {len(o_reel)-nb_on}/{len(o_reel)}")

    # ── 2. Paramètres ─────────────────────────────────────────────────────────
    print("\n[3] Paramètres...")
    params = obtenir_parametres_defaut()
    afficher_parametres(params)

    # ── 3. Données modèle ─────────────────────────────────────────────────────
    print("\n[4] Construction des données modèle...")
    # construire_donnees_modele injecte params['heures'] en interne — on passe
    # une copie pour ne pas modifier params entre deux appels successifs
    donnees = construire_donnees_modele(df_jour, params)

    # ── 4. Modèle + résolution ────────────────────────────────────────────────
    print("\n[5] Création du modèle...")
    modele = creer_modele_optimisation(donnees, params)

    print("\n[6] Résolution MOSEK...")
    resultats = resoudre_optimisation(modele, verbose=False)

    if resultats is None:
        print("\n  ÉCHEC : solveur n'a pas trouvé de solution.")
        sys.exit(1)

    # ── 5. Métriques ──────────────────────────────────────────────────────────
    print("\n[7] Calcul des métriques...")

    res_clim  = resultats["appareils"]["climatisation"]
    o_estime  = res_clim["o"]          # vecteur float retourné par MOSEK
    p_estimee = res_clim["P_estimee"]  # x @ niveaux, shape (T,)

    print(f"\n  Diagnostic ON/OFF estimé :")
    print(f"    Valeurs uniques de o_estime : {np.unique(np.round(o_estime, 2))}")
    print(f"    Somme o_estime (avant round) : {o_estime.sum():.2f}")
    print(f"    Pas ON estimés (après round) : {int(np.round(o_estime).sum())}")
    print(f"    Pas ON réels                 : {nb_on}")

    metriques = calculer_metriques(o_estime, o_reel, p_estimee, p_clim_reel)

    print(f"\n  Résultats :")
    print(f"    TP={metriques['tp']}  FP={metriques['fp']}  "
          f"FN={metriques['fn']}  TN={metriques['tn']}")
    print(f"    Précision : {metriques['precision']}")
    print(f"    Rappel    : {metriques['rappel']}")
    print(f"    F1        : {metriques['f1']}")
    print(f"    RMSE      : {metriques['rmse']} kW")
    print(f"    MAE       : {metriques['mae']} kW")

    # ── 6. Construction de la ligne CSV ───────────────────────────────────────
    duty_clim = params["duty_cycle"]["climatisation"]
    ligne = {
        # Identifiants
        "dataid":            dataid_reel,
        "date":              date_reel,
        # Paramètres du run
        "lambda1":           params["lambda1"],
        "d_min":             params["d_min"],
        "x_DUTY_clim":       duty_clim["x_DUTY"],
        "x_DUTY_prime_clim": duty_clim["x_DUTY_prime"],
        "niveaux_clim":      str(params["niveaux_puissance"]["climatisation"]),
        "M":                 params["M"],           # ← ajouter cette ligne
        "T_ext_MIN":         params["thermique"]["climatisation"]["T_ext_MIN"],
        # Métriques
        "tp":                metriques["tp"],
        "fp":                metriques["fp"],
        "fn":                metriques["fn"],
        "tn":                metriques["tn"],
        "precision":         metriques["precision"],
        "rappel":            metriques["rappel"],
        "f1":                metriques["f1"],
        "rmse":              metriques["rmse"],
        "mae":               metriques["mae"],
        # Info solveur
        "statut_solveur":    resultats["statut"],
        "valeur_objectif":   round(resultats["valeur_optimale"], 4),
    }

    # ── 7. Écriture CSV ───────────────────────────────────────────────────────
    print("\n[8] Écriture dans le CSV...")
    ecrire_ligne_csv(chemin_csv, ligne)

    return ligne


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Point d'entrée CLI du script run_un_scenario.

    Returns:
        None

    Example:
        >>> # python run_un_scenario.py --dataid 661 --date 2018-07-15
    """
    parser = argparse.ArgumentParser(
        description="Run unique MIQP + métriques F1/précision/rappel"
    )
    parser.add_argument("--dataid", type=int,  default=None)
    parser.add_argument("--date",   type=str,  default=None,
                        help="YYYY-MM-DD")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "resultats_sensibilite.csv"),
                        help="Chemin du CSV de sortie")
    args = parser.parse_args()

    run_scenario(
        dataid     = args.dataid,
        date       = args.date,
        chemin_csv = args.output,
    )


if __name__ == "__main__":
    main()
