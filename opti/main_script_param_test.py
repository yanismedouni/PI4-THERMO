# =============================================================================
# Module      : main_sensibilite.py
# Auteur      : Équipe THERMO — ELE8080
# Date        : 2025-01
# Description : Point d'entrée pour l'analyse de sensibilité OAT (One-At-a-Time).
#               Fait varier chaque paramètre sur une grille prédéfinie,
#               résout le MIQP pour chaque configuration et exporte
#               les métriques (F1, précision, rappel, RMSE) dans un CSV.
#
# Usage :
#   python src/main_sensibilite.py
#   python src/main_sensibilite.py --dataid 661 --date 2018-07-15
#   python src/main_sensibilite.py --dataid 661 --date 2018-07-15 --output resultats/sensi.csv
#   python src/main_sensibilite.py --verbose_solveur
# =============================================================================

import argparse
import logging
import sys
from pathlib import Path

# Ajout du répertoire courant au path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent))

from parametres import obtenir_parametres_defaut, afficher_parametres
from modele_opti import creer_modele_optimisation, resoudre_optimisation
from main_opti import charger_journee, construire_donnees_modele

from analyse_sensibilite import (
    construire_grille_experiences,
    executer_experiences,
    extraire_verite_terrain,
    sauvegarder_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CHEMINS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS DU MAIN
# ─────────────────────────────────────────────────────────────────────────────

def construire_chemin_sortie(dataid: int, date: str, chemin_custom: str | None) -> str:
    """Détermine le chemin du CSV de sortie.

    Si --output est fourni, on l'utilise directement.
    Sinon, on génère un nom automatique dans output/ basé sur dataid et date.

    Args:
        dataid        (int): Identifiant du client.
        date          (str): Date au format 'YYYY-MM-DD'.
        chemin_custom (str | None): Valeur de l'argument --output, ou None.

    Returns:
        str: Chemin absolu du fichier CSV à créer.

    Example:
        >>> construire_chemin_sortie(661, "2018-07-15", None)
        '.../output/sensibilite_661_2018-07-15.csv'
    """
    if chemin_custom:
        return chemin_custom

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUTPUT_DIR / f"sensibilite_{dataid}_{date}.csv")


def afficher_resume_grille(experiences: list[dict]) -> None:
    """Affiche un résumé de la grille d'expériences avant l'exécution.

    Utile pour vérifier la configuration sans lancer le solveur.
    Groupe les expériences par paramètre varié et liste les valeurs testées.

    Args:
        experiences (list[dict]): Sortie de construire_grille_experiences().

    Returns:
        None

    Example:
        >>> afficher_resume_grille(experiences)
    """
    print("\n" + "=" * 70)
    print("GRILLE D'EXPÉRIENCES OAT")
    print("=" * 70)

    # Regroupement par paramètre pour un affichage compact
    groupes: dict[str, list] = {}
    for exp in experiences:
        p = exp["param_varie"]
        groupes.setdefault(p, []).append(exp["valeur_testee"])

    for param, valeurs in groupes.items():
        label = "  [DÉFAUT]" if param == "DEFAUT" else f"  {param}"
        print(f"{label:30s} : {valeurs}")

    print(f"\n  Total runs : {len(experiences)}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal du script d'analyse de sensibilité.

    Orchestre le pipeline complet :
      1. Charge la journée et construit la vérité terrain
      2. Construit la grille OAT d'expériences
      3. Exécute tous les runs MIQP
      4. Exporte les résultats en CSV

    Returns:
        None

    Raises:
        SystemExit: Si les données ne peuvent pas être chargées.

    Example:
        >>> main()
    """
    parser = argparse.ArgumentParser(
        description="Analyse de sensibilité OAT — Désagrégation TCL THERMO"
    )
    parser.add_argument(
        "--dataid", type=int, default=None,
        help="Identifiant client Pecan Street. Défaut : premier client du fichier."
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date à analyser, format YYYY-MM-DD. Défaut : première date disponible."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Chemin du CSV de sortie. Défaut : output/sensibilite_<dataid>_<date>.csv"
    )
    parser.add_argument(
        "--verbose_solveur", action="store_true", default=False,
        help="Afficher la sortie MOSEK pour chaque run (verbeux)."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ANALYSE DE SENSIBILITÉ OAT — PIPELINE THERMO")
    print("=" * 70)

    # ── Étape 1 : Chargement des données ──────────────────────────────────────
    print("\n[1] Chargement de la journée...")
    try:
        df_jour, dataid, date = charger_journee(
            dataid=args.dataid, date=args.date
        )
    except Exception as err:
        logger.error(f"Impossible de charger les données : {err}")
        sys.exit(1)

    # ── Étape 2 : Vérité terrain (fixe pour tous les runs) ────────────────────
    print("\n[2] Extraction de la vérité terrain...")
    o_reel, p_clim_reel = extraire_verite_terrain(df_jour)
    nb_on  = int(o_reel.sum())
    nb_off = len(o_reel) - nb_on
    print(f"  Pas ON  (réel) : {nb_on}/{len(o_reel)} ({100*nb_on/len(o_reel):.1f}%)")
    print(f"  Pas OFF (réel) : {nb_off}/{len(o_reel)} ({100*nb_off/len(o_reel):.1f}%)")

    # ── Étape 3 : Paramètres de référence et données modèle ───────────────────
    print("\n[3] Paramètres de référence...")
    params_defaut = obtenir_parametres_defaut()
    afficher_parametres(params_defaut)

    # Les données modèle (P_total, T_ext, heures) sont communes à tous les runs.
    # Seuls les paramètres changent entre les runs — pas besoin de recharger.
    print("\n[4] Construction des données modèle (communes à tous les runs)...")
    donnees_modele = construire_donnees_modele(df_jour, params_defaut)

    # ── Étape 4 : Grille d'expériences ────────────────────────────────────────
    print("\n[5] Construction de la grille OAT...")
    experiences = construire_grille_experiences(params_defaut)
    afficher_resume_grille(experiences)

    # ── Étape 5 : Exécution des runs ──────────────────────────────────────────
    print("\n[6] Exécution des expériences...")
    df_resultats = executer_experiences(
        experiences      = experiences,
        df_jour          = df_jour,
        donnees_modele   = donnees_modele,
        o_reel           = o_reel,
        p_clim_reel      = p_clim_reel,
        creer_modele_fn  = creer_modele_optimisation,
        resoudre_fn      = resoudre_optimisation,
        verbose_solveur  = args.verbose_solveur,
    )

    # ── Étape 6 : Export CSV ──────────────────────────────────────────────────
    print("\n[7] Export des résultats...")
    chemin_csv = construire_chemin_sortie(dataid, date, args.output)
    sauvegarder_csv(df_resultats, chemin_csv)

    # ── Résumé terminal ───────────────────────────────────────────────────────
    _afficher_resume_final(df_resultats)


def _afficher_resume_final(df: "pd.DataFrame") -> None:
    """Affiche un tableau récapitulatif trié par F1 décroissant.

    Permet de repérer immédiatement les configurations les plus performantes
    sans ouvrir le CSV.

    Args:
        df (pd.DataFrame): Sortie de executer_experiences().

    Returns:
        None

    Example:
        >>> _afficher_resume_final(df_resultats)
    """
    import pandas as pd

    print("\n" + "=" * 70)
    print("RÉSUMÉ — TOP 10 CONFIGURATIONS PAR F1 DÉCROISSANT")
    print("=" * 70)

    # On exclut les runs en échec pour le résumé
    df_ok = df[df["statut_solveur"].isin(["optimal", "optimal_inaccurate"])].copy()

    if df_ok.empty:
        print("  Aucun run optimal — vérifier MOSEK et les paramètres.")
        return

    cols_affichage = [
        "run_id", "param_varie", "valeur_testee",
        "f1", "precision", "rappel", "rmse", "statut_solveur"
    ]
    df_top = (
        df_ok[cols_affichage]
        .sort_values("f1", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    print(df_top.to_string(index=False))

    # Meilleure configuration
    meilleur = df_ok.loc[df_ok["f1"].idxmax()]
    print(f"\n  ➤ Meilleure config : param={meilleur['param_varie']}"
          f"  valeur={meilleur['valeur_testee']}"
          f"  F1={meilleur['f1']}")

    # Nombre de runs échoués
    n_echecs = len(df) - len(df_ok)
    if n_echecs > 0:
        print(f"\n  ⚠ Runs en échec (NaN) : {n_echecs}/{len(df)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
