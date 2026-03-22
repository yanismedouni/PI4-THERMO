# =============================================================================
# Module      : script_multi.py
# Auteur      : Équipe THERMO — ELE8080
# Date        : 2025-01
# Description : Pipeline multi-clients multi-journées pour la désagrégation
#               MIQP. Pour les 25 premiers clients du fichier, exécute un run
#               par jour sur les 7 premiers jours valides de juin, juillet et
#               août. Chaque run produit une ligne dans le CSV de sortie.
#               Robuste aux crashes : NaN écrit et boucle continue.
#
# Usage :
#   python script_multi.py
#   python script_multi.py --nb_clients 25 --output output/resultats_multi.csv
# =============================================================================

import argparse
import copy
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from parametres import obtenir_parametres_defaut
from modele_opti import creer_modele_optimisation, resoudre_optimisation
from main_opti   import charger_journee, construire_donnees_modele

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = Path(__file__).parent.parent / "output"
OPTI_DIR    = Path(__file__).parent
ENERGY_FILE = OPTI_DIR / "austin_processed_energy_data.csv"

NB_CLIENTS_DEFAUT = 25
MOIS_ETE          = [6, 7, 8]   # juin, juillet, août
NB_JOURS_PAR_MOIS = 7
SEUIL_ON_KW       = 0.05        # seuil binarisation ON/OFF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SÉLECTION DES JOURNÉES VALIDES
# =============================================================================

def selectionner_journees(df: pd.DataFrame, dataid: int) -> list[str]:
    """Sélectionne les 7 premiers jours valides par mois d'été pour un client.

    Un jour est valide si la colonne 'grid' contient au moins 90 pas de temps
    non nuls — filtre les journées incomplètes ou corrompues.

    Args:
        df     (pd.DataFrame): DataFrame complet du fichier énergie.
        dataid (int): Identifiant du client.

    Returns:
        list[str]: Liste de dates au format 'YYYY-MM-DD', max 21 dates
            (7 par mois × 3 mois). Peut être plus courte si données insuffisantes.

    Example:
        >>> dates = selectionner_journees(df, 661)
        >>> len(dates)
        21
    """
    df_client = df[df['dataid'] == dataid].copy()
    df_client['date'] = df_client['local_15min'].dt.date
    df_client['mois'] = df_client['local_15min'].dt.month

    dates_selectionnees = []

    for mois in MOIS_ETE:
        df_mois = df_client[df_client['mois'] == mois]

        # Compter les pas de temps valides par jour
        jours_valides = (
            df_mois.groupby('date')['grid']
            .apply(lambda s: (s.fillna(0) > 0).sum())
            .reset_index()
        )
        jours_valides.columns = ['date', 'nb_pas_valides']

        # Garder les jours avec au moins 90 pas valides sur 96
        jours_ok = (
            jours_valides[jours_valides['nb_pas_valides'] >= 90]
            .sort_values('date')
            .head(NB_JOURS_PAR_MOIS)
        )

        dates_selectionnees.extend([str(d) for d in jours_ok['date'].tolist()])

    return dates_selectionnees


def selectionner_clients(df: pd.DataFrame, nb_clients: int) -> list[int]:
    """Retourne les N premiers clients disponibles dans le fichier.

    Args:
        df         (pd.DataFrame): DataFrame complet du fichier énergie.
        nb_clients (int): Nombre de clients à sélectionner.

    Returns:
        list[int]: Liste des dataid, max nb_clients éléments.

    Example:
        >>> clients = selectionner_clients(df, 25)
    """
    return df['dataid'].unique()[:nb_clients].tolist()


# =============================================================================
# VÉRITÉ TERRAIN
# =============================================================================

def extraire_verite_terrain(df_jour: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extrait le vecteur ON/OFF réel et la puissance réelle de la clim.

    Args:
        df_jour (pd.DataFrame): Journée chargée par charger_journee().

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - o_reel     : binaire ON/OFF shape (T,)
            - p_clim_reel: puissance agrégée réelle shape (T,)

    Raises:
        KeyError: Si 'P_clim_reel' est absent du DataFrame.

    Example:
        >>> o_reel, p_reel = extraire_verite_terrain(df_jour)
    """
    if "P_clim_reel" not in df_jour.columns:
        raise KeyError("Colonne 'P_clim_reel' absente.")

    p_clim_reel = df_jour["P_clim_reel"].fillna(0.0).values
    o_reel      = (p_clim_reel > SEUIL_ON_KW).astype(int)
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
    """Calcule F1, précision, rappel, RMSE, MAE et diff_moy pour un run.

    Args:
        o_estime  (np.ndarray): ON/OFF estimé, shape (T,).
        o_reel    (np.ndarray): ON/OFF réel, shape (T,).
        p_estimee (np.ndarray): Puissance estimée (kW), shape (T,).
        p_reelle  (np.ndarray): Puissance réelle (kW), shape (T,).

    Returns:
        dict: tp, fp, fn, tn, precision, rappel, f1, rmse, mae, diff_moy.

    Example:
        >>> m = calculer_metriques(o_est, o_ref, p_est, p_ref)
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

    rmse     = float(np.sqrt(np.mean((p_reelle - p_estimee) ** 2)))
    mae      = float(np.mean(np.abs(p_reelle - p_estimee)))
    diff_moy = float(np.mean(p_reelle - p_estimee))

    def r(v): return round(v, 4) if not math.isnan(v) else float("nan")

    return {
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "tn":        tn,
        "precision": r(precision),
        "rappel":    r(rappel),
        "f1":        r(f1),
        "rmse":      r(rmse),
        "mae":       r(mae),
        "diff_moy":  r(diff_moy),
    }


def metriques_nan() -> dict:
    """Retourne un dict de métriques NaN pour un run échoué.

    Returns:
        dict: Toutes les métriques à NaN ou 0.

    Example:
        >>> metriques_nan()['f1']
        nan
    """
    return {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "precision": float("nan"), "rappel":   float("nan"),
        "f1":        float("nan"), "rmse":     float("nan"),
        "mae":       float("nan"), "diff_moy": float("nan"),
    }


# =============================================================================
# EXPORT CSV
# =============================================================================

def ecrire_ligne_csv(chemin_csv: str, ligne: dict) -> None:
    """Ajoute une ligne dans le CSV, crée le fichier avec en-tête si absent.

    Args:
        chemin_csv (str): Chemin du CSV de sortie.
        ligne      (dict): Dictionnaire {colonne: valeur}.

    Returns:
        None

    Example:
        >>> ecrire_ligne_csv("output/resultats.csv", {"f1": 0.82})
    """
    Path(chemin_csv).parent.mkdir(parents=True, exist_ok=True)
    df_ligne = pd.DataFrame([ligne])

    if Path(chemin_csv).exists():
        df_ligne.to_csv(chemin_csv, mode="a", header=False, index=False)
    else:
        df_ligne.to_csv(chemin_csv, mode="w", header=True, index=False)


# =============================================================================
# CONSTRUCTION D'UNE LIGNE CSV
# =============================================================================

def construire_ligne(
    run_id: int,
    dataid: int,
    date: str,
    params: dict,
    metriques: dict,
    statut: str,
    valeur_objectif: float,
    gap: float,
) -> dict:
    """Construit le dict d'une ligne CSV à partir des résultats d'un run.

    Args:
        run_id          (int): Numéro du run.
        dataid          (int): ID client.
        date            (str): Date YYYY-MM-DD.
        params          (dict): Paramètres utilisés.
        metriques       (dict): Métriques calculées.
        statut          (str): Statut du solveur.
        valeur_objectif (float): Valeur de la fonction objectif.
        gap             (float): Gap relatif MIP.

    Returns:
        dict: Ligne complète prête pour le CSV.

    Example:
        >>> ligne = construire_ligne(1, 661, "2018-07-01", params, m, "optimal", 30.1, nan)
    """
    duty_clim = params["duty_cycle"]["climatisation"]
    return {
        "run_id":            run_id,
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
        "diff_moy":          metriques["diff_moy"],
        "statut_solveur":    statut,
        "valeur_objectif":   round(valeur_objectif, 4) if not math.isnan(valeur_objectif) else float("nan"),
        "gap_relatif":       gap,
    }


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_multi(nb_clients: int, chemin_csv: str) -> None:
    """Lance les 525 runs multi-clients multi-journées.

    Pour chacun des nb_clients premiers clients, sélectionne les 7 premiers
    jours valides de juin, juillet et août, puis résout le MIQP pour chaque
    journée avec les paramètres par défaut.

    Args:
        nb_clients (int): Nombre de clients à traiter.
        chemin_csv (str): Chemin du CSV de sortie.

    Returns:
        None

    Example:
        >>> run_multi(25, "output/resultats_multi.csv")
    """
    print("=" * 70)
    print("PIPELINE MULTI-CLIENTS — DÉSAGRÉGATION MIQP")
    print(f"  Clients     : {nb_clients} premiers")
    print(f"  Mois        : juin, juillet, août")
    print(f"  Jours/mois  : {NB_JOURS_PAR_MOIS} premiers jours valides")
    print(f"  Runs max    : {nb_clients * 3 * NB_JOURS_PAR_MOIS}")
    print("=" * 70)

    # ── Chargement du fichier complet une seule fois ───────────────────────
    print("\nChargement du fichier énergie...")
    df_all = pd.read_csv(ENERGY_FILE, parse_dates=['local_15min'])

    clients = selectionner_clients(df_all, nb_clients)
    print(f"Clients sélectionnés : {clients}")

    params_defaut = obtenir_parametres_defaut()

    # Supprimer le CSV existant pour repartir avec des colonnes propres
    if Path(chemin_csv).exists():
        try:
            Path(chemin_csv).unlink()
            print(f"CSV existant supprimé : {chemin_csv}")
        except PermissionError:
            print(f"ATTENTION : impossible de supprimer {chemin_csv} — fermez Excel.")
            sys.exit(1)

    run_id     = 0
    nb_echecs  = 0
    nb_total   = 0

    for dataid in clients:
        dates = selectionner_journees(df_all, dataid)

        if not dates:
            logger.warning(f"Client {dataid} : aucune journée valide trouvée.")
            continue

        nb_total += len(dates)
        logger.info(f"Client {dataid} : {len(dates)} journées — {dates}")

        for date in dates:
            run_id += 1
            print(f"\n[{run_id:>3}] dataid={dataid}  date={date}")

            try:
                # Charger la journée
                df_jour, _, _ = charger_journee(dataid=dataid, date=date)
                o_reel, p_clim_reel = extraire_verite_terrain(df_jour)

                # Paramètres — deep copy pour isolation entre runs
                params = copy.deepcopy(params_defaut)
                donnees = construire_donnees_modele(df_jour, params)

                # Résolution
                modele    = creer_modele_optimisation(donnees, params)
                resultats = resoudre_optimisation(modele, verbose=False)

                if resultats is None:
                    raise RuntimeError("Solveur a retourné None")

                res_clim  = resultats["appareils"]["climatisation"]
                metriques = calculer_metriques(
                    res_clim["o"], o_reel,
                    res_clim["P_estimee"], p_clim_reel
                )
                statut          = resultats["statut"]
                valeur_objectif = resultats["valeur_optimale"]
                gap             = resultats.get("gap_relatif", float("nan"))

            except Exception as e:
                logger.error(f"Run {run_id} ({dataid}/{date}) ÉCHEC : {e}")
                nb_echecs  += 1
                metriques   = metriques_nan()
                statut          = f"echec"
                valeur_objectif = float("nan")
                gap             = float("nan")

            ligne = construire_ligne(
                run_id, dataid, date, params_defaut,
                metriques, statut, valeur_objectif, gap
            )
            ecrire_ligne_csv(chemin_csv, ligne)

            f1_str = f"{metriques['f1']:.4f}" if not math.isnan(metriques['f1']) else "NaN"
            print(f"       F1={f1_str}  RMSE={metriques['rmse']}  statut={statut}")

    print("\n" + "=" * 70)
    print(f"TERMINÉ — {run_id} runs  |  {nb_echecs} échecs  |  CSV : {chemin_csv}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Point d'entrée CLI du pipeline multi-clients.

    Returns:
        None

    Example:
        >>> # python script_multi.py --nb_clients 25
    """
    parser = argparse.ArgumentParser(
        description="Pipeline multi-clients MIQP — 525 runs été"
    )
    parser.add_argument("--nb_clients", type=int, default=NB_CLIENTS_DEFAUT,
                        help="Nombre de clients à traiter (défaut: 25)")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "resultats_multi.csv"),
                        help="Chemin du CSV de sortie")
    args = parser.parse_args()

    run_multi(
        nb_clients = args.nb_clients,
        chemin_csv = args.output,
    )


if __name__ == "__main__":
    main()
