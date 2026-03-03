# =============================================================================
# Module      : analyse_sensibilite.py
# Auteur      : Équipe THERMO — ELE8080
# Date        : 2025-01
# Description : Analyse de sensibilité OAT (One-At-a-Time) sur les paramètres
#               du modèle MIQP de désagrégation TCL.
#               Pour chaque paramètre, on fait varier sa valeur sur une grille
#               prédéfinie en maintenant tous les autres à leur valeur par
#               défaut, puis on mesure F1, précision, rappel et RMSE.
# =============================================================================

import copy
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# GRILLE DE VARIATION DES PARAMÈTRES
# =============================================================================

# Chaque entrée décrit comment modifier le dict `params` pour un paramètre donné.
# La clé est un identifiant lisible (nom du paramètre dans le CSV).
# La valeur est un dict avec :
#   'valeurs'  : list  — valeurs à tester
#   'appliquer': callable(params, valeur) -> None  — mutation in-place du dict params
#
# On sépare x_DUTY et x_DUTY_prime pour mesurer leur effet indépendamment.

GRILLE_PARAMETRES: dict[str, dict] = {
    "lambda1": {
        "valeurs": [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        "appliquer": lambda p, v: p.update({"lambda1": v}),
    },
    "d_min": {
        "valeurs": [1, 2, 3, 4, 5, 6],
        "appliquer": lambda p, v: p.update({"d_min": int(v)}),
    },
    "x_DUTY_clim": {
        "valeurs": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        # x_DUTY contrôle ON >= |T_POINTE| / x_DUTY — plus grand = moins contraint
        "appliquer": lambda p, v: p["duty_cycle"]["climatisation"].update({"x_DUTY": v}),
    },
    "x_DUTY_prime_clim": {
        "valeurs": [1.5, 2.0, 3.0, 4.0, 6.0],
        # x_DUTY_prime contrôle ON <= |T_HORS_POINTE| / x_DUTY_prime
        "appliquer": lambda p, v: p["duty_cycle"]["climatisation"].update({"x_DUTY_prime": v}),
    },
    "niveaux_clim_2": {
        # On garde toujours [0.0, niv_bas, niv_haut] pour conserver 2 niveaux ON
        "valeurs": [
            [0.0, 0.5, 1.5],
            [0.0, 0.5, 2.0],
            [0.0, 0.5, 2.5],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.5],
            [0.0, 1.5, 2.5],
            [0.0, 1.5, 3.0],
            [0.0, 2.0, 3.0],
        ],
        "appliquer": lambda p, v: p["niveaux_puissance"].update({"climatisation": v}),
    },
}


# =============================================================================
# CONSTRUCTION DE LA GRILLE D'EXPÉRIENCES
# =============================================================================

def construire_grille_experiences(params_defaut: dict) -> list[dict]:
    """Génère la liste complète des expériences OAT à exécuter.

    Pour chaque paramètre de GRILLE_PARAMETRES, on produit autant d'expériences
    qu'il y a de valeurs candidates. Tous les autres paramètres restent à leur
    valeur par défaut (deep copy pour éviter les effets de bord).

    Args:
        params_defaut (dict): Paramètres de référence retournés par
            obtenir_parametres_defaut().

    Returns:
        list[dict]: Liste d'expériences, chaque élément contenant :
            - 'params'        : dict complet prêt pour creer_modele_optimisation
            - 'param_varie'   : str, nom du paramètre modifié
            - 'valeur_testee' : Any, valeur utilisée pour ce run
            - 'est_defaut'    : bool, True si c'est la ligne de référence

    Example:
        >>> exps = construire_grille_experiences(obtenir_parametres_defaut())
        >>> len(exps)  # dépend de la grille
        37
    """
    experiences = []

    # Ligne de référence : paramètres par défaut (pour avoir une baseline dans le CSV)
    experiences.append({
        "params":        copy.deepcopy(params_defaut),
        "param_varie":   "DEFAUT",
        "valeur_testee": "defaut",
        "est_defaut":    True,
    })

    for nom_param, config in GRILLE_PARAMETRES.items():
        for valeur in config["valeurs"]:
            params_copie = copy.deepcopy(params_defaut)
            config["appliquer"](params_copie, valeur)

            experiences.append({
                "params":        params_copie,
                "param_varie":   nom_param,
                "valeur_testee": valeur,
                "est_defaut":    False,
            })

    logger.info(f"Grille construite : {len(experiences)} expériences.")
    return experiences


# =============================================================================
# CALCUL DES MÉTRIQUES DE PERFORMANCE
# =============================================================================

def calculer_metriques(
    o_estime: np.ndarray,
    o_reel: np.ndarray,
    p_estimee: np.ndarray,
    p_reelle: np.ndarray,
) -> dict[str, float]:
    """Calcule F1, précision, rappel, RMSE et MAE pour un run donné.

    La comparaison ON/OFF se fait au niveau de chaque pas de temps t.
    o_estime est arrondi à {0,1} pour tenir compte des imprécisions du solveur.

    Args:
        o_estime  (np.ndarray): Vecteur ON/OFF estimé, shape (T,), valeurs [0,1].
        o_reel    (np.ndarray): Vecteur ON/OFF réel (vérité terrain), shape (T,).
        p_estimee (np.ndarray): Puissance estimée (kW), shape (T,).
        p_reelle  (np.ndarray): Puissance réelle (kW), shape (T,).

    Returns:
        dict[str, float]: Clés : tp, fp, fn, tn, precision, rappel, f1, rmse, mae.
            Vaut NaN pour precision/rappel/f1 si dénominateur nul.

    Example:
        >>> calculer_metriques(np.array([1,0,1]), np.array([1,1,0]),
        ...                    np.array([2.0,0,1.5]), np.array([2.0,1.5,0]))
    """
    # Binarisation robuste — le solveur peut retourner 0.999 au lieu de 1
    o_bin = (np.round(o_estime).astype(int))
    o_ref = o_reel.astype(int)

    tp = int(np.sum((o_bin == 1) & (o_ref == 1)))
    fp = int(np.sum((o_bin == 1) & (o_ref == 0)))
    fn = int(np.sum((o_bin == 0) & (o_ref == 1)))
    tn = int(np.sum((o_bin == 0) & (o_ref == 0)))

    # Précision et rappel — NaN si division par zéro (pas de ON estimé / réel)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rappel    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    # F1 harmonique — NaN si les deux termes sont NaN ou nuls
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
        "rappel":    round(rappel, 4)    if not np.isnan(rappel)    else float("nan"),
        "f1":        round(f1, 4)        if not np.isnan(f1)        else float("nan"),
        "rmse":      round(rmse, 4),
        "mae":       round(mae, 4),
    }


def extraire_verite_terrain(df_jour: "pd.DataFrame") -> tuple[np.ndarray, np.ndarray]:
    """Construit les vecteurs ON/OFF et puissance réels à partir des colonnes sources.

    Les colonnes réelles (air1, air2, air3) ont déjà été agrégées en
    'P_clim_reel' par charger_journee(). On seuille à P > 0.05 kW pour
    distinguer ON de OFF (bruit résiduel possible dans les données).

    Args:
        df_jour (pd.DataFrame): Journée chargée par charger_journee().

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - o_reel    : vecteur binaire ON/OFF, shape (T,)
            - p_clim_reel : puissance agrégée réelle, shape (T,)

    Raises:
        KeyError: Si la colonne 'P_clim_reel' est absente du DataFrame.

    Example:
        >>> o_reel, p_reel = extraire_verite_terrain(df_jour)
    """
    if "P_clim_reel" not in df_jour.columns:
        raise KeyError(
            "Colonne 'P_clim_reel' absente. "
            "Vérifier que charger_journee() a bien été appelé."
        )

    p_clim_reel = df_jour["P_clim_reel"].fillna(0.0).values

    # Seuil de 0.05 kW pour éliminer le bruit de mesure résiduel
    seuil_on = 0.05
    o_reel   = (p_clim_reel > seuil_on).astype(int)

    return o_reel, p_clim_reel


# =============================================================================
# BOUCLE PRINCIPALE D'EXPÉRIENCES
# =============================================================================

def executer_experiences(
    experiences: list[dict],
    df_jour: "pd.DataFrame",
    donnees_modele: dict,
    o_reel: np.ndarray,
    p_clim_reel: np.ndarray,
    creer_modele_fn: callable,
    resoudre_fn: callable,
    verbose_solveur: bool = False,
) -> pd.DataFrame:
    """Exécute toutes les expériences OAT et retourne un DataFrame de résultats.

    Pour chaque expérience, reconstruit le modèle avec les paramètres modifiés,
    résout le MIQP, calcule les métriques et stocke le tout dans une ligne CSV.
    Si le solveur échoue ou retourne un statut non-optimal, les colonnes de
    métriques sont remplies de NaN et le run est marqué 'echec'.

    Args:
        experiences      (list[dict]): Sortie de construire_grille_experiences().
        df_jour          (pd.DataFrame): Journée chargée.
        donnees_modele   (dict): Sortie de construire_donnees_modele() — fixe pour
            tous les runs (seuls les params changent).
        o_reel           (np.ndarray): Vecteur ON/OFF vérité terrain, shape (T,).
        p_clim_reel      (np.ndarray): Puissance réelle clim, shape (T,).
        creer_modele_fn  (callable): Référence à creer_modele_optimisation().
        resoudre_fn      (callable): Référence à resoudre_optimisation().
        verbose_solveur  (bool): Passer verbose=True au solveur MOSEK si True.

    Returns:
        pd.DataFrame: Une ligne par expérience. Colonnes :
            param_varie, valeur_testee, lambda1, d_min, x_DUTY_clim,
            x_DUTY_prime_clim, niveaux_clim, f1, precision, rappel,
            tp, fp, fn, tn, rmse, mae, statut_solveur, valeur_objectif.

    Example:
        >>> df_resultats = executer_experiences(exps, df_jour, donnees,
        ...                                     o_reel, p_reel, creer, resoudre)
    """
    lignes = []
    n_total = len(experiences)

    for idx, exp in enumerate(experiences, start=1):
        params        = exp["params"]
        param_varie   = exp["param_varie"]
        valeur_testee = exp["valeur_testee"]

        print(f"\n[{idx:>3}/{n_total}] param={param_varie}  valeur={valeur_testee}")

        # --- Extraction des valeurs scalaires pour le CSV ---
        meta = _extraire_meta_params(params)

        # --- Résolution ---
        statut_solveur  = "echec"
        valeur_objectif = float("nan")
        metriques       = _metriques_nan()

        try:
            # On reconstruit le modèle à chaque run car les params peuvent
            # modifier la structure des contraintes (d_min, duty-cycle, niveaux)
            modele    = creer_modele_fn(donnees_modele, params)
            resultats = resoudre_fn(modele, verbose=verbose_solveur)

            if resultats is not None:
                statut_solveur  = resultats["statut"]
                valeur_objectif = resultats["valeur_optimale"]

                # Extraction des résultats climatisation uniquement
                if "climatisation" in resultats["appareils"]:
                    res_clim  = resultats["appareils"]["climatisation"]
                    o_estime  = res_clim["o"]
                    p_estimee = res_clim["P_estimee"]
                    metriques = calculer_metriques(
                        o_estime, o_reel, p_estimee, p_clim_reel
                    )
                else:
                    logger.warning(f"Run {idx}: 'climatisation' absent des résultats.")
            else:
                logger.warning(f"Run {idx}: solveur a retourné None — marqué NaN.")

        except Exception as err:
            # On capture toute exception pour ne pas interrompre la boucle
            logger.error(f"Run {idx} ({param_varie}={valeur_testee}) : {err}")
            statut_solveur = f"exception: {type(err).__name__}"

        ligne = {
            "run_id":           idx,
            "param_varie":      param_varie,
            "valeur_testee":    str(valeur_testee),   # str pour compatibilité CSV (listes)
            **meta,
            "f1":               metriques["f1"],
            "precision":        metriques["precision"],
            "rappel":           metriques["rappel"],
            "tp":               metriques["tp"],
            "fp":               metriques["fp"],
            "fn":               metriques["fn"],
            "tn":               metriques["tn"],
            "rmse":             metriques["rmse"],
            "mae":              metriques["mae"],
            "statut_solveur":   statut_solveur,
            "valeur_objectif":  round(valeur_objectif, 6) if not np.isnan(valeur_objectif) else float("nan"),
        }
        lignes.append(ligne)

        print(
            f"         F1={metriques['f1']}  "
            f"P={metriques['precision']}  "
            f"R={metriques['rappel']}  "
            f"RMSE={metriques['rmse']}  "
            f"statut={statut_solveur}"
        )

    return pd.DataFrame(lignes)


# =============================================================================
# FONCTIONS UTILITAIRES INTERNES
# =============================================================================

def _extraire_meta_params(params: dict) -> dict[str, Any]:
    """Extrait les valeurs scalaires de params pour le CSV.

    Chaque paramètre variable devient une colonne dans le CSV afin de
    permettre des filtres et tris directs sans parser la colonne valeur_testee.

    Args:
        params (dict): Dictionnaire complet de paramètres.

    Returns:
        dict[str, Any]: Valeurs aplaties prêtes pour une ligne DataFrame.

    Example:
        >>> meta = _extraire_meta_params(obtenir_parametres_defaut())
        >>> meta['lambda1']
        1.5
    """
    duty_clim = params.get("duty_cycle", {}).get("climatisation", {})
    niveaux   = params.get("niveaux_puissance", {}).get("climatisation", [])

    return {
        "lambda1":           params.get("lambda1"),
        "d_min":             params.get("d_min"),
        "x_DUTY_clim":       duty_clim.get("x_DUTY"),
        "x_DUTY_prime_clim": duty_clim.get("x_DUTY_prime"),
        "niveaux_clim":      str(niveaux),   # liste → string pour le CSV
    }


def _metriques_nan() -> dict[str, float]:
    """Retourne un dict de métriques remplies de NaN (run échoué).

    Returns:
        dict[str, float]: Toutes les métriques à float('nan') ou 0.

    Example:
        >>> _metriques_nan()['f1']
        nan
    """
    return {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "precision": float("nan"),
        "rappel":    float("nan"),
        "f1":        float("nan"),
        "rmse":      float("nan"),
        "mae":       float("nan"),
    }


def sauvegarder_csv(df: pd.DataFrame, chemin_sortie: str) -> None:
    """Sauvegarde le DataFrame de résultats en CSV UTF-8.

    Args:
        df             (pd.DataFrame): Résultats de executer_experiences().
        chemin_sortie  (str): Chemin complet du fichier CSV à créer.

    Raises:
        OSError: Si le répertoire parent n'existe pas et ne peut être créé.

    Example:
        >>> sauvegarder_csv(df_resultats, "output/sensibilite_661_2018-07-15.csv")
    """
    from pathlib import Path

    Path(chemin_sortie).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(chemin_sortie, index=False, encoding="utf-8")
    print(f"\nCSV sauvegardé : {chemin_sortie}  ({len(df)} lignes)")
