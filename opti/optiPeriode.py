"""
Optimisation MICQP sur une période étendue (été ou année complète).

L'horizon d'optimisation reste de 7 jours (identique à optiUneSemaine.py).
Le script découpe automatiquement la période choisie en semaines consécutives,
résout chaque semaine, puis fusionne tous les CSVs en un seul fichier de sortie.

Usage :
    python opti/optiPeriode.py --dataid 3864 --periode ete
    python opti/optiPeriode.py --dataid 3864 --periode annee --annee 2015
    python opti/optiPeriode.py --dataid 3864 --periode ete   --annee 2014
    python opti/optiPeriode.py --dataid 3864 --periode ete   --max_time 480 --gap 0.02
    python opti/optiPeriode.py --batch --periode ete         (tous les clients de RUNS)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from optiUneSemaine import (
    charger_semaine_clients,
    construire_donnees_modele,
    resoudre_optimisation,
    sauvegarder_resultats,
    afficher_resultats,
    MAX_TIME_DEFAUT,
    GAP_DEFAUT,
    DATA_PATH,
    OUTPUT_DIR,
    NB_JOURS,
)
from parametres import obtenir_parametres_defaut, afficher_parametres
from modele_opti import creer_modele_optimisation


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Clients et années à traiter en mode --batch
RUNS = [
    (3864, 2015),
    (3938, 2016),
    (6377, 2014),
    (7062, 2014),
    (7114, 2015),
    (8061, 2016),
    (8342, 2018),
    (8574, 2014),
    (8733, 2014),
    (9213, 2014),
    (9612, 2014),
]

# Définition des périodes
PERIODES = {
    "ete"  : {"mois_debut": 6,  "mois_fin": 8},   # juin → août inclus
    "annee": {"mois_debut": 1,  "mois_fin": 12},   # janvier → décembre
}

ANNEE_PAR_DEFAUT = 2015


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION DES SEMAINES
# ─────────────────────────────────────────────────────────────────────────────

def generer_semaines(periode: str, annee: int) -> list[str]:
    """
    Retourne la liste des dates de début de semaine (str "YYYY-MM-DD")
    couvrant la période demandée, sans chevauchement.

    Les semaines sont consécutives (pas glissantes) : chaque semaine
    commence là où la précédente s'est terminée.
    """
    config = PERIODES[periode]
    debut  = pd.Timestamp(annee, config["mois_debut"], 1)
    mois_fin = config["mois_fin"]
    # Dernier jour du mois de fin
    fin = pd.Timestamp(annee, mois_fin, 1) + pd.offsets.MonthEnd(0)

    semaines = []
    curseur  = debut
    while curseur + pd.Timedelta(days=NB_JOURS) <= fin + pd.Timedelta(days=1):
        semaines.append(str(curseur.date()))
        curseur += pd.Timedelta(days=NB_JOURS)

    return semaines


# ─────────────────────────────────────────────────────────────────────────────
# FUSION DES CSVS D'UNE PÉRIODE
# ─────────────────────────────────────────────────────────────────────────────

def fusionner_resultats(dataid: int, periode: str, annee: int) -> Path | None:
    """
    Concatène tous les CSVs hebdomadaires d'un client pour la période
    et sauvegarde un CSV consolidé.
    """
    pattern = f"resultats_desagregation_{dataid}_*_7jours.csv"
    fichiers = sorted(OUTPUT_DIR.glob(pattern))

    if not fichiers:
        print(f"  Aucun CSV trouvé pour le client {dataid}.")
        return None

    fragments = [pd.read_csv(f) for f in fichiers]
    df_fusionne = pd.concat(fragments, ignore_index=True)
    df_fusionne = df_fusionne.sort_values("timestamp").reset_index(drop=True)

    nom_sortie = OUTPUT_DIR / f"resultats_{dataid}_{periode}_{annee}_complet.csv"
    df_fusionne.to_csv(nom_sortie, index=False)
    print(f"  ✔ Résultats consolidés : {nom_sortie.name}  ({len(df_fusionne)} lignes)")
    return nom_sortie


# ─────────────────────────────────────────────────────────────────────────────
# TRAITEMENT D'UN CLIENT SUR TOUTE LA PÉRIODE
# ─────────────────────────────────────────────────────────────────────────────

def executer_client_periode(
    dataid: int,
    periode: str,
    annee: int,
    max_time: float,
    gap: float,
    afficher: bool = False,
) -> dict:
    """
    Résout toutes les semaines de la période pour un client.
    Retourne un dict de métriques agrégées.
    """
    semaines = generer_semaines(periode, annee)
    n_semaines = len(semaines)

    print(f"\n  Client {dataid} | période : {periode} {annee}")
    print(f"  {n_semaines} semaine(s) à traiter : {semaines[0]} → {semaines[-1]}")

    n_ok, n_err = 0, 0
    rmse_total, mae_total = [], []

    for i, date_debut in enumerate(semaines, 1):
        print(f"\n  ── Semaine {i}/{n_semaines} : {date_debut} ──")

        try:
            items = charger_semaine_clients(
                date_debut=date_debut,
                nb_clients=1,
                dataid=dataid,
            )
        except Exception as e:
            print(f"    ✗ Chargement échoué : {e}")
            n_err += 1
            continue

        if not items:
            print(f"    Aucune donnée disponible — semaine ignorée.")
            n_err += 1
            continue

        item = items[0]
        df_client = item["df"]

        params  = obtenir_parametres_defaut()
        donnees = construire_donnees_modele(df_client, params)
        modele  = creer_modele_optimisation(donnees, params)

        resultats = resoudre_optimisation(modele, verbose=False,
                                          max_time=max_time, gap=gap)
        if resultats is None:
            print(f"    ✗ Optimisation sans solution.")
            n_err += 1
            continue

        sauvegarder_resultats(donnees, resultats, dataid, date_debut, df_client)

        if afficher:
            afficher_resultats(donnees, resultats, params, dataid, date_debut, df_client)

        # Métriques hebdomadaires
        res_clim = resultats["appareils"]["climatisation"]
        p_est    = res_clim["P_estimee"]
        p_reel   = df_client["P_clim_reel"].values
        rmse_total.append(np.sqrt(np.mean((p_reel - p_est) ** 2)))
        mae_total.append(np.mean(np.abs(p_reel - p_est)))
        n_ok += 1

    print(f"\n  Semaines résolues : {n_ok}/{n_semaines}  ({n_err} échec(s))")

    metriques = {
        "dataid"          : dataid,
        "periode"         : periode,
        "annee"           : annee,
        "semaines_ok"     : n_ok,
        "semaines_erreur" : n_err,
        "RMSE moyen (kW)" : round(float(np.mean(rmse_total)), 4) if rmse_total else float("nan"),
        "MAE moyen (kW)"  : round(float(np.mean(mae_total)),  4) if mae_total  else float("nan"),
    }

    if n_ok > 0:
        fusionner_resultats(dataid, periode, annee)

    return metriques


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Désagrégation MICQP — période étendue (été ou année)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python optiPeriode.py --dataid 3864 --periode ete
  python optiPeriode.py --dataid 3864 --periode ete   --annee 2014
  python optiPeriode.py --dataid 3864 --periode annee --annee 2015
  python optiPeriode.py --batch --periode ete
        """,
    )
    parser.add_argument("--dataid",   type=int,   default=None,
                        help="Client spécifique (ignoré si --batch)")
    parser.add_argument("--periode",  type=str,   default="ete",
                        choices=list(PERIODES.keys()),
                        help="'ete' (juin-août) ou 'annee' (jan-déc)")
    parser.add_argument("--annee",    type=int,   default=ANNEE_PAR_DEFAUT,
                        help="Année civile à analyser")
    parser.add_argument("--max_time", type=float, default=MAX_TIME_DEFAUT)
    parser.add_argument("--gap",      type=float, default=GAP_DEFAUT)
    parser.add_argument("--afficher", action="store_true",
                        help="Afficher les graphiques de chaque semaine")
    parser.add_argument("--batch",    action="store_true",
                        help="Traiter tous les clients définis dans RUNS")
    args = parser.parse_args()

    print("=" * 70)
    print(f"PIPELINE OPTIMISATION MICQP — PÉRIODE : {args.periode.upper()} {args.annee}")
    print("=" * 70)
    print(f"  Fichier source : {DATA_PATH}")
    print(f"  Dossier output : {OUTPUT_DIR}")
    print(f"  Temps max/sem  : {args.max_time:.0f} s")
    print(f"  Gap cible      : {args.gap * 100:.1f} %")

    # Prévisualisation des semaines
    semaines = generer_semaines(args.periode, args.annee)
    print(f"\n  Semaines générées : {len(semaines)}")
    print(f"    Première : {semaines[0]}   Dernière : {semaines[-1]}")

    # Liste de clients à traiter
    if args.batch:
        clients = [(did, annee) for did, annee in RUNS]
        print(f"\n  Mode batch : {len(clients)} client(s)")
    else:
        if args.dataid is None:
            parser.error("--dataid est requis sauf en mode --batch")
        clients = [(args.dataid, args.annee)]
        print(f"\n  Mode individuel : dataid={args.dataid}  année={args.annee}")

    # Boucle principale
    tous_metriques = []

    for dataid_run, annee_run in clients:
        print(f"\n{'━' * 70}")
        print(f"CLIENT {dataid_run}  —  {args.periode.upper()} {annee_run}")
        print(f"{'━' * 70}")

        try:
            m = executer_client_periode(
                dataid   = dataid_run,
                periode  = args.periode,
                annee    = annee_run,
                max_time = args.max_time,
                gap      = args.gap,
                afficher = args.afficher,
            )
            tous_metriques.append(m)
        except KeyboardInterrupt:
            print(f"\n  Interruption — résultats partiels déjà sauvegardés dans output/")
            break
        except Exception as e:
            print(f"\n  ✗ Client {dataid_run} échoué : {e}")
            continue

    # Tableau récapitulatif
    if tous_metriques:
        print("\n" + "=" * 70)
        print("RÉCAPITULATIF")
        print("=" * 70)
        df_recap = pd.DataFrame(tous_metriques)
        print(df_recap.to_string(index=False))

        recap_path = OUTPUT_DIR / f"recap_{args.periode}_{args.annee}.csv"
        df_recap.to_csv(recap_path, index=False)
        print(f"\n  ✔ Récapitulatif sauvegardé : {recap_path.name}")

    print("\n" + "=" * 70)
    print("Pipeline terminé.")
    print("=" * 70)


if __name__ == "__main__":
    main()
