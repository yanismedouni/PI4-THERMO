"""
Point d'entree pour l'optimisation MICQP de desagregation des TCLs.

Usage :
    python opti/optiUneSemaine.py
    python opti/optiUneSemaine.py --dataid 1642
    python opti/optiUneSemaine.py --dataid 1642 --date 2015-08-10
    python opti/optiUneSemaine.py --nb_clients 3
    python opti/optiUneSemaine.py --dataid 1642 --max_time 480
    python opti/optiUneSemaine.py --dataid 1642 --gap 0.02
    python opti/optiUneSemaine.py --dataid 1642 --max_time 600 --gap 0.01
"""

import argparse
import sys
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from parametres import obtenir_parametres_defaut, afficher_parametres
from modele_opti import creer_modele_optimisation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
OPTI_DIR    = Path(__file__).parent
OUTPUT_DIR  = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_DIR / "csv" / "california_processed_energy_data.csv"

DATE_DEBUT_PAR_DEFAUT = "2014-07-01"
NB_CLIENTS_PAR_DEFAUT = 1
PAS_PAR_JOUR          = 96
NB_JOURS              = 7
PAS_PAR_SEMAINE       = PAS_PAR_JOUR * NB_JOURS

MAX_TIME_DEFAUT = 600
GAP_DEFAUT      = 0.7

COLONNES_TCL = {
    "climatisation": ["air1", "air2", "air3"],
    "chauffage"    : ["furnace1", "furnace2", "heater1", "heater2", "heater3"],
}


# ─────────────────────────────────────────────────────────────────────────────
# RÉSOLUTION MOSEK
# ─────────────────────────────────────────────────────────────────────────────

def resoudre_optimisation(
    modele: dict,
    verbose: bool = True,
    max_time: float = MAX_TIME_DEFAUT,
    gap: float = GAP_DEFAUT,
) -> dict | None:
    """
    Résout le problème MICQP avec MOSEK.
    """
    print("\n" + "=" * 70)
    print("RÉSOLUTION DU PROBLÈME")
    print("=" * 70)
    print(f"  Solveur          : MOSEK")
    print(f"  Temps max        : {max_time:.0f} s")
    print(f"  Gap relatif cible: {gap * 100:.1f} %")
    sys.stdout.flush()

    probleme = modele["probleme"]

    try:
        probleme.solve(
            solver=cp.MOSEK,
            verbose=verbose,
            mosek_params={
                "MSK_DPAR_MIO_TOL_REL_GAP": gap,
                "MSK_DPAR_MIO_MAX_TIME"   : max_time,
            },
        )
    except Exception as e:
        print(f"ERREUR MOSEK : {e}")
        return None

    sys.stdout.flush()

    print(f"\n  Statut           : {probleme.status}")

    if probleme.status not in ("optimal", "optimal_inaccurate"):
        print("  ATTENTION : pas de solution optimale trouvée.")
        if probleme.status == "optimal_inaccurate":
            print(
                "  → Le temps maximal a peut-être été atteint avant convergence.\n"
                "    Essayez d'augmenter --max_time ou d'assouplir --gap."
            )
        return None

    print(f"  Valeur objectif  : {probleme.value:.6f}")

    if probleme.status == "optimal_inaccurate":
        print(
            "  ⚠️  Solution retournée mais hors tolérance de gap.\n"
            "     Résultats utilisables mais potentiellement sous-optimaux."
        )

    variables = modele["variables"]
    resultats = {
        "statut"         : probleme.status,
        "valeur_optimale": probleme.value,
        "p_BASE"         : variables["p_BASE"].value,
        "appareils"      : {},
    }

    for appareil, va in variables["appareils"].items():
        niveaux = np.array(va["niveaux"], dtype=float)
        x_val   = va["x"].value
        resultats["appareils"][appareil] = {
            "o"        : va["o"].value,
            "x"        : x_val,
            "s"        : va["s"].value,
            "f"        : va["f"].value,
            "u"        : va["u"].value,
            "niveaux"  : va["niveaux"],
            "P_estimee": x_val @ niveaux,
        }

    print("=" * 70)
    print("RÉSOLUTION TERMINÉE")
    print("=" * 70)
    return resultats


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def charger_semaine_clients(
    date_debut: str | None = None,
    nb_clients: int = NB_CLIENTS_PAR_DEFAUT,
    dataid: int | None = None,
) -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {DATA_PATH}\n"
            f"Vérifiez que le fichier est bien dans le dossier csv/ du projet."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["local_15min"])

    date_debut_ts = pd.to_datetime(
        DATE_DEBUT_PAR_DEFAUT if date_debut is None else date_debut
    )
    date_fin_ts = date_debut_ts + pd.Timedelta(days=NB_JOURS)

    tous_clients = sorted(df["dataid"].dropna().unique().tolist())

    if dataid is not None:
        if dataid not in tous_clients:
            raise ValueError(
                f"dataid {dataid} introuvable dans le fichier.\n"
                f"Clients disponibles : {tous_clients}"
            )
        clients = [dataid]
        print(f"  Mode              : client spécifique")
    else:
        clients = tous_clients[:nb_clients]
        print(f"  Mode              : {nb_clients} premiers clients")

    print(f"  Date début        : {date_debut_ts.date()}")
    print(f"  Date fin exclue   : {date_fin_ts.date()}")
    print(f"  Clients           : {clients}")

    semaines_clients = []

    for client_id in clients:
        df_client  = df[df["dataid"] == client_id].copy()
        df_semaine = (
            df_client[
                (df_client["local_15min"] >= date_debut_ts)
                & (df_client["local_15min"] < date_fin_ts)
            ]
            .sort_values("local_15min")
            .reset_index(drop=True)
        )

        if df_semaine.empty:
            print(f"  [Client {client_id}] ignoré : aucune donnée sur la semaine demandée.")
            continue

        df_semaine = df_semaine.iloc[:PAS_PAR_SEMAINE].copy()

        cols_clim  = [c for c in COLONNES_TCL["climatisation"] if c in df_semaine.columns]
        cols_chauf = [c for c in COLONNES_TCL["chauffage"]     if c in df_semaine.columns]

        df_semaine["P_clim_reel"]  = (
            df_semaine[cols_clim].fillna(0).sum(axis=1)  if cols_clim  else 0.0
        )
        df_semaine["P_chauf_reel"] = (
            df_semaine[cols_chauf].fillna(0).sum(axis=1) if cols_chauf else 0.0
        )

        p_grid = df_semaine["grid"].fillna(0).clip(lower=0)

        print(f"\n  Client (dataid)      : {client_id}")
        print(f"  Pas de temps         : {len(df_semaine)}")
        print(
            f"  P_total (grid)       : moy={p_grid.mean():.3f} kW  "
            f"max={p_grid.max():.3f} kW"
        )
        print(f"  P_clim_reel  (valid) : moy={df_semaine['P_clim_reel'].mean():.3f} kW")
        print(f"  P_chauf_reel (valid) : moy={df_semaine['P_chauf_reel'].mean():.3f} kW")

        semaines_clients.append({
            "dataid"    : int(client_id),
            "date_debut": str(date_debut_ts.date()),
            "date_fin"  : str((date_fin_ts - pd.Timedelta(minutes=15)).date()),
            "df"        : df_semaine,
        })

    if len(semaines_clients) == 0:
        raise ValueError("Aucune semaine valide chargée pour les clients demandés.")

    return semaines_clients


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DES DONNÉES MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

def construire_donnees_modele(df_jour: pd.DataFrame, params: dict) -> dict:
    t_horizon = len(df_jour)
    p_total   = df_jour["grid"].fillna(0).clip(lower=0).values

    temp_candidates = ["temperature", "temp_ext", "temperature_2m", "temp", "T_ext"]
    temp_col = next((c for c in temp_candidates if c in df_jour.columns), None)

    if temp_col is None:
        raise KeyError(
            f"Aucune colonne température trouvée.\n"
            f"Colonnes disponibles : {df_jour.columns.tolist()}\n"
            f"Noms acceptés : {temp_candidates}"
        )

    print(f"  Colonne température  : '{temp_col}'")
    t_ext  = df_jour[temp_col].ffill().bfill().values
    heures = df_jour["local_15min"].dt.hour.to_numpy()

    params["heures"] = heures.tolist()

    for app in params["appareils"]:
        if app not in params["niveaux_puissance"]:
            raise KeyError(
                f"Appareil '{app}' absent de 'niveaux_puissance'.\n"
                f"Appareils disponibles : {list(params['niveaux_puissance'].keys())}"
            )

    print(f"  Pas de temps         : {t_horizon}")
    print(f"  T_ext                : min={t_ext.min():.1f} °C  max={t_ext.max():.1f} °C")
    print(f"  Appareils            : {params['appareils']}")
    for app in params["appareils"]:
        print(f"    {app:15s} niveaux = {params['niveaux_puissance'][app]} kW")

    return {
        "T"      : t_horizon,
        "P_total": p_total,
        "T_ext"  : t_ext,
        "heures" : heures,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

def sauvegarder_resultats(
    donnees: dict,
    resultats: dict,
    dataid: int,
    date_debut: str,
    df_jour: pd.DataFrame,
) -> None:
    t_horizon   = donnees["T"]
    p_base      = resultats["p_BASE"]
    output_path = OUTPUT_DIR / f"resultats_desagregation_{dataid}_{date_debut}_7jours.csv"

    header = [
        "timestamp", "P_total", "p_BASE", "T_ext",
        "P_estime_clim", "o_climatisation", "P_reel_clim", "aggregate_estimated",
    ]

    if "climatisation" not in resultats["appareils"]:
        raise KeyError("'climatisation' absente de resultats['appareils'].")

    res_clim = resultats["appareils"]["climatisation"]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

    for t in range(t_horizon):
        timestamp       = df_jour.loc[t, "local_15min"] if "local_15min" in df_jour.columns else t
        p_estime_clim_t = float(res_clim["P_estimee"][t])

        ligne = {
            "timestamp"          : timestamp,
            "P_total"            : float(donnees["P_total"][t]),
            "p_BASE"             : float(p_base[t]),
            "T_ext"              : float(donnees["T_ext"][t]),
            "P_estime_clim"      : p_estime_clim_t,
            "o_climatisation"    : int(np.round(res_clim["o"][t])),
            "P_reel_clim"        : float(df_jour.loc[t, "P_clim_reel"]),
            "aggregate_estimated": p_estime_clim_t,
        }

        pd.DataFrame([ligne], columns=header).to_csv(
            output_path, mode="a", header=False, index=False
        )

    print(f"  ✔ Résultats sauvegardés : {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────

def afficher_resultats(
    donnees: dict,
    resultats: dict,
    params: dict,
    dataid: int,
    date: str,
    df_jour: pd.DataFrame,
) -> None:
    if resultats is None:
        print("Pas de résultats à afficher.")
        return

    t_horizon      = donnees["T"]
    p_base         = resultats["p_BASE"]
    t_axis         = np.arange(t_horizon) * 15 / 60
    p_estime_total = np.zeros(t_horizon)

    print("\n" + "=" * 70)
    print("RÉSULTATS")
    print("=" * 70)

    for app in params["appareils"]:
        res    = resultats["appareils"][app]
        p_est  = res["P_estimee"]
        cle    = "clim" if app == "climatisation" else "chauf"
        p_reel = df_jour[f"P_{cle}_reel"].values

        rmse  = np.sqrt(np.mean((p_reel - p_est) ** 2))
        mae   = np.mean(np.abs(p_reel - p_est))
        n_on  = int(np.round(res["o"]).sum())

        print(f"  [{app}]")
        print(f"    RMSE : {rmse:.4f} kW  |  MAE : {mae:.4f} kW")
        print(f"    ON   : {n_on}/{t_horizon} pas ({100 * n_on / t_horizon:.1f}%)")
        p_estime_total += p_est

    print(f"\n  Baseload moyen : {p_base.mean():.4f} kW")

    sauvegarder_resultats(donnees, resultats, dataid, date, df_jour)

    n_panneaux = 2 + len(params["appareils"])
    fig, axes  = plt.subplots(n_panneaux, 1, figsize=(14, 4 * n_panneaux), sharex=True)

    axes[0].plot(t_axis, donnees["P_total"], "k-",  lw=2, label="Grid (mesure)")
    axes[0].plot(t_axis, p_estime_total,     "r--", lw=2, label="TCL estimé total")
    axes[0].fill_between(t_axis, p_base, alpha=0.25, color="gray", label="Baseload")
    axes[0].set_ylabel("Puissance (kW)")
    axes[0].set_title(f"Désagrégation TCL — Client {dataid} — semaine du {date}")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    couleurs = {"climatisation": ("b", "r"), "chauffage": ("g", "r")}
    for idx, app in enumerate(params["appareils"], start=1):
        res    = resultats["appareils"][app]
        cle    = "clim" if app == "climatisation" else "chauf"
        p_reel = df_jour[f"P_{cle}_reel"].values
        p_est  = res["P_estimee"]
        c_r, c_e = couleurs.get(app, ("b", "r"))

        axes[idx].plot(t_axis, p_reel, color=c_r, lw=1.5,        label=f"{app} réel")
        axes[idx].plot(t_axis, p_est,  color=c_e, lw=2, ls="--", label=f"{app} estimé")
        axes[idx].set_ylabel("Puissance (kW)")
        axes[idx].set_title(app.capitalize())
        axes[idx].legend(loc="upper right", fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    ax_t = axes[-1]
    ax_t.plot(t_axis, donnees["T_ext"], "b-", lw=1.5, label="T_ext")

    couleurs_seuils = {"climatisation": ("r", "g"), "chauffage": ("orange", "cyan")}
    for app in params["appareils"]:
        therm    = params["thermique"][app]
        c_max, c_min = couleurs_seuils.get(app, ("r", "g"))
        ax_t.axhline(therm["T_ext_MAX"], color=c_max, ls="--", alpha=0.8,
                     label=f"{app} T_MAX={therm['T_ext_MAX']:.1f} °C")
        ax_t.axhline(therm["T_ext_MIN"], color=c_min, ls="--", alpha=0.8,
                     label=f"{app} T_MIN={therm['T_ext_MIN']:.1f} °C")

    ax_t.set_xlabel("Heure depuis le début de la semaine")
    ax_t.set_ylabel("Température (°C)")
    ax_t.set_title("Température extérieure")
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"graphique_{dataid}_{date}_7jours.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  ✔ Graphique sauvegardé : {fig_path.name}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# TRAITEMENT D'UN CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def executer_client(
    dataid: int,
    date_debut: str,
    df_client: pd.DataFrame,
    max_time: float,
    gap: float,
) -> None:
    print("\n" + "#" * 70)
    print(f"TRAITEMENT DU CLIENT {dataid}")
    print("#" * 70)

    print("\n[2] Configuration des paramètres...")
    params = obtenir_parametres_defaut()
    afficher_parametres(params)

    print("\n[3] Construction des données pour le modèle...")
    donnees = construire_donnees_modele(df_client, params)

    print("\n[4] Création du modèle...")
    modele = creer_modele_optimisation(donnees, params)

    print("\n[5] Résolution...")
    resultats = resoudre_optimisation(
        modele,
        verbose=True,
        max_time=max_time,
        gap=gap,
    )

    afficher_resultats(donnees, resultats, params, dataid, date_debut, df_client)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# LISTE DE RUNS (dataid, date_debut)
# ─────────────────────────────────────────────────────────────────────────────
RUNS = [
    (3864, "2015-07-01"),
    (3938, "2016-07-01"),
    (4495, "2018-07-01"),
    (4934, "2015-07-01"),
    (5938, "2016-07-01"),
    (6377, "2014-07-01"),
    (6547, "2015-07-01"),
    (7062, "2014-07-01"),
    (7114, "2015-07-01"),
    (8061, "2016-07-01"),
    (8342, "2018-07-01"),
    (8574, "2014-07-01"),
    (8733, "2014-07-01"),
    (9213, "2014-07-01"),
    (9612, "2014-07-01"),
]
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Désagrégation MICQP — TCLs sur une semaine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python optiUneSemaine.py
  python optiUneSemaine.py --dataid 1642
  python optiUneSemaine.py --dataid 1642 --date 2015-08-10
  python optiUneSemaine.py --max_time 480
  python optiUneSemaine.py --gap 0.01
  python optiUneSemaine.py --batch          ← nouveau : roule tous les RUNS
        """,
    )
    parser.add_argument("--dataid",     type=int,   default=None)
    parser.add_argument("--date",       type=str,   default=DATE_DEBUT_PAR_DEFAUT)
    parser.add_argument("--nb_clients", type=int,   default=NB_CLIENTS_PAR_DEFAUT)
    parser.add_argument("--max_time",   type=float, default=MAX_TIME_DEFAUT)
    parser.add_argument("--gap",        type=float, default=GAP_DEFAUT)
    parser.add_argument("--batch",      action="store_true",
        help="Roule tous les runs définis dans RUNS (ignore --dataid et --date)")
    args = parser.parse_args()

    print("=" * 70)
    print("PIPELINE OPTIMISATION MICQP — DÉSAGRÉGATION TCL")
    print("=" * 70)
    print(f"\n  Fichier source : {DATA_PATH}")
    print(f"  Dossier output : {OUTPUT_DIR}")
    print(f"  Temps max      : {args.max_time:.0f} s")
    print(f"  Gap cible      : {args.gap * 100:.1f} %")

    # ── Construire la liste de runs ──────────────────────────────────────────
    if args.batch:
        runs = RUNS
        print(f"\n  Mode batch : {len(runs)} run(s) planifiés")
        for i, (did, d) in enumerate(runs, 1):
            print(f"    {i:2d}. dataid={did}  date={d}")
    else:
        runs = [(args.dataid, args.date)]
        print(f"\n  Mode individuel : dataid={args.dataid}  date={args.date}")

    # ── Boucle sur tous les runs ─────────────────────────────────────────────
    n_ok, n_err = 0, 0

    for run_idx, (dataid_run, date_run) in enumerate(runs, 1):
        print(f"\n{'━' * 70}")
        print(f"RUN {run_idx}/{len(runs)} — dataid={dataid_run}  date={date_run}")
        print(f"{'━' * 70}")

        try:
            print("\n[1] Chargement des données...")
            semaines = charger_semaine_clients(
                date_debut=date_run,
                nb_clients=1,
                dataid=dataid_run,
            )

            for item in semaines:
                executer_client(
                    dataid    =item["dataid"],
                    date_debut=item["date_debut"],
                    df_client =item["df"],
                    max_time  =args.max_time,
                    gap       =args.gap,
                )
            n_ok += 1

        except KeyboardInterrupt:
            print(f"\n⚠️  Interruption au run {run_idx} — arrêt du batch.")
            print(f"✔ {n_ok} run(s) complétés, résultats dans output/")
            break
        except Exception as e:
            print(f"\n✗ Run {run_idx} échoué : {e}")
            n_err += 1
            continue  # passe au run suivant sans tout arrêter

    print("\n" + "=" * 70)
    print(f"✔ Pipeline terminé — {n_ok} succès, {n_err} échec(s).")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:          # aucun argument → force le mode batch
        sys.argv.append("--batch")
    main()