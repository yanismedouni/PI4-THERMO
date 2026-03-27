"""
Point d'entree pour l'optimisation MICQP de desagregation des TCLs.

L'algorithme voit uniquement la colonne 'grid' (consommation nette de la maison)
et estime la contribution de chaque appareil TCL (climatisation et/ou chauffage).

Les colonnes desagreees (air1, furnace1, etc.) sont utilisees uniquement
pour la validation des resultats, jamais par l'optimiseur.

Usage :
    python src/main_opti.py
    python src/main_opti.py --date 2015-07-02
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from parametres import obtenir_parametres_defaut, afficher_parametres
from modele_opti import creer_modele_optimisation, resoudre_optimisation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
OPTI_DIR = Path(__file__).parent

OUTPUT_DIR = Path(
    r"C:\Users\Samia\OneDrive - polymtl\Bureau\DBSCAN_Git\PI4-THERMO\monOpti\mes resultats"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = (
    r"C:\Users\Samia\OneDrive - polymtl\Bureau\DataThermo\yanis\california_processed_energy_data.csv"
)
DATE_DEBUT_PAR_DEFAUT = "2015-07-02"
NB_CLIENTS_PAR_DEFAUT = 12
PAS_PAR_JOUR = 96
NB_JOURS = 7
PAS_PAR_SEMAINE = PAS_PAR_JOUR * NB_JOURS

COLONNES_TCL = {
    "climatisation": ["air1", "air2", "air3"],
    "chauffage": ["furnace1", "furnace2", "heater1", "heater2", "heater3"],
}


def charger_semaine_clients(date_debut=None, nb_clients=NB_CLIENTS_PAR_DEFAUT):
    """
    Charge une semaine de donnees pour les nb_clients premiers clients
    du fichier Californie, a partir de date_debut.
    """
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["local_15min"],
    )

    date_debut_ts = pd.to_datetime(
        DATE_DEBUT_PAR_DEFAUT if date_debut is None else date_debut
    )
    date_fin_ts = date_debut_ts + pd.Timedelta(days=NB_JOURS)

    clients = sorted(df["dataid"].dropna().unique().tolist())[:nb_clients]

    if len(clients) == 0:
        raise ValueError("Aucun client trouve dans le fichier California.")

    print(f"  Date debut           : {date_debut_ts.date()}")
    print(f"  Date fin exclue      : {date_fin_ts.date()}")
    print(f"  Nombre de clients    : {len(clients)}")
    print(f"  Clients selectionnes : {clients}")

    semaines_clients = []

    for dataid in clients:
        df_client = df[df["dataid"] == dataid].copy()
        df_semaine = (
            df_client[
                (df_client["local_15min"] >= date_debut_ts)
                & (df_client["local_15min"] < date_fin_ts)
            ]
            .sort_values("local_15min")
            .reset_index(drop=True)
        )

        if df_semaine.empty:
            print(f"  [Client {dataid}] ignore : aucune donnee sur la semaine demandee.")
            continue

        df_semaine = df_semaine.iloc[:PAS_PAR_SEMAINE].copy()

        cols_clim = [c for c in COLONNES_TCL["climatisation"] if c in df_semaine.columns]
        cols_chauf = [c for c in COLONNES_TCL["chauffage"] if c in df_semaine.columns]

        df_semaine["P_clim_reel"] = (
            df_semaine[cols_clim].fillna(0).sum(axis=1) if cols_clim else 0.0
        )
        df_semaine["P_chauf_reel"] = (
            df_semaine[cols_chauf].fillna(0).sum(axis=1) if cols_chauf else 0.0
        )

        p_grid = df_semaine["grid"].fillna(0).clip(lower=0)

        print(f"\n  Client (dataid)      : {dataid}")
        print(f"  Pas de temps         : {len(df_semaine)}")
        print(
            f"  P_total (grid)       : moy={p_grid.mean():.3f} kW  "
            f"max={p_grid.max():.3f} kW"
        )
        print(
            f"  P_clim_reel  (valid) : moy={df_semaine['P_clim_reel'].mean():.3f} kW"
        )
        print(
            f"  P_chauf_reel (valid) : moy={df_semaine['P_chauf_reel'].mean():.3f} kW"
        )

        semaines_clients.append(
            {
                "dataid": int(dataid),
                "date_debut": str(date_debut_ts.date()),
                "date_fin": str((date_fin_ts - pd.Timedelta(minutes=15)).date()),
                "df": df_semaine,
            }
        )

    if len(semaines_clients) == 0:
        raise ValueError("Aucune semaine valide n'a ete chargee pour les clients demandes.")

    return semaines_clients


def construire_donnees_modele(df_jour, params):
    """
    Construit le dictionnaire de donnees attendu par le modele.
    """
    t_horizon = len(df_jour)

    p_total = df_jour["grid"].fillna(0).clip(lower=0).values

    temp_candidates = ["temperature", "temp_ext", "temperature_2m", "temp", "T_ext"]
    temp_col = next((c for c in temp_candidates if c in df_jour.columns), None)

    if temp_col is None:
        raise KeyError(
            f"Aucune colonne temperature trouvee dans df_jour.\n"
            f"Colonnes disponibles : {df_jour.columns.tolist()}\n"
            f"Noms acceptes : {temp_candidates}"
        )

    print(f"  Colonne temperature  : '{temp_col}'")
    t_ext = df_jour[temp_col].ffill().bfill().values

    heures = df_jour["local_15min"].dt.hour.to_numpy()
    params["heures"] = heures.tolist()

    for app in params["appareils"]:
        if app not in params["niveaux_puissance"]:
            raise KeyError(
                f"Appareil '{app}' absent de 'niveaux_puissance' dans parametres.py. "
                f"Appareils disponibles : {list(params['niveaux_puissance'].keys())}"
            )

    print(f"  Pas de temps         : {t_horizon}")
    print(f"  T_ext                : min={t_ext.min():.1f} C  max={t_ext.max():.1f} C")
    print(f"  Appareils            : {params['appareils']}")
    for app in params["appareils"]:
        print(f"    {app:15s} niveaux = {params['niveaux_puissance'][app]} kW")

    return {
        "T": t_horizon,
        "P_total": p_total,
        "T_ext": t_ext,
        "heures": heures,
    }


def sauvegarder_resultats_progressivement(donnees, resultats, dataid, date_debut, df_jour):
    """
    Sauvegarde les resultats avec exactement le header voulu.
    """
    t_horizon = donnees["T"]
    p_base = resultats["p_BASE"]

    output_path = OUTPUT_DIR / f"resultats_desagregation_{dataid}_{date_debut}_7jours.csv"

    header_exact = [
        "timestamp",
        "P_total",
        "p_BASE",
        "T_ext",
        "P_estime_clim",
        "o_climatisation",
        "P_reel_clima",
        "aggregate_estimated",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as fichier:
        fichier.write(",".join(header_exact) + "\n")

    if "climatisation" not in resultats["appareils"]:
        raise KeyError(
            "Le fichier de sortie demande les colonnes de climatisation, "
            "mais 'climatisation' n'est pas presente dans resultats['appareils']."
        )

    res_clim = resultats["appareils"]["climatisation"]

    for t in range(t_horizon):
        if "local_15min" in df_jour.columns:
            timestamp = df_jour.loc[t, "local_15min"]
        else:
            timestamp = t

        p_total_t = float(donnees["P_total"][t])
        p_base_t = float(p_base[t])
        t_ext_t = float(donnees["T_ext"][t])

        p_estime_clim_t = float(res_clim["P_estimee"][t])
        o_climatisation_t = int(np.round(res_clim["o"][t]))
        p_reel_clima_t = float(df_jour.loc[t, "P_clim_reel"])

        aggregate_estimated_t = p_estime_clim_t

        ligne = {
            "timestamp": timestamp,
            "P_total": p_total_t,
            "p_BASE": p_base_t,
            "T_ext": t_ext_t,
            "P_estime_clim": p_estime_clim_t,
            "o_climatisation": o_climatisation_t,
            "P_reel_clima": p_reel_clima_t,
            "aggregate_estimated": aggregate_estimated_t,
        }

        pd.DataFrame([ligne], columns=header_exact).to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
        )

    print(f"\nFichier CSV sauvegarde progressivement : {output_path}")


def afficher_resultats(donnees, resultats, params, dataid, date, df_jour):
    """
    Affiche les resultats et sauvegarde le CSV.
    """
    if resultats is None:
        print("Pas de resultats a afficher.")
        return

    t_horizon = donnees["T"]
    p_base = resultats["p_BASE"]
    t_axis = np.arange(t_horizon) * 15 / 60

    print("\n" + "=" * 70)
    print("RESULTATS")
    print("=" * 70)

    p_estime_total = np.zeros(t_horizon)

    for app in params["appareils"]:
        res = resultats["appareils"][app]
        p_est = res["P_estimee"]
        cle = "clim" if app == "climatisation" else "chauf"
        p_reel = df_jour[f"P_{cle}_reel"].values

        rmse = np.sqrt(np.mean((p_reel - p_est) ** 2))
        mae = np.mean(np.abs(p_reel - p_est))
        n_on = int(np.round(res["o"]).sum())

        print(f"  [{app}]")
        print(f"    RMSE : {rmse:.4f} kW  |  MAE : {mae:.4f} kW")
        print(f"    ON   : {n_on}/{t_horizon} pas ({100 * n_on / t_horizon:.1f}%)")

        p_estime_total += p_est

    print(f"\n  Baseload moyen : {p_base.mean():.4f} kW")

    sauvegarder_resultats_progressivement(
        donnees=donnees,
        resultats=resultats,
        dataid=dataid,
        date_debut=date,
        df_jour=df_jour,
    )

    n_panneaux = 2 + len(params["appareils"])
    fig, axes = plt.subplots(n_panneaux, 1, figsize=(14, 4 * n_panneaux), sharex=True)

    axes[0].plot(t_axis, donnees["P_total"], "k-", lw=2, label="Grid (mesure)")
    axes[0].plot(t_axis, p_estime_total, "r--", lw=2, label="TCL estime total")
    axes[0].fill_between(t_axis, p_base, alpha=0.25, color="gray", label="Baseload")
    axes[0].set_ylabel("Puissance (kW)")
    axes[0].set_title(f"Desagregation TCL — Client {dataid} — semaine a partir du {date}")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    couleurs = {"climatisation": ("b", "r"), "chauffage": ("g", "r")}

    for idx, app in enumerate(params["appareils"], start=1):
        res = resultats["appareils"][app]
        cle = "clim" if app == "climatisation" else "chauf"
        p_reel = df_jour[f"P_{cle}_reel"].values
        p_est = res["P_estimee"]
        c_r, c_e = couleurs.get(app, ("b", "r"))

        axes[idx].plot(t_axis, p_reel, color=c_r, lw=1.5, label=f"{app} reel")
        axes[idx].plot(t_axis, p_est, color=c_e, lw=2, ls="--", label=f"{app} estime")
        axes[idx].set_ylabel("Puissance (kW)")
        axes[idx].set_title(app.capitalize())
        axes[idx].legend(loc="upper right", fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    ax_t = axes[-1]
    ax_t.plot(t_axis, donnees["T_ext"], "b-", lw=1.5, label="T_ext")

    couleurs_seuils = {"climatisation": ("r", "g"), "chauffage": ("orange", "cyan")}

    for app in params["appareils"]:
        therm = params["thermique"][app]
        c_max, c_min = couleurs_seuils.get(app, ("r", "g"))

        ax_t.axhline(
            therm["T_ext_MAX"],
            color=c_max,
            ls="--",
            alpha=0.8,
            label=f"{app} T_MAX={therm['T_ext_MAX']:.1f} C",
        )
        ax_t.axhline(
            therm["T_ext_MIN"],
            color=c_min,
            ls="--",
            alpha=0.8,
            label=f"{app} T_MIN={therm['T_ext_MIN']:.1f} C",
        )

    ax_t.set_xlabel("Heure depuis le debut de la semaine")
    ax_t.set_ylabel("Temperature (C)")
    ax_t.set_title("Temperature exterieure")
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def executer_client(dataid, date_debut, df_client):
    print("\n" + "#" * 70)
    print(f"TRAITEMENT DU CLIENT {dataid}")
    print("#" * 70)

    print("\n[2] Configuration des parametres...")
    params = obtenir_parametres_defaut()
    afficher_parametres(params)

    print("\n[3] Construction des donnees pour le modele...")
    donnees = construire_donnees_modele(df_client, params)

    print("\n[4] Creation du modele...")
    modele = creer_modele_optimisation(donnees, params)

    print("\n[5] Resolution...")
    resultats = resoudre_optimisation(modele, verbose=True)

    afficher_resultats(donnees, resultats, params, dataid, date_debut, df_client)


def main():
    parser = argparse.ArgumentParser(description="Desagregation MICQP — TCLs")
    parser.add_argument("--date", type=str, default=DATE_DEBUT_PAR_DEFAUT, help="YYYY-MM-DD")
    parser.add_argument(
        "--nb_clients",
        type=int,
        default=NB_CLIENTS_PAR_DEFAUT,
        help="Nombre de premiers clients a traiter",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PIPELINE OPTIMISATION MICQP — DESAGREGATION TCL")
    print("=" * 70)

    print("\n[1] Chargement des donnees...")
    semaines_clients = charger_semaine_clients(
        date_debut=args.date,
        nb_clients=args.nb_clients,
    )

    for item in semaines_clients:
        executer_client(
            dataid=item["dataid"],
            date_debut=item["date_debut"],
            df_client=item["df"],
        )


if __name__ == "__main__":
    main()
