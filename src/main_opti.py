"""
Point d'entree pour l'optimisation MICQP de desagregation des TCLs.

L'algorithme voit uniquement la colonne 'grid' (consommation nette de la maison)
et estime la contribution de chaque appareil TCL (climatisation et/ou chauffage).

Les colonnes desagreees (air1, furnace1, etc.) sont utilisees uniquement
pour la validation des resultats, jamais par l'optimiseur.

Usage :
    python src/main_opti.py
    python src/main_opti.py --dataid 661 --date 2018-07-15
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parametres import obtenir_parametres_defaut, afficher_parametres
from modele_opti import creer_modele_optimisation, resoudre_optimisation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / "output"
ENERGY_FILE = OUTPUT_DIR / "austin_processed_energy_data.csv"

# Colonnes TCL par appareil (pour calcul de P_reel — validation seulement)
COLONNES_TCL = {
    'climatisation': ['air1', 'air2', 'air3'],
    'chauffage':     ['furnace1', 'furnace2', 'heater1', 'heater2', 'heater3'],
}


def charger_journee(dataid=None, date=None):
    """
    Charge une journee de donnees pour un client depuis le fichier Pecan Street.

    Parametres :
    -----------
    dataid : int, optional
        Identifiant du client. Si None, prend le premier client disponible.
    date : str, optional
        Date au format 'YYYY-MM-DD'. Si None, prend la premiere date disponible.

    Retourne :
    ---------
    df_jour : DataFrame
        96 lignes (pas de 15 min) avec toutes les colonnes du fichier source.
        Colonnes ajoutees :
          - 'P_clim_reel'  : somme air1+air2+air3        (validation seulement)
          - 'P_chauf_reel' : somme furnace+heater         (validation seulement)
    dataid : int
    date_str : str
    """
    df = pd.read_csv(ENERGY_FILE, parse_dates=['local_15min'])

    # Selectionner le client
    if dataid is None:
        dataid = int(df['dataid'].iloc[0])
    df_client = df[df['dataid'] == dataid].copy()

    # Selectionner la journee
    df_client['date'] = df_client['local_15min'].dt.date
    if date is None:
        date_choisie = sorted(df_client['date'].unique())[0]
    else:
        date_choisie = pd.to_datetime(date).date()

    df_jour = (df_client[df_client['date'] == date_choisie]
               .sort_values('local_15min')
               .reset_index(drop=True)
               .iloc[:96])

    # Colonnes TCL presentes dans le fichier
    cols_clim  = [c for c in COLONNES_TCL['climatisation'] if c in df_jour.columns]
    cols_chauf = [c for c in COLONNES_TCL['chauffage']     if c in df_jour.columns]

    # Puissance reelle par appareil — validation seulement, pas transmise au modele
    df_jour['P_clim_reel']  = df_jour[cols_clim].fillna(0).sum(axis=1)
    df_jour['P_chauf_reel'] = df_jour[cols_chauf].fillna(0).sum(axis=1)

    P_grid = df_jour['grid'].fillna(0).clip(lower=0)

    print(f"  Client (dataid)      : {dataid}")
    print(f"  Date                 : {date_choisie}")
    print(f"  Pas de temps         : {len(df_jour)}")
    print(f"  P_total (grid)       : moy={P_grid.mean():.3f} kW  "
          f"max={P_grid.max():.3f} kW")
    print(f"  P_clim_reel  (valid) : moy={df_jour['P_clim_reel'].mean():.3f} kW")
    print(f"  P_chauf_reel (valid) : moy={df_jour['P_chauf_reel'].mean():.3f} kW")

    return df_jour, dataid, str(date_choisie)


def construire_donnees_modele(df_jour, params):
    """
    Construit le dictionnaire de donnees attendu par creer_modele_optimisation
    directement depuis df_jour, sans fichier intermediaire.

    Parametres :
    -----------
    df_jour : DataFrame  (sortie de charger_journee)
    params  : dict       (sortie de obtenir_parametres_defaut)

    Retourne :
    ---------
    dict :
        'T'       : int   — nombre de pas de temps
        'P_total' : array — consommation nette (grid), shape (T,)
        'T_ext'   : array — temperature exterieure, shape (T,)
        'heures'  : array — heure locale (0-23) pour chaque pas, shape (T,)
    """
    T = len(df_jour)

    # Consommation nette — seule donnee visible par l'optimiseur
    P_total = df_jour['grid'].fillna(0).clip(lower=0).values

    # Temperature exterieure (colonne 'temperature' ajoutee par add_temperature.py)
    T_ext = df_jour['temperature'].ffill().bfill().values

    # Heure locale pour chaque pas (utilisee pour les contraintes duty-cycle)
    heures = (np.arange(T) * 15 // 60)

    # Injecter les heures dans params
    params['heures'] = heures.tolist()

    # Verification que chaque appareil a bien ses niveaux definis
    for app in params['appareils']:
        if app not in params['niveaux_puissance']:
            raise KeyError(
                f"Appareil '{app}' absent de 'niveaux_puissance' dans parametres.py. "
                f"Appareils disponibles : {list(params['niveaux_puissance'].keys())}"
            )

    print(f"  Pas de temps  : {T}")
    print(f"  T_ext         : min={T_ext.min():.1f} C  max={T_ext.max():.1f} C")
    print(f"  Appareils     : {params['appareils']}")
    for app in params['appareils']:
        print(f"    {app:15s} niveaux = {params['niveaux_puissance'][app]} kW")

    return {
        'T':       T,
        'P_total': P_total,
        'T_ext':   T_ext,
        'heures':  heures,
    }


def afficher_resultats(donnees, resultats, params, dataid, date, df_jour):
    """Affiche les metriques et le graphique de desagregation."""
    if resultats is None:
        print("Pas de resultats a afficher.")
        return

    T      = donnees['T']
    p_BASE = resultats['p_BASE']
    t_axis = np.arange(T) * 15 / 60

    print("\n" + "=" * 70)
    print("RESULTATS")
    print("=" * 70)

    P_estime_total = np.zeros(T)
    for app in params['appareils']:
        res = resultats['appareils'][app]
        P_e = res['P_estimee']
        cle = 'clim' if app == 'climatisation' else 'chauf'
        P_r = df_jour[f'P_{cle}_reel'].values

        rmse = np.sqrt(np.mean((P_r - P_e) ** 2))
        mae  = np.mean(np.abs(P_r - P_e))
        n_on = int(np.round(res['o']).sum())

        print(f"  [{app}]")
        print(f"    RMSE : {rmse:.4f} kW  |  MAE : {mae:.4f} kW")
        print(f"    ON   : {n_on}/{T} pas ({100*n_on/T:.1f}%)")
        P_estime_total += P_e

    print(f"\n  Baseload moyen : {p_BASE.mean():.4f} kW")

    # ── Graphique ─────────────────────────────────────────────────────────────
    n_panneaux = 2 + len(params['appareils'])  # global + 1 par appareil + temp
    fig, axes  = plt.subplots(n_panneaux, 1, figsize=(14, 4 * n_panneaux), sharex=True)

    # Panneau 0 : puissances globales
    axes[0].plot(t_axis, donnees['P_total'], 'k-',  lw=2,  label='Grid (mesure)')
    axes[0].plot(t_axis, P_estime_total,     'r--', lw=2,  label='TCL estime total')
    axes[0].fill_between(t_axis, p_BASE, alpha=0.25, color='gray', label='Baseload')
    axes[0].set_ylabel('Puissance (kW)')
    axes[0].set_title(f"Desagregation TCL — Client {dataid} — {date}")
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panneaux intermediaires : un par appareil
    couleurs = {'climatisation': ('b', 'r'), 'chauffage': ('g', 'r')}
    for idx, app in enumerate(params['appareils'], start=1):
        res      = resultats['appareils'][app]
        cle      = 'clim' if app == 'climatisation' else 'chauf'
        P_r      = df_jour[f'P_{cle}_reel'].values
        P_e      = res['P_estimee']
        c_r, c_e = couleurs.get(app, ('b', 'r'))

        axes[idx].plot(t_axis, P_r, color=c_r, lw=1.5,       label=f'{app} reel')
        axes[idx].plot(t_axis, P_e, color=c_e, lw=2, ls='--', label=f'{app} estime')
        axes[idx].set_ylabel('Puissance (kW)')
        axes[idx].set_title(app.capitalize())
        axes[idx].legend(loc='upper right', fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    # Dernier panneau : temperature
    ax_t = axes[-1]
    ax_t.plot(t_axis, donnees['T_ext'], 'b-', lw=1.5, label='T_ext')

    # Tracer les seuils thermiques pour chaque appareil actif
    couleurs_seuils = {'climatisation': ('r', 'g'), 'chauffage': ('orange', 'cyan')}
    for app in params['appareils']:
        therm    = params['thermique'][app]
        c_max, c_min = couleurs_seuils.get(app, ('r', 'g'))
        ax_t.axhline(therm['T_ext_MAX'], color=c_max, ls='--', alpha=0.8,
                     label=f"{app} T_MAX={therm['T_ext_MAX']:.1f} C")
        ax_t.axhline(therm['T_ext_MIN'], color=c_min, ls='--', alpha=0.8,
                     label=f"{app} T_MIN={therm['T_ext_MIN']:.1f} C")

    ax_t.set_xlabel('Heure de la journee')
    ax_t.set_ylabel('Temperature (C)')
    ax_t.set_title('Temperature exterieure')
    ax_t.legend(loc='upper right', fontsize=8)
    ax_t.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"resultats_opti_{dataid}_{date}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Graphique sauvegarde : {fig_path.name}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Desagregation MICQP — TCLs")
    parser.add_argument('--dataid', type=int, default=None)
    parser.add_argument('--date',   type=str, default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    print("=" * 70)
    print("PIPELINE OPTIMISATION MICQP — DESAGREGATION TCL")
    print("=" * 70)

    print("\n[1] Chargement des donnees...")
    df_jour, dataid, date = charger_journee(dataid=args.dataid, date=args.date)

    print("\n[2] Configuration des parametres...")
    params = obtenir_parametres_defaut()
    afficher_parametres(params)

    print("\n[3] Construction des donnees pour le modele...")
    donnees = construire_donnees_modele(df_jour, params)

    print("\n[4] Creation du modele...")
    modele = creer_modele_optimisation(donnees, params)

    print("\n[5] Resolution...")
    resultats = resoudre_optimisation(modele, verbose=False)

    afficher_resultats(donnees, resultats, params, dataid, date, df_jour)


if __name__ == "__main__":
    main()