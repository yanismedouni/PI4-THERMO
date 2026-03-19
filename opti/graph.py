# =============================================================================
# Module      : graph.py
# Auteur      : Équipe THERMO — ELE8080
# Date        : 2025-01
# Description : Fonctions de visualisation pour la désagrégation TCL.
#               Génère et sauvegarde les graphiques puissance réelle vs estimée.
#               Appelé depuis script_param.py après chaque run.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"


# =============================================================================
# GRAPHIQUE PRINCIPAL
# =============================================================================

def tracer_desagregation(
    donnees: dict,
    resultats: dict,
    params: dict,
    df_jour: "pd.DataFrame",
    dataid: int,
    date: str,
    sauvegarder: bool = True,
    afficher: bool = True,
) -> Path | None:
    """Trace la puissance réelle vs estimée pour chaque appareil TCL.

    Génère un graphique multi-panneaux :
      - Panneau 0 : grid mesuré, TCL estimé total, baseload
      - Panneau N : un panneau par appareil (réel vs estimé)
      - Panneau final : température extérieure avec seuils thermiques

    Args:
        donnees     (dict): Sortie de construire_donnees_modele().
        resultats   (dict): Sortie de resoudre_optimisation().
        params      (dict): Paramètres du run (obtenir_parametres_defaut()).
        df_jour     (pd.DataFrame): Journée chargée par charger_journee().
        dataid      (int): Identifiant client, utilisé dans le titre et le nom de fichier.
        date        (str): Date YYYY-MM-DD, utilisée dans le titre et le nom de fichier.
        sauvegarder (bool): Si True, sauvegarde le PNG dans output/. Défaut : True.
        afficher    (bool): Si True, appelle plt.show(). Défaut : True.

    Returns:
        Path | None: Chemin du fichier PNG sauvegardé, ou None si sauvegarder=False.

    Raises:
        KeyError: Si une colonne P_clim_reel ou P_chauf_reel est absente de df_jour.

    Example:
        >>> chemin = tracer_desagregation(donnees, resultats, params,
        ...                               df_jour, 661, "2018-07-15")
    """
    T      = donnees['T']
    p_BASE = resultats['p_BASE']
    t_axis = np.arange(T) * 15 / 60  # axe temps en heures

    # Puissance totale estimée = somme sur tous les appareils
    P_estime_total = np.zeros(T)
    for app in params['appareils']:
        P_estime_total += resultats['appareils'][app]['P_estimee']

    n_panneaux = 1 + len(params['appareils']) + 1  # global + appareils + température
    fig, axes  = plt.subplots(n_panneaux, 1,
                               figsize=(14, 4 * n_panneaux),
                               sharex=True)

    # ── Panneau 0 : vue globale ───────────────────────────────────────────────
    axes[0].plot(t_axis, donnees['P_total'], 'k-',  lw=2,  label='Grid (mesuré)')
    axes[0].plot(t_axis, P_estime_total,     'r--', lw=2,  label='TCL estimé total')
    axes[0].fill_between(t_axis, p_BASE, alpha=0.25, color='gray', label='Baseload')
    axes[0].set_ylabel('Puissance (kW)')
    axes[0].set_title(f"Désagrégation TCL — Client {dataid} — {date}")
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Panneaux intermédiaires : un par appareil ─────────────────────────────
    couleurs = {'climatisation': ('b', 'r'), 'chauffage': ('g', 'darkorange')}
    for idx, app in enumerate(params['appareils'], start=1):
        cle      = 'clim' if app == 'climatisation' else 'chauf'
        col_reel = f'P_{cle}_reel'

        if col_reel not in df_jour.columns:
            raise KeyError(
                f"Colonne '{col_reel}' absente de df_jour. "
                f"Vérifier que charger_journee() a bien été appelé."
            )

        P_r      = df_jour[col_reel].fillna(0.0).values
        P_e      = resultats['appareils'][app]['P_estimee']
        c_r, c_e = couleurs.get(app, ('b', 'r'))

        # Calcul RMSE pour annotation dans le titre du panneau
        rmse = np.sqrt(np.mean((P_r - P_e) ** 2))

        axes[idx].plot(t_axis, P_r, color=c_r, lw=1.5,        label=f'{app} réel')
        axes[idx].plot(t_axis, P_e, color=c_e, lw=2, ls='--', label=f'{app} estimé')
        axes[idx].set_ylabel('Puissance (kW)')
        axes[idx].set_title(f"{app.capitalize()}  —  RMSE = {rmse:.4f} kW")
        axes[idx].legend(loc='upper right', fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    # ── Panneau final : température extérieure ────────────────────────────────
    ax_t = axes[-1]
    ax_t.plot(t_axis, donnees['T_ext'], 'b-', lw=1.5, label='T_ext')

    couleurs_seuils = {'climatisation': ('r', 'g'), 'chauffage': ('orange', 'cyan')}
    for app in params['appareils']:
        therm        = params['thermique'][app]
        c_max, c_min = couleurs_seuils.get(app, ('r', 'g'))
        ax_t.axhline(therm['T_ext_MAX'], color=c_max, ls='--', alpha=0.8,
                     label=f"{app} T_MAX={therm['T_ext_MAX']:.1f} °C")
        ax_t.axhline(therm['T_ext_MIN'], color=c_min, ls='--', alpha=0.8,
                     label=f"{app} T_MIN={therm['T_ext_MIN']:.1f} °C")

    ax_t.set_xlabel('Heure de la journée')
    ax_t.set_ylabel('Température (°C)')
    ax_t.set_title('Température extérieure')
    ax_t.legend(loc='upper right', fontsize=8)
    ax_t.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    chemin_fig = None
    if sauvegarder:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chemin_fig = OUTPUT_DIR / f"desagregation_{dataid}_{date}.png"
        plt.savefig(chemin_fig, dpi=150, bbox_inches='tight')
        print(f"  Graphique sauvegardé : {chemin_fig.name}")

    if afficher:
        plt.show()
    else:
        plt.close(fig)

    return chemin_fig


# =============================================================================
# GRAPHIQUE ON/OFF — COMPARAISON ÉTATS BINAIRES
# =============================================================================

def tracer_etats_onoff(
    resultats: dict,
    params: dict,
    df_jour: "pd.DataFrame",
    dataid: int,
    date: str,
    sauvegarder: bool = True,
    afficher: bool = True,
) -> Path | None:
    """Trace les états ON/OFF estimés vs réels sous forme de barres horizontales.

    Utile pour visualiser les faux positifs et faux négatifs pas à pas.

    Args:
        resultats   (dict): Sortie de resoudre_optimisation().
        params      (dict): Paramètres du run.
        df_jour     (pd.DataFrame): Journée chargée par charger_journee().
        dataid      (int): Identifiant client.
        date        (str): Date YYYY-MM-DD.
        sauvegarder (bool): Si True, sauvegarde le PNG. Défaut : True.
        afficher    (bool): Si True, appelle plt.show(). Défaut : True.

    Returns:
        Path | None: Chemin du fichier PNG, ou None si sauvegarder=False.

    Example:
        >>> tracer_etats_onoff(resultats, params, df_jour, 661, "2018-07-15")
    """
    T      = len(df_jour)
    t_axis = np.arange(T) * 15 / 60

    n_appareils = len(params['appareils'])
    fig, axes   = plt.subplots(n_appareils, 1,
                                figsize=(14, 3 * n_appareils),
                                sharex=True)

    # Garantir que axes est toujours itérable même pour un seul appareil
    if n_appareils == 1:
        axes = [axes]

    for idx, app in enumerate(params['appareils']):
        cle      = 'clim' if app == 'climatisation' else 'chauf'
        col_reel = f'P_{cle}_reel'

        p_clim_reel = df_jour[col_reel].fillna(0.0).values
        o_reel      = (p_clim_reel > 0.05).astype(int)
        o_estime    = np.round(resultats['appareils'][app]['o']).astype(int)

        # TP=vert, FP=rouge, FN=orange, TN=transparent
        for t in range(T):
            if o_reel[t] == 1 and o_estime[t] == 1:
                couleur, label = 'green',  'TP'
            elif o_reel[t] == 0 and o_estime[t] == 1:
                couleur, label = 'red',    'FP'
            elif o_reel[t] == 1 and o_estime[t] == 0:
                couleur, label = 'orange', 'FN'
            else:
                continue  # TN — on ne trace pas pour garder le graphe lisible

            axes[idx].axvspan(t * 15 / 60, (t + 1) * 15 / 60,
                              alpha=0.4, color=couleur)

        # Légende manuelle — une seule entrée par couleur
        from matplotlib.patches import Patch
        legende = [
            Patch(color='green',  alpha=0.4, label='TP — ON correct'),
            Patch(color='red',    alpha=0.4, label='FP — ON faux'),
            Patch(color='orange', alpha=0.4, label='FN — ON manqué'),
        ]
        axes[idx].legend(handles=legende, loc='upper right', fontsize=8)
        axes[idx].set_ylabel('État')
        axes[idx].set_title(f"{app.capitalize()} — états ON/OFF estimés vs réels")
        axes[idx].set_yticks([])
        axes[idx].grid(True, alpha=0.2)

    axes[-1].set_xlabel('Heure de la journée')
    plt.suptitle(f"Comparaison ON/OFF — Client {dataid} — {date}", fontsize=12)
    plt.tight_layout()

    chemin_fig = None
    if sauvegarder:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chemin_fig = OUTPUT_DIR / f"onoff_{dataid}_{date}.png"
        plt.savefig(chemin_fig, dpi=150, bbox_inches='tight')
        print(f"  Graphique ON/OFF sauvegardé : {chemin_fig.name}")

    if afficher:
        plt.show()
    else:
        plt.close(fig)

    return chemin_fig