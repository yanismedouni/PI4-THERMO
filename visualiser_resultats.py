"""
Script de visualisation des resultats de desagregation (VERSION SIMPLIFIEE)
Focus sur la desagregation du furnace uniquement
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
import pandas as pd

from src.parametres import obtenir_parametres_defaut
from src.donnees import charger_donnees, preparer_donnees_optimisation
from src.modele_opti import creer_modele_optimisation, resoudre_optimisation


def visualiser_resultats(donnees, resultats, parametres):
    """
    Cree des graphiques pour comparer les resultats reels et estimes.
    Focus sur le furnace seulement.
    
    Parametres :
    -----------
    donnees : dict
        Donnees d'entree
    resultats : dict
        Resultats de l'optimisation
    parametres : dict
        Parametres utilises
    """
    
    # Extraire les donnees
    timestamps = pd.to_datetime(donnees['timestamps'])
    T_ext = donnees['T_ext']
    furnace_real = donnees['furnace_real']
    
    # Extraire les resultats
    P_furnace_est = resultats['appareils']['furnace']['P_estimee']
    o_furnace = resultats['appareils']['furnace']['o']
    
    # ========================================================================
    # Figure 1 : Desagregation du Furnace
    # ========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Desagregation du Chauffage (Furnace) - Client 3687 (2 Jan 2014)', 
                 fontsize=16, fontweight='bold')
    
    # --- Subplot 1 : Furnace (reel vs estime) ---
    ax1 = axes[0]
    ax1.plot(timestamps, furnace_real, 'o-', label='Furnace reel', 
             color='green', linewidth=2, markersize=4)
    ax1.plot(timestamps, P_furnace_est, 's--', label='Furnace estime', 
             color='orange', linewidth=1.5, markersize=3, alpha=0.7)
    ax1.set_ylabel('Puissance (kW)', fontsize=12)
    ax1.set_title('Chauffage (Furnace) : Reel vs Estime', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Calculer et afficher les metriques
    erreur_furnace = np.abs(furnace_real - P_furnace_est).mean()
    erreur_relative = (erreur_furnace / furnace_real.mean()) * 100 if furnace_real.mean() > 0 else 0
    
    stats_text = f'Erreur moyenne : {erreur_furnace:.4f} kW\n'
    stats_text += f'Erreur relative : {erreur_relative:.1f}%\n\n'
    stats_text += f'Reel moyen     : {furnace_real.mean():.3f} kW\n'
    stats_text += f'Estime moyen   : {P_furnace_est.mean():.3f} kW\n\n'
    stats_text += f'Reel max       : {furnace_real.max():.3f} kW\n'
    stats_text += f'Estime max     : {P_furnace_est.max():.3f} kW'
    
    ax1.text(0.02, 0.98, stats_text, 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
             verticalalignment='top', family='monospace')
    
    # --- Subplot 2 : Temperature et etat ON/OFF ---
    ax2 = axes[1]
    ax2_temp = ax2.twinx()
    
    # Etat ON/OFF (binaire)
    ax2.fill_between(timestamps, 0, o_furnace, 
                      alpha=0.3, color='red', label='Furnace ON (estime)', step='post')
    ax2.set_ylabel('Etat ON/OFF', fontsize=12, color='red')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'])
    ax2.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    # Temperature
    ax2_temp.plot(timestamps, T_ext, '-', label='Temperature ext.', 
                  color='blue', linewidth=2)
    ax2_temp.set_ylabel('Temperature (C)', fontsize=12, color='blue')
    ax2_temp.tick_params(axis='y', labelcolor='blue', labelsize=11)
    
    # Ajouter ligne de reference pour T_MIN et T_MAX
    T_MIN = parametres['T_ext_MIN']
    T_MAX = parametres['T_ext_MAX']
    ax2_temp.axhline(y=T_MIN, color='blue', linestyle='--', linewidth=1, alpha=0.5, label=f'T_MIN ({T_MIN}C)')
    ax2_temp.axhline(y=T_MAX, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'T_MAX ({T_MAX}C)')
    
    ax2.set_xlabel('Temps', fontsize=12)
    ax2.set_title('Etat du Chauffage et Temperature Exterieure', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Stats ON/OFF
    temps_ON_estime = o_furnace.sum() * 15  # minutes
    temps_total = len(o_furnace) * 15
    pourcentage_ON = (temps_ON_estime / temps_total) * 100
    
    stats_on_text = f'Temps ON estime : {temps_ON_estime:.0f} min ({pourcentage_ON:.1f}%)\n'
    stats_on_text += f'Niveaux puissance : {parametres["niveaux_puissance"]["furnace"]} kW'
    
    ax2.text(0.02, 0.95, stats_on_text, 
             transform=ax2.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
             verticalalignment='top', family='monospace')
    
    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_temp.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Format x-axis pour tous les subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = 'resultats_desagregation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegarde : {output_path}")
    
    # ========================================================================
    # Figure 2 : Metriques de performance
    # ========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Metriques de Performance - Furnace', fontsize=16, fontweight='bold')
    
    # --- Subplot 1 : Scatter plot (reel vs estime) ---
    ax_scatter = axes2[0]
    ax_scatter.scatter(furnace_real, P_furnace_est, alpha=0.6, s=50, color='purple')
    
    # Ligne de reference (parfait)
    min_val = min(furnace_real.min(), P_furnace_est.min())
    max_val = max(furnace_real.max(), P_furnace_est.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Prediction parfaite')
    
    ax_scatter.set_xlabel('Furnace Reel (kW)', fontsize=12)
    ax_scatter.set_ylabel('Furnace Estime (kW)', fontsize=12)
    ax_scatter.set_title('Correlation : Reel vs Estime', fontsize=13, fontweight='bold')
    ax_scatter.legend(fontsize=10)
    ax_scatter.grid(True, alpha=0.3)
    
    # Calculer R^2
    correlation_matrix = np.corrcoef(furnace_real, P_furnace_est)
    r_squared = correlation_matrix[0, 1]**2
    
    # Calculer RMSE
    rmse = np.sqrt(np.mean((furnace_real - P_furnace_est)**2))
    
    metrics_text = f'R2 = {r_squared:.4f}\n'
    metrics_text += f'RMSE = {rmse:.4f} kW\n'
    metrics_text += f'MAE = {erreur_furnace:.4f} kW'
    
    ax_scatter.text(0.05, 0.95, metrics_text, 
                    transform=ax_scatter.transAxes, fontsize=11, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    verticalalignment='top', family='monospace')
    
    # --- Subplot 2 : Distribution des erreurs ---
    ax_hist = axes2[1]
    erreurs = P_furnace_est - furnace_real
    
    ax_hist.hist(erreurs, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
    ax_hist.axvline(x=erreurs.mean(), color='green', linestyle='--', linewidth=2, label=f'Moyenne ({erreurs.mean():.4f})')
    
    ax_hist.set_xlabel('Erreur (kW)', fontsize=12)
    ax_hist.set_ylabel('Frequence', fontsize=12)
    ax_hist.set_title('Distribution des Erreurs', fontsize=13, fontweight='bold')
    ax_hist.legend(fontsize=10)
    ax_hist.grid(True, alpha=0.3, axis='y')
    
    # Stats erreur
    stats_err = f'Moyenne : {erreurs.mean():.4f} kW\n'
    stats_err += f'Std     : {erreurs.std():.4f} kW\n'
    stats_err += f'Min     : {erreurs.min():.4f} kW\n'
    stats_err += f'Max     : {erreurs.max():.4f} kW'
    
    ax_hist.text(0.98, 0.95, stats_err, 
                 transform=ax_hist.transAxes, fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7),
                 verticalalignment='top', horizontalalignment='right',
                 family='monospace')
    
    plt.tight_layout()
    
    output_path2 = 'metriques_performance.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegarde : {output_path2}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("VISUALISATION TERMINEE")
    print("=" * 70)


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    try:
        print("=" * 70)
        print("VISUALISATION DES RESULTATS DE DESAGREGATION")
        print("=" * 70)
        
        # 1. Charger les donnees
        print("\n1. Chargement des donnees...")
        df = charger_donnees('Data/merged_data.csv')
        donnees = preparer_donnees_optimisation(df)
        
        # 2. Charger les parametres
        print("\n2. Chargement des parametres...")
        parametres = obtenir_parametres_defaut()
        
        # 3. Creer et resoudre le modele
        print("\n3. Creation et resolution du modele...")
        modele = creer_modele_optimisation(donnees, parametres)
        resultats = resoudre_optimisation(modele, verbose=False)
        
        if not resultats:
            print("ERREUR : Probleme non resolu")
            sys.exit(1)
        
        # 4. Visualiser
        print("\n4. Creation des visualisations...")
        visualiser_resultats(donnees, resultats, parametres)
        
        print("\nTermine!")
        
    except Exception as e:
        print(f"\nERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)