"""
Script principal pour exécuter l'algorithme de désagrégation
"""

from modele_optimisation import ModeleDesagregation
from donnees import creer_donnees_exemple, valider_donnees
from parametres import obtenir_parametres_defaut


def main():
    """
    Fonction principale d'exécution.
    """
    print("="*70)
    print("ALGORITHME DE DÉSAGRÉGATION")
    print("="*70 + "\n")
    
    # ====================================================================
    # ÉTAPE 1 : PRÉPARATION DES DONNÉES
    # ====================================================================
    
    print("ÉTAPE 1 : Préparation des données")
    print("-" * 70)
    
    # Créer des données d'exemple
    donnees = creer_donnees_exemple(
        n_pas_temps=48,      # 48 = 24h avec intervalles de 30min
        n_clients=1,
        n_appareils=2
    )
    
    print(f"  - Pas de temps : {len(donnees['T'])}")
    print(f"  - Clients : {len(donnees['I'])}")
    print(f"  - Appareils : {len(donnees['A'])}")
    print(f"  - Période de pointe : {len(donnees['T_POINTE'])} pas de temps")
    print(f"  - Période hors-pointe : {len(donnees['T_HORS_POINTE'])} pas de temps")
    
    # Valider les données
    valider_donnees(donnees)
    
    print()
    
    # ====================================================================
    # ÉTAPE 2 : CONFIGURATION DES PARAMÈTRES
    # ====================================================================
    
    print("ÉTAPE 2 : Configuration des paramètres")
    print("-" * 70)
    
    parametres = obtenir_parametres_defaut()
    
    print("  Paramètres de l'algorithme :")
    for param, valeur in parametres.items():
        print(f"    {param}: {valeur}")
    
    print()
    
    # ====================================================================
    # ÉTAPE 3 : CONSTRUCTION ET RÉSOLUTION DU MODÈLE
    # ====================================================================
    
    print("ÉTAPE 3 : Construction et résolution")
    print("-" * 70)
    
    # Initialiser le modèle
    modele = ModeleDesagregation(donnees, parametres)
    
    # Construire le modèle d'optimisation
    modele.construire_modele()
    
    # Résoudre
    resultats = modele.resoudre()
    
    # ====================================================================
    # ÉTAPE 4 : AFFICHAGE DES RÉSULTATS
    # ====================================================================
    
    print("\nÉTAPE 4 : Résultats")
    print("-" * 70)
    
    print(f"\nValeur de l'objectif : {resultats['valeur_objectif']:.4f}")
    
    # Statistiques de la charge de base
    print("\nCharge de base :")
    for client in donnees['I']:
        p_base = resultats['p_BASE'][client]
        print(f"  {client}:")
        print(f"    Moyenne : {np.mean(p_base):.2f} kW")
        print(f"    Min     : {np.min(p_base):.2f} kW")
        print(f"    Max     : {np.max(p_base):.2f} kW")
    
    # Statistiques des appareils
    print("\nUtilisation des appareils :")
    for client in donnees['I']:
        print(f"  {client}:")
        for appareil in donnees['A']:
            cle = f"{client}_{appareil}"
            o_values = resultats['o'][cle]
            temps_on = np.sum(o_values)
            pourcentage_on = 100 * temps_on / len(donnees['T'])
            
            print(f"    {appareil}:")
            print(f"      Temps ON : {int(temps_on)} / {len(donnees['T'])} ({pourcentage_on:.1f}%)")
    
    print("\n" + "="*70)
    print("Exécution terminée avec succès!")
    print("="*70 + "\n")
    
    return resultats


if __name__ == "__main__":
    import numpy as np
    resultats = main()
