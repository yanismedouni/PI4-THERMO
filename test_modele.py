"""
Script de test pour modele_opti.py
"""

import sys
sys.path.append('src')

print("="*70)
print("TEST DU MODELE D'OPTIMISATION")
print("="*70)

try:
    # Import des modules
    print("\n1. Import des modules...")
    from src.parametres import obtenir_parametres_defaut
    from src.donnees import charger_donnees, preparer_donnees_optimisation
    from src.modele_opti import creer_modele_optimisation, resoudre_optimisation
    print("   OK")
    
    # Chargement des donnees
    print("\n2. Chargement des donnees...")
    df = charger_donnees('Data/merged_data.csv')
    donnees = preparer_donnees_optimisation(df)
    print("   OK")
    
    # Chargement des parametres
    print("\n3. Chargement des parametres...")
    parametres = obtenir_parametres_defaut()
    print("   OK")
    
    # Creation du modele
    print("\n4. Creation du modele...")
    modele = creer_modele_optimisation(donnees, parametres)
    print("   OK")
    
    # Resolution
    print("\n5. Resolution avec MOSEK...")
    resultats = resoudre_optimisation(modele, verbose=True)
    
    if resultats:
        print("\n"+"="*70)
        print("RESULTATS")
        print("="*70)
        print(f"Baseload moyen estime : {resultats['p_BASE'].mean():.3f} kW")
        for appareil in resultats['appareils']:
            P_est = resultats['appareils'][appareil]['P_estimee']
            print(f"{appareil} moyen estime : {P_est.mean():.3f} kW")
        print("\nTest reussi!")
    else:
        print("\nTest echoue - probleme non resolu")

except Exception as e:
    print(f"\nERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
