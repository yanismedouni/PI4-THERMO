
"""
# Pour tester directement ce fichier
if __name__ == "__main__":
    print("="*70)
    print("TEST DU MODULE DONNEES")
    print("="*70)
    
    # Test 1 : Chargement depuis CSV
    print("\nTest 1 : Chargement depuis CSV")
    print("-"*70)
    try:
        donnees_csv = charger_donnees_csv(
            'donnees_finales.csv',
            liste_appareils=['Climatisation', 'Chauffe-eau'],
            niveaux_puissance_defaut=[0.0, 0.3, 0.6, 1.0, 1.5]
        )
        print("\n  Résumé des données chargées :")
        print(f"    - Nombre de pas de temps : {len(donnees_csv['T'])}")
        print(f"    - Appareils : {donnees_csv['A']}")
        print(f"    - Clients : {donnees_csv['I']}")
        print(f"    - Pas de temps en pointe : {len(donnees_csv['T_POINTE'])}")
        
        # Validation
        valider_donnees(donnees_csv)
        
    except FileNotFoundError:
        print("  ⚠️ Fichier 'donnees_finales.csv' non trouvé")
        print("  → Placez le fichier dans le même répertoire que donnees.py")
    
    # Test 2 : Création de données exemple
    print("\n\nTest 2 : Création de données exemple (synthétiques)")
    print("-"*70)
    donnees_synth = creer_donnees_exemple(n_pas_temps=48, n_clients=1, n_appareils=2)
    print(f"  ✓ Données synthétiques créées")
    print(f"  - Nombre de pas de temps : {len(donnees_synth['T'])}")
    print(f"  - Appareils : {donnees_synth['A']}")
    valider_donnees(donnees_synth)
    
    print("\n" + "="*70)
    print("✓ Tests terminés")
    print("="*70)
"""

