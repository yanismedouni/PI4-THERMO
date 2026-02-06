"""
Configuration des paramètres de l'algorithme de désagrégation
"""


def obtenir_parametres_defaut():
    """
    Retourne les paramètres par défaut de l'algorithme.
    
    Retourne :
    ---------
    dict : Dictionnaire contenant tous les paramètres de l'algorithme
    
    Paramètres inclus :
    - lambda1 : Poids de régularisation L1 du baseload
    - lambda2 : Poids de régularisation L2 (non utilisé actuellement)
    - d_min : Durée minimale ON (en pas de temps)
    - x_DUTY : Paramètre duty-cycle en période de pointe
    - x_DUTY_prime : Paramètre duty-cycle hors-pointe
    - T_ext_MAX : Température extérieure maximale (°C)
    - T_ext_MIN : Température extérieure minimale (°C)
    - M : Constante Big-M pour linéarisation
    """
    return {
        # Paramètres de régularisation
        'lambda1': 0.01,        # Poids de régularisation L1 du baseload
        'lambda2': 0.001,       # Poids de régularisation L2 (non utilisé)
        
        # Contraintes opérationnelles
        'd_min': 3,             # Durée minimale ON (en pas de temps)
        'x_DUTY': 2.0,          # Paramètre duty-cycle en période de pointe
        'x_DUTY_prime': 3.0,    # Paramètre duty-cycle hors-pointe
        
        # Contraintes thermiques
        'T_ext_MAX': 30.0,      # Température extérieure maximale (°C)
        'T_ext_MIN': 15.0,      # Température extérieure minimale (°C)
        
        # Paramètres d'optimisation
        'M': 1000.0,            # Constante Big-M pour linéarisation
    }





# Pour tester directement ce fichier
if __name__ == "__main__":
    print("="*70)
    print("TEST DU MODULE PARAMETRES")
    print("="*70)
    
    # Test 1 : Paramètres par défaut
    print("\nTest 1 : Paramètres par défaut")
    print("-"*70)
    params = obtenir_parametres_defaut()
    for cle, valeur in params.items():
        print(f"  {cle:15} : {valeur}")
    
    