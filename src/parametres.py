"""
Configuration des parametres de l'algorithme de desagregation
"""


def obtenir_parametres_defaut():
    """
    Retourne les parametres par defaut de l'algorithme.
    
    Retourne :
    ---------
    dict : Dictionnaire contenant tous les parametres de l'algorithme
    
    Parametres inclus :
    - lambda1 : Poids de regularisation L1 du baseload
    - lambda2 : Poids de regularisation L2 (non utilise actuellement)
    - d_min : Duree minimale ON (en pas de temps)
    - x_DUTY : Parametre duty-cycle en periode de pointe
    - x_DUTY_prime : Parametre duty-cycle hors-pointe
    - T_ext_MAX : Temperature exterieure maximale (C)
    - T_ext_MIN : Temperature exterieure minimale (C)
    - M : Constante Big-M pour linearisation
    - niveaux_puissance : Niveaux de puissance pour chaque appareil (kW)
    - appareils : Liste des appareils a desagreger
    """
    return {
        # Parametres de regularisation
        'lambda1': 0.1,         # AUGMENTE (etait 0.01) pour penaliser plus les variations de baseload
        'lambda2': 0.001,       # Poids de regularisation L2 (non utilise)
        
        # Contraintes operationnelles
        'd_min': 2,             # REDUIT (etait 3) - duree minimale ON en pas de temps (30 min au lieu de 45)
        'x_DUTY': 10.0,         # RELACHE (etait 2.0) - moins de contrainte en pointe (10% au lieu de 50%)
        'x_DUTY_prime': 2.0,    # RELACHE (etait 3.0) - plus de flexibilite hors-pointe (50% au lieu de 33%)
        
        # Contraintes thermiques
        # Ajuste pour San Diego en janvier (7-22 C observe)
        'T_ext_MAX': 20.0,      # REDUIT (etait 25.0) - au-dessus de 20C, pas de chauffage
        'T_ext_MIN': 10.0,      # AUGMENTE (etait 5.0) - en dessous de 10C, chauffage autorise
        
        # Parametres d'optimisation
        'M': 1000.0,            # Constante Big-M pour linearisation
        
        # Appareils a desagreger
        'appareils': ['furnace'],  # Pour l'instant, seulement le chauffage
        
        # Niveaux de puissance par appareil (en kW)
        # AJUSTE selon les donnees reelles observees
        # Reel: moyenne = 0.060 kW, max = 0.306 kW
        'niveaux_puissance': {
            'furnace': [0.0, 0.15, 0.30],  # OFF, MOYEN (0.15), HAUT (0.30)
            # Avant: [0.0, 0.3, 0.5] - trop eleve
            'air': [0.0, 1.0, 2.0, 3.0],   # OFF, BAS, MOYEN, HAUT
        },
        
        # Periodes de pointe (indices d'heures, 0-23)
        # REDUIT la fenetre de pointe pour eviter de forcer l'activation le soir
        'heures_pointe': list(range(18, 21)),  # 18h a 20h seulement (etait 17h-22h)
    }


def obtenir_parametres_simples():
    """
    Version simplifiee avec seulement 2 niveaux (ON/OFF).
    Utile pour tester et comparer.
    """
    params = obtenir_parametres_defaut()
    params['niveaux_puissance']['furnace'] = [0.0, 0.15]  # OFF, ON (niveau unique)
    params['d_min'] = 2  # 30 minutes minimum
    return params


def afficher_parametres(params):
    """
    Affiche les parametres de maniere formatee.
    
    Parametres :
    -----------
    params : dict
        Dictionnaire des parametres a afficher
    """
    print("=" * 70)
    print("PARAMETRES DE L'ALGORITHME")
    print("=" * 70)
    
    print("\nRegularisation:")
    print(f"  lambda1 (L1 baseload)   : {params['lambda1']}")
    print(f"  lambda2 (L2 baseload)   : {params['lambda2']}")
    
    print("\nContraintes operationnelles:")
    print(f"  d_min (duree min ON)    : {params['d_min']} pas de temps ({params['d_min']*15} min)")
    print(f"  x_DUTY (pointe)         : {params['x_DUTY']}")
    print(f"  x_DUTY_prime (hors-pointe): {params['x_DUTY_prime']}")
    
    print("\nContraintes thermiques:")
    print(f"  T_ext_MAX               : {params['T_ext_MAX']} C")
    print(f"  T_ext_MIN               : {params['T_ext_MIN']} C")
    
    print("\nOptimisation:")
    print(f"  M (Big-M)               : {params['M']}")
    
    print("\nAppareils:")
    print(f"  Liste                   : {params['appareils']}")
    
    print("\nNiveaux de puissance:")
    for appareil, niveaux in params['niveaux_puissance'].items():
        print(f"  {appareil:15} : {niveaux} kW")
    
    print("\nHeures de pointe:")
    print(f"  {params['heures_pointe'][0]}h a {params['heures_pointe'][-1]}h")
    
    print("=" * 70)


# Pour tester directement ce fichier
if __name__ == "__main__":
    print("=" * 70)
    print("TEST DU MODULE PARAMETRES")
    print("=" * 70)
    
    # Test 1 : Parametres par defaut
    print("\nTest 1 : Parametres par defaut (ajustes)")
    params = obtenir_parametres_defaut()
    afficher_parametres(params)
    
    # Test 2 : Parametres simples
    print("\n\nTest 2 : Parametres simples (2 niveaux)")
    params_simples = obtenir_parametres_simples()
    afficher_parametres(params_simples)
    
    print("\nTest reussi!")