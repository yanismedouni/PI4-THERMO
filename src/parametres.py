"""
Configuration des paramètres de l'algorithme de désagrégation
"""


def obtenir_parametres_defaut():
    """
    Retourne les paramètres par défaut de l'algorithme.
    
    Retourne :
    ---------
    dict : Dictionnaire des paramètres
    """
    return {
        # Régularisation
        'lambda1': 0.01,        # Poids de régularisation L1 du baseload
        'lambda2': 0.001,       # Poids de régularisation L2 (non utilisé actuellement)
        
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


def creer_parametres_personnalises(**kwargs):
    """
    Crée un ensemble de paramètres personnalisés.
    
    Paramètres :
    -----------
    **kwargs : dict
        Paramètres à remplacer par rapport aux valeurs par défaut
    
    Retourne :
    ---------
    dict : Dictionnaire des paramètres
    
    Exemple :
    --------
    params = creer_parametres_personnalises(lambda1=0.05, d_min=5)
    """
    params = obtenir_parametres_defaut()
    params.update(kwargs)
    return params
