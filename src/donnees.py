"""
Gestion des données pour l'algorithme de désagrégation
"""

import numpy as np


def creer_donnees_exemple(n_pas_temps=96, n_clients=1, n_appareils=2):
    """
    Crée des données d'exemple pour tester l'algorithme.
    
    Paramètres :
    -----------
    n_pas_temps : int
        Nombre de pas de temps (ex: 96 pour 15min sur 24h)
    n_clients : int
        Nombre de clients
    n_appareils : int
        Nombre d'appareils
    
    Retourne :
    ---------
    dict : Dictionnaire contenant toutes les données d'entrée
    """
    # Pas de temps
    T = np.arange(n_pas_temps)
    
    # Périodes de pointe : 8h-12h et 18h-22h
    # Pour 96 intervalles : 8h = 32, 12h = 48, 18h = 72, 22h = 88
    T_POINTE = np.concatenate([
        np.arange(32, 48),  # 8h - 12h
        np.arange(72, 88)   # 18h - 22h
    ])
    
    T_HORS_POINTE = np.array([t for t in T if t not in T_POINTE])
    
    # Appareils
    A = [f"Appareil_{i+1}" for i in range(n_appareils)]
    
    # Niveaux de puissance pour chaque appareil (en kW)
    L_a = {
        appareil: [0.5, 1.0, 1.5, 2.0] for appareil in A
    }
    
    # Clients
    I = [f"Client_{i+1}" for i in range(n_clients)]
    
    # Puissance agrégée simulée (en kW)
    np.random.seed(42)
    p_total_i_d = np.zeros((n_clients, n_pas_temps))
    
    for i in range(n_clients):
        # Charge de base : varie au cours de la journée
        baseload = 0.5 + 0.3 * np.sin(2 * np.pi * T / n_pas_temps)
        
        # Ajouter l'utilisation des appareils (simplifié)
        usage_appareils = np.zeros(n_pas_temps)
        for a_idx in range(n_appareils):
            # Périodes ON aléatoires
            debut_on = np.random.randint(0, n_pas_temps - 20)
            duree_on = np.random.randint(10, 20)
            usage_appareils[debut_on:debut_on+duree_on] += np.random.choice([0.5, 1.0, 1.5])
        
        p_total_i_d[i, :] = baseload + usage_appareils + np.random.normal(0, 0.05, n_pas_temps)
    
    # Température extérieure (en °C)
    # Varie de ~15°C la nuit à ~25°C le jour
    T_ext = 20 + 5 * np.sin(2 * np.pi * (T - 24) / n_pas_temps)
    
    return {
        'T': T,
        'T_POINTE': T_POINTE,
        'T_HORS_POINTE': T_HORS_POINTE,
        'A': A,
        'L_a': L_a,
        'I': I,
        'p_total_i_d': p_total_i_d,
        'T_ext': T_ext
    }


def valider_donnees(donnees):
    """
    Valide que les données d'entrée sont correctes.
    
    Paramètres :
    -----------
    donnees : dict
        Dictionnaire de données à valider
    
    Retourne :
    ---------
    bool : True si valide, lève une exception sinon
    """
    print("Validation des données d'entrée...")
    
    cles_requises = ['T', 'T_POINTE', 'T_HORS_POINTE', 'A', 'L_a', 'I', 
                     'p_total_i_d', 'T_ext']
    
    # Vérifier les clés requises
    for cle in cles_requises:
        if cle not in donnees:
            raise ValueError(f"Clé manquante : {cle}")
    
    # Valider les dimensions
    n_pas_temps = len(donnees['T'])
    n_clients = len(donnees['I'])
    
    if donnees['p_total_i_d'].shape != (n_clients, n_pas_temps):
        raise ValueError(
            f"Dimensions de p_total_i_d {donnees['p_total_i_d'].shape} "
            f"ne correspondent pas à ({n_clients}, {n_pas_temps})"
        )
    
    if len(donnees['T_ext']) != n_pas_temps:
        raise ValueError(
            f"Longueur de T_ext {len(donnees['T_ext'])} "
            f"ne correspond pas à {n_pas_temps} pas de temps"
        )
    
    # Valider les périodes de pointe
    if not all(t in donnees['T'] for t in donnees['T_POINTE']):
        raise ValueError("T_POINTE contient des indices invalides")
    
    if not all(t in donnees['T'] for t in donnees['T_HORS_POINTE']):
        raise ValueError("T_HORS_POINTE contient des indices invalides")
    
    # Valider les spécifications des appareils
    for appareil in donnees['A']:
        if appareil not in donnees['L_a']:
            raise ValueError(f"Niveaux de puissance manquants pour : {appareil}")
        
        if not donnees['L_a'][appareil]:
            raise ValueError(f"Appareil {appareil} n'a pas de niveaux de puissance")
        
        if any(p < 0 for p in donnees['L_a'][appareil]):
            raise ValueError(f"Appareil {appareil} a des niveaux de puissance négatifs")
    
    # Valider les données de puissance
    if np.any(donnees['p_total_i_d'] < 0):
        print("  Attention : valeurs de puissance négatives détectées")
    
    if np.any(np.isnan(donnees['p_total_i_d'])):
        raise ValueError("Valeurs NaN détectées dans p_total_i_d")
    
    print("   Validation réussie")
    return True
