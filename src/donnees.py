"""
Gestion des données pour l'algorithme de désagrégation
"""

import numpy as np
import pandas as pd


def creer_ensembles(n_pas_temps, heures_pointe=[(8, 12), (18, 22)]):
    """
    Crée les ensembles de pas de temps (T, T_POINTE, T_HORS_POINTE).
    
    Paramètres :
    -----------
    n_pas_temps : int
        Nombre total de pas de temps (ex: 96 pour 15min sur 24h)
    heures_pointe : list of tuple
        Liste de tuples (heure_debut, heure_fin) définissant les périodes de pointe
        Par défaut : [(8, 12), (18, 22)] = 8h-12h et 18h-22h
    
    Retourne :
    ---------
    tuple : (T, T_POINTE, T_HORS_POINTE)
        - T : array des indices de pas de temps [0, 1, 2, ..., n_pas_temps-1]
        - T_POINTE : array des indices en période de pointe
        - T_HORS_POINTE : array des indices hors période de pointe
    """
    # Tous les pas de temps
    T = np.arange(n_pas_temps)
    
    # Calculer les pas de temps par heure (ex: 96 pas / 24h = 4 pas par heure)
    pas_par_heure = n_pas_temps // 24
    
    # Créer les indices de période de pointe
    indices_pointe = []
    for heure_debut, heure_fin in heures_pointe:
        debut_idx = heure_debut * pas_par_heure
        fin_idx = heure_fin * pas_par_heure
        indices_pointe.extend(range(debut_idx, min(fin_idx, n_pas_temps)))
    
    T_POINTE = np.array(indices_pointe)
    
    # Période hors-pointe = tous les pas de temps sauf ceux de pointe
    T_HORS_POINTE = np.array([t for t in T if t not in T_POINTE])
    print(T_POINTE)
    print(f"Nombre de pas de temps en période de pointe : {len(T_POINTE)}")
    print(f"Nombre de pas de temps hors période de pointe : {len(T_HORS_POINTE)}")
    return T, T_POINTE, T_HORS_POINTE



def charger_donnees_csv(chemin_csv, liste_appareils=None, niveaux_puissance_defaut=None):
    """
    Charge les données depuis un fichier CSV et prépare le dictionnaire pour l'algorithme.
    
    Paramètres :
    -----------
    chemin_csv : str
        Chemin vers le fichier CSV contenant les colonnes :
        - temps : timestamp
        - consommation : puissance totale en kW
        - temperature : température extérieure en °C
    liste_appareils : list, optional
        Liste des noms d'appareils à désagréger
        Par défaut : ['Appareil_1', 'Appareil_2']
    niveaux_puissance_defaut : list, optional
        Niveaux de puissance par défaut pour chaque appareil
        Par défaut : [0.0, 0.5, 1.0, 1.5, 2.0]
    
    Retourne :
    ---------
    dict : Dictionnaire contenant toutes les données nécessaires
    """
    print(f"Chargement des données depuis : {chemin_csv}")
    
    # Charger le CSV
    df = pd.read_csv(chemin_csv)
    
    # Vérifier les colonnes requises
    colonnes_requises = ['temps', 'consommation', 'temperature']
    for col in colonnes_requises:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le CSV : '{col}'")
    
    # Extraire les données
    n_pas_temps = len(df)
    p_total = df['consommation'].values  # Puissance totale
    T_ext = df['temperature'].values      # Température extérieure
    
    print(f"  ✓ {n_pas_temps} pas de temps chargés")
    print(f"  ✓ Plage de consommation : [{p_total.min():.3f}, {p_total.max():.3f}] kW")
    print(f"  ✓ Plage de température : [{T_ext.min():.1f}, {T_ext.max():.1f}] °C")
    
    # Créer les ensembles de temps
    T, T_POINTE, T_HORS_POINTE = creer_ensembles(n_pas_temps)
    
    # Définir les appareils
    if liste_appareils is None:
        liste_appareils = ['Appareil_1', 'Appareil_2']
    
    A = liste_appareils
    
    # Définir les niveaux de puissance
    if niveaux_puissance_defaut is None:
        niveaux_puissance_defaut = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    L_a = {
        appareil: niveaux_puissance_defaut for appareil in A
    }
    
    # Client unique (les données sont pour un seul client)
    I = ['Client_1']
    
    # Formater p_total pour avoir la forme (n_clients, n_pas_temps)
    p_total_i_d = p_total.reshape(1, -1)
    
    print(f"  ✓ {len(A)} appareils configurés : {A}")
    print(f"  ✓ Niveaux de puissance : {niveaux_puissance_defaut}")
    
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


def creer_donnees_exemple(n_pas_temps=96, n_clients=1, n_appareils=2):
    """
    Crée des données synthétiques pour tester l'algorithme.
    
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
    # Créer les ensembles de temps
    T, T_POINTE, T_HORS_POINTE = creer_ensembles(n_pas_temps)
    
    # Définir les appareils
    A = [f"Appareil_{i+1}" for i in range(n_appareils)]
    
    # Niveaux de puissance pour chaque appareil (en kW)
    L_a = {
        appareil: [0.0, 0.5, 1.0, 1.5, 2.0] for appareil in A
    }
    
    # Définir les clients
    I = [f"Client_{i+1}" for i in range(n_clients)]
    
    # Générer la puissance agrégée simulée (en kW)
    np.random.seed(42)
    p_total_i_d = np.zeros((n_clients, n_pas_temps))
    
    for i in range(n_clients):
        # Charge de base : varie au cours de la journée
        baseload = 0.5 + 0.3 * np.sin(2 * np.pi * T / n_pas_temps)
        
        # Simuler l'utilisation des appareils
        usage_appareils = np.zeros(n_pas_temps)
        
        for a_idx in range(n_appareils):
            # Créer des périodes ON aléatoires
            debut_on = np.random.randint(0, n_pas_temps - 20)
            duree_on = np.random.randint(10, 20)
            puissance = np.random.choice([0.5, 1.0, 1.5])
            
            usage_appareils[debut_on:debut_on+duree_on] += puissance
        
        # Combiner baseload + appareils + bruit
        p_total_i_d[i, :] = baseload + usage_appareils + np.random.normal(0, 0.05, n_pas_temps)
    
    # Générer la température extérieure (en °C)
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
    Valide que les données d'entrée sont correctes et cohérentes.
    
    Paramètres :
    -----------
    donnees : dict
        Dictionnaire de données à valider
    
    Retourne :
    ---------
    bool : True si valide
    
    Lève :
    -----
    ValueError : Si les données sont invalides
    """
    print("Validation des données d'entrée...")
    
    # Vérifier les clés requises
    cles_requises = ['T', 'T_POINTE', 'T_HORS_POINTE', 'A', 'L_a', 'I', 
                     'p_total_i_d', 'T_ext']
    
    for cle in cles_requises:
        if cle not in donnees:
            raise ValueError(f"Clé manquante dans les données : '{cle}'")
    
    # Extraire les dimensions
    n_pas_temps = len(donnees['T'])
    n_clients = len(donnees['I'])
    
    # Valider les dimensions de p_total_i_d
    if donnees['p_total_i_d'].shape != (n_clients, n_pas_temps):
        raise ValueError(
            f"Dimensions de p_total_i_d incorrectes : "
            f"{donnees['p_total_i_d'].shape} != ({n_clients}, {n_pas_temps})"
        )
    
    # Valider les dimensions de T_ext
    if len(donnees['T_ext']) != n_pas_temps:
        raise ValueError(
            f"Longueur de T_ext incorrecte : "
            f"{len(donnees['T_ext'])} != {n_pas_temps}"
        )
    
    # Valider les périodes de pointe
    if not all(t in donnees['T'] for t in donnees['T_POINTE']):
        raise ValueError("T_POINTE contient des indices hors de T")
    
    if not all(t in donnees['T'] for t in donnees['T_HORS_POINTE']):
        raise ValueError("T_HORS_POINTE contient des indices hors de T")
    
    # Valider les spécifications des appareils
    for appareil in donnees['A']:
        if appareil not in donnees['L_a']:
            raise ValueError(f"Niveaux de puissance manquants pour l'appareil : {appareil}")
        
        if len(donnees['L_a'][appareil]) == 0:
            raise ValueError(f"L'appareil {appareil} n'a aucun niveau de puissance")
        
        if any(p < 0 for p in donnees['L_a'][appareil]):
            raise ValueError(f"L'appareil {appareil} a des niveaux de puissance négatifs")
    
    # Vérifier les valeurs de puissance
    if np.any(np.isnan(donnees['p_total_i_d'])):
        raise ValueError("Valeurs NaN détectées dans p_total_i_d")
    
    if np.any(donnees['p_total_i_d'] < 0):
        print("  ⚠️ Avertissement : valeurs de puissance négatives détectées")
    
    print("  ✓ Validation réussie")
    return True


