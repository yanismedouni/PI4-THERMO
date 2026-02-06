"""
Modele d'optimisation MICQP pour la desagregation des TCLs
Base sur la formulation du document section 6.5.4
"""

import cvxpy as cp
import numpy as np


def creer_modele_optimisation(donnees, parametres):
    """
    Cree le modele d'optimisation MICQP pour la desagregation.
    
    Parametres :
    -----------
    donnees : dict
        Dictionnaire contenant les donnees preparees (de donnees.py)
    parametres : dict
        Dictionnaire contenant les parametres (de parametres.py)
        
    Retourne :
    ---------
    dict : Dictionnaire contenant le modele et les variables
        - probleme : probleme CVXPY
        - variables : dictionnaire des variables d'optimisation
    """
    print("=" * 70)
    print("CREATION DU MODELE D'OPTIMISATION")
    print("=" * 70)
    
    # Extraire les donnees
    T = donnees['T']
    P_total = donnees['P_total']
    T_ext = donnees['T_ext']
    heures = donnees['heures']
    
    # Extraire les parametres
    lambda1 = parametres['lambda1']
    d_min = parametres['d_min']
    x_DUTY = parametres['x_DUTY']
    x_DUTY_prime = parametres['x_DUTY_prime']
    T_ext_MAX = parametres['T_ext_MAX']
    T_ext_MIN = parametres['T_ext_MIN']
    M = parametres['M']
    appareils = parametres['appareils']
    niveaux_puissance = parametres['niveaux_puissance']
    heures_pointe = parametres['heures_pointe']
    
    print(f"\nConfiguration:")
    print(f"  Pas de temps (T)     : {T}")
    print(f"  Appareils            : {appareils}")
    print(f"  Lambda1 (reg L1)     : {lambda1}")
    print(f"  d_min (duree min ON) : {d_min}")
    
    # ========================================================================
    # VARIABLES D'OPTIMISATION
    # ========================================================================
    print("\nCreation des variables d'optimisation...")
    
    # Baseload (consommation de base)
    p_BASE = cp.Variable(T, nonneg=True)  # eq (7)
    
    # Dictionnaire pour stocker les variables par appareil
    variables_appareils = {}
    
    for appareil in appareils:
        L_a = len(niveaux_puissance[appareil])  # Nombre de niveaux
        
        # Variables binaires pour chaque appareil
        o_a = cp.Variable(T, boolean=True)      # eq (1) - ON/OFF
        x_a_l = cp.Variable((T, L_a), boolean=True)  # eq (2) - niveau l
        s_a = cp.Variable(T, boolean=True)      # eq (3) - start
        f_a = cp.Variable(T, boolean=True)      # eq (4) - finish
        u_a = cp.Variable(T, boolean=True)      # eq (6) - autorisation thermique
        
        variables_appareils[appareil] = {
            'o': o_a,
            'x': x_a_l,
            's': s_a,
            'f': f_a,
            'u': u_a,
            'niveaux': niveaux_puissance[appareil]
        }
    
    print(f"  Variables creees pour {len(appareils)} appareil(s)")
    
    # ========================================================================
    # FONCTION OBJECTIF
    # ========================================================================
    print("\nConstruction de la fonction objectif...")
    
    # Puissance estimee pour chaque appareil
    P_estimee_appareils = []
    for appareil in appareils:
        vars_a = variables_appareils[appareil]
        niveaux = np.array(vars_a['niveaux'])
        
        # Puissance = somme sur les niveaux : x_a_l[t,l] * niveau[l]
        P_a = cp.sum(vars_a['x'] @ niveaux)  # Pour chaque t
        P_estimee_appareils.append(vars_a['x'] @ niveaux)
    
    # Somme des puissances de tous les appareils
    P_appareils_total = cp.sum(P_estimee_appareils, axis=0)
    
    # Erreur de reconstruction (terme quadratique)
    erreur = P_total - p_BASE - P_appareils_total
    terme_quadratique = cp.sum_squares(erreur)
    
    # Regularisation L1 sur les variations du baseload
    B = cp.diff(p_BASE)  # eq (8) - B_i(t) = p_BASE(t) - p_BASE(t-1)
    terme_L1 = lambda1 * cp.sum(cp.abs(B))
    
    # Fonction objectif totale
    objectif = cp.Minimize(terme_quadratique + terme_L1)
    
    print("  Fonction objectif construite")
    
    # ========================================================================
    # CONTRAINTES
    # ========================================================================
    print("\nConstruction des contraintes...")
    contraintes = []
    
    for appareil in appareils:
        vars_a = variables_appareils[appareil]
        o_a = vars_a['o']
        x_a_l = vars_a['x']
        s_a = vars_a['s']
        f_a = vars_a['f']
        u_a = vars_a['u']
        L_a = len(vars_a['niveaux'])
        
        # --- Contraintes par pas de temps ---
        for t in range(T):
            
            # eq (9) : o_a(t) <= u_a(t) - autorisation thermique
            contraintes.append(o_a[t] <= u_a[t])
            
            # eq (10) : somme x_a_l = 1 (un seul niveau a la fois)
            contraintes.append(cp.sum(x_a_l[t, :]) == 1)
            
            # eq (11) : somme x_a_l (l != 0) = o_a (ON si niveau > 0)
            contraintes.append(cp.sum(x_a_l[t, 1:]) == o_a[t])
            
            # eq (17-18) : Contraintes thermiques (Big-M)
            # Pour chauffage : autorise si T_ext < T_ext_MAX
            contraintes.append(T_ext[t] - T_ext_MAX <= M * u_a[t])
            contraintes.append(T_ext[t] - T_ext_MIN >= -M * (1 - u_a[t]))
        
        # --- Contraintes temporelles (transitions) ---
        for t in range(1, T):
            # eq (12) : o_a(t) - o_a(t-1) = s_a(t) - f_a(t)
            contraintes.append(o_a[t] - o_a[t-1] == s_a[t] - f_a[t])
            
            # eq (13) : s_a(t) + f_a(t) <= 1 (pas de start et finish simultanes)
            contraintes.append(s_a[t] + f_a[t] <= 1)
        
        # --- Contrainte de duree minimale ON (eq 14) ---
        if d_min > 1:
            for t in range(T - d_min + 1):
                # Pas plus d'une transition dans une fenetre de d_min pas
                contraintes.append(
                    cp.sum(s_a[t:t+d_min]) + cp.sum(f_a[t:t+d_min]) <= 1
                )
        
        # --- Contraintes de duty-cycle (eq 15-16) ---
        # Identifier les periodes de pointe et hors-pointe
        indices_pointe = [i for i in range(T) if heures[i] in heures_pointe]
        indices_hors_pointe = [i for i in range(T) if heures[i] not in heures_pointe]
        
        if len(indices_pointe) > 0:
            # eq (15) : Minimum ON en pointe
            contraintes.append(
                cp.sum(o_a[indices_pointe]) >= len(indices_pointe) / x_DUTY
            )
        
        if len(indices_hors_pointe) > 0:
            # eq (16) : Maximum ON hors-pointe
            contraintes.append(
                cp.sum(o_a[indices_hors_pointe]) <= len(indices_hors_pointe) / x_DUTY_prime
            )
    
    print(f"  Contraintes creees : {len(contraintes)} contraintes")
    
    # ========================================================================
    # PROBLEME CVXPY
    # ========================================================================
    probleme = cp.Problem(objectif, contraintes)
    
    print("\n" + "=" * 70)
    print("MODELE CREE AVEC SUCCES")
    print("=" * 70)
    print(f"  Variables binaires   : {sum([vars_a['o'].size + vars_a['x'].size + vars_a['s'].size + vars_a['f'].size + vars_a['u'].size for vars_a in variables_appareils.values()])}")
    print(f"  Variables continues  : {p_BASE.size}")
    print(f"  Contraintes          : {len(contraintes)}")
    
    return {
        'probleme': probleme,
        'variables': {
            'p_BASE': p_BASE,
            'appareils': variables_appareils
        }
    }


def resoudre_optimisation(modele, verbose=True):
    """
    Resout le probleme d'optimisation avec MOSEK.
    
    Parametres :
    -----------
    modele : dict
        Dictionnaire contenant le probleme et les variables
    verbose : bool
        Afficher les informations de resolution
        
    Retourne :
    ---------
    dict : Dictionnaire contenant les resultats
        - statut : statut de la resolution
        - valeur_optimale : valeur de l'objectif
        - p_BASE : baseload estime
        - appareils : resultats par appareil
    """
    print("\n" + "=" * 70)
    print("RESOLUTION DU PROBLEME D'OPTIMISATION")
    print("=" * 70)
    
    probleme = modele['probleme']
    
    # Resoudre avec MOSEK
    print("\nResolution avec MOSEK...")
    try:
        probleme.solve(solver=cp.MOSEK, verbose=verbose)
    except Exception as e:
        print(f"ERREUR lors de la resolution : {e}")
        return None
    
    # Verifier le statut
    print(f"\nStatut de la resolution : {probleme.status}")
    
    if probleme.status not in ["optimal", "optimal_inaccurate"]:
        print("ATTENTION : La resolution n'a pas converge vers l'optimal")
        return None
    
    print(f"Valeur optimale de l'objectif : {probleme.value:.4f}")
    
    # Extraire les resultats
    variables = modele['variables']
    
    resultats = {
        'statut': probleme.status,
        'valeur_optimale': probleme.value,
        'p_BASE': variables['p_BASE'].value,
        'appareils': {}
    }
    
    for appareil, vars_a in variables['appareils'].items():
        resultats['appareils'][appareil] = {
            'o': vars_a['o'].value,
            'x': vars_a['x'].value,
            's': vars_a['s'].value,
            'f': vars_a['f'].value,
            'u': vars_a['u'].value,
            'niveaux': vars_a['niveaux'],
            # Calculer la puissance estimee
            'P_estimee': (vars_a['x'].value @ np.array(vars_a['niveaux']))
        }
    
    print("\n" + "=" * 70)
    print("RESOLUTION TERMINEE")
    print("=" * 70)
    
    return resultats

