"""
Algorithme de désagrégation - Modèle d'optimisation
"""

import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense


class ModeleDesagregation:
    """
    Modèle d'optimisation mixte (MIP) pour la désagrégation de charge.
    
    Implémente l'algorithme décrit dans le pseudo-code pour estimer :
    - La charge de base (baseload)
    - Les états ON/OFF des appareils
    - Les niveaux de puissance sélectionnés
    - Les autorisations thermiques
    """
    
    def __init__(self, donnees, parametres):
        """
        Initialise le modèle de désagrégation.
        
        Paramètres :
        -----------
        donnees : dict
            Dictionnaire contenant les données d'entrée (ensembles et mesures)
        parametres : dict
            Dictionnaire contenant les paramètres de l'algorithme
        """
        # Stockage des données
        self.donnees = donnees
        self.parametres = parametres
        
        # Dimensions du problème
        self.n_clients = len(donnees['I'])
        self.n_appareils = len(donnees['A'])
        self.n_pas_temps = len(donnees['T'])
        
        # Initialisation
        self.modele = None
        self.variables = {}
        self.resultats = None
        
    def construire_modele(self):
        """
        Construit le modèle d'optimisation MOSEK Fusion.
        """
        print("Construction du modèle d'optimisation...")
        
        M = Model("DesagregationMIP")
        
        # =================================================================
        # VARIABLES DE DÉCISION
        # =================================================================
        
        # Charge de base : p_BASE_i_d(t) >= 0
        p_BASE = M.variable(
            "p_BASE",
            [self.n_clients, self.n_pas_temps],
            Domain.greaterThan(0.0)
        )
        
        # Variables binaires pour chaque client et appareil
        for i_idx, client in enumerate(self.donnees['I']):
            for a_idx, appareil in enumerate(self.donnees['A']):
                
                # o_a(t) : état ON/OFF
                self.variables[f'o_{client}_{appareil}'] = M.variable(
                    f'o_{client}_{appareil}',
                    self.n_pas_temps,
                    Domain.binary()
                )
                
                # s_a(t) : indicateur de démarrage
                self.variables[f's_{client}_{appareil}'] = M.variable(
                    f's_{client}_{appareil}',
                    self.n_pas_temps,
                    Domain.binary()
                )
                
                # f_a(t) : indicateur d'arrêt
                self.variables[f'f_{client}_{appareil}'] = M.variable(
                    f'f_{client}_{appareil}',
                    self.n_pas_temps,
                    Domain.binary()
                )
                
                # u_a(t) : autorisation thermique
                self.variables[f'u_{client}_{appareil}'] = M.variable(
                    f'u_{client}_{appareil}',
                    self.n_pas_temps,
                    Domain.binary()
                )
                
                # x_a_l(t) : sélection du niveau de puissance
                n_niveaux = len(self.donnees['L_a'][appareil])
                self.variables[f'x_{client}_{appareil}'] = M.variable(
                    f'x_{client}_{appareil}',
                    [self.n_pas_temps, n_niveaux],
                    Domain.binary()
                )
        
        # Variable pour régularisation L1 : |B_i_d(t)|
        B_abs = M.variable(
            "B_abs",
            [self.n_clients, self.n_pas_temps],
            Domain.greaterThan(0.0)
        )
        
        # =================================================================
        # FONCTION OBJECTIF
        # =================================================================
        
        print("Définition de la fonction objectif...")
        
        termes_objectif = []
        
        # Terme 1 : Erreur de reconstruction
        # Sum_{i,t} (p_total(t) - p_BASE(t) - Sum_{a,l} x_a_l(t))^2
        for i_idx, client in enumerate(self.donnees['I']):
            for t in range(self.n_pas_temps):
                
                # Terme de prédiction : p_total - p_BASE
                prediction = Expr.sub(
                    self.donnees['p_total_i_d'][i_idx, t],
                    p_BASE.index([i_idx, t])
                )
                
                # Soustraire les contributions des appareils
                for appareil in self.donnees['A']:
                    niveaux_puissance = self.donnees['L_a'][appareil]
                    x_var = self.variables[f'x_{client}_{appareil}']
                    
                    # Sum_l (P_l * x_a_l(t))
                    for l_idx, P_l in enumerate(niveaux_puissance):
                        prediction = Expr.sub(
                            prediction,
                            Expr.mul(P_l, x_var.index([t, l_idx]))
                        )
                
                # Carré de l'erreur
                termes_objectif.append(Expr.mul(prediction, prediction))
        
        erreur_reconstruction = Expr.add(termes_objectif)
        
        # Terme 2 : Régularisation L1 sur les variations de baseload
        # lambda1 * Sum_t |B_i_d(t)|
        
        # Définir B(t) = p_BASE(t) - p_BASE(t-1) et contraintes pour valeur absolue
        for i_idx in range(self.n_clients):
            for t in range(1, self.n_pas_temps):
                B_diff = Expr.sub(
                    p_BASE.index([i_idx, t]),
                    p_BASE.index([i_idx, t-1])
                )
                
                # B_abs >= B_diff
                M.constraint(
                    f"B_abs_pos_{i_idx}_{t}",
                    Expr.sub(B_abs.index([i_idx, t]), B_diff),
                    Domain.greaterThan(0.0)
                )
                
                # B_abs >= -B_diff
                M.constraint(
                    f"B_abs_neg_{i_idx}_{t}",
                    Expr.add(B_abs.index([i_idx, t]), B_diff),
                    Domain.greaterThan(0.0)
                )
        
        regularisation_l1 = Expr.mul(
            self.parametres['lambda1'],
            Expr.sum(B_abs)
        )
        
        # Objectif combiné
        objectif_total = Expr.add(erreur_reconstruction, regularisation_l1)
        
        M.objective("cout_total", ObjectiveSense.Minimize, objectif_total)
        
        # =================================================================
        # CONTRAINTES
        # =================================================================
        
        print("Ajout des contraintes...")
        
        for i_idx, client in enumerate(self.donnees['I']):
            for appareil in self.donnees['A']:
                
                o_var = self.variables[f'o_{client}_{appareil}']
                s_var = self.variables[f's_{client}_{appareil}']
                f_var = self.variables[f'f_{client}_{appareil}']
                u_var = self.variables[f'u_{client}_{appareil}']
                x_var = self.variables[f'x_{client}_{appareil}']
                n_niveaux = len(self.donnees['L_a'][appareil])
                
                for t in range(self.n_pas_temps):
                    
                    # (3.2) Contrainte d'autorisation thermique
                    # o_a(t) <= u_a(t)
                    M.constraint(
                        f"autoris_therm_{client}_{appareil}_{t}",
                        Expr.sub(o_var.index(t), u_var.index(t)),
                        Domain.lessThan(0.0)
                    )
                    
                    # T_ext(t) - T_ext_MAX <= M * u_a(t)
                    M.constraint(
                        f"temp_max_{client}_{appareil}_{t}",
                        Expr.sub(
                            self.donnees['T_ext'][t] - self.parametres['T_ext_MAX'],
                            Expr.mul(self.parametres['M'], u_var.index(t))
                        ),
                        Domain.lessThan(0.0)
                    )
                    
                    # T_ext(t) - T_ext_MIN >= -M * (1 - u_a(t))
                    M.constraint(
                        f"temp_min_{client}_{appareil}_{t}",
                        Expr.sub(
                            Expr.add(
                                self.donnees['T_ext'][t] - self.parametres['T_ext_MIN'],
                                Expr.mul(self.parametres['M'], u_var.index(t))
                            ),
                            self.parametres['M']
                        ),
                        Domain.greaterThan(0.0)
                    )
                    
                    # (3.3) Sélection du niveau de puissance
                    # Sum_l x_a_l(t) = o_a(t)
                    somme_niveaux = Expr.sum(x_var.slice([t, 0], [t+1, n_niveaux]))
                    M.constraint(
                        f"selection_niveau_{client}_{appareil}_{t}",
                        Expr.sub(somme_niveaux, o_var.index(t)),
                        Domain.equalsTo(0.0)
                    )
                    
                    # (3.4) Détection des cycles ON/OFF
                    # o_a(t) - o_a(t-1) = s_a(t) - f_a(t)
                    if t > 0:
                        M.constraint(
                            f"detection_cycle_{client}_{appareil}_{t}",
                            Expr.sub(
                                Expr.sub(o_var.index(t), o_var.index(t-1)),
                                Expr.sub(s_var.index(t), f_var.index(t))
                            ),
                            Domain.equalsTo(0.0)
                        )
                    
                    # s_a(t) + f_a(t) <= 1
                    M.constraint(
                        f"mutex_start_finish_{client}_{appareil}_{t}",
                        Expr.add(s_var.index(t), f_var.index(t)),
                        Domain.lessThan(1.0)
                    )
                
                # (3.5) Contrainte de durée minimale ON
                d_min = self.parametres['d_min']
                for t in range(self.n_pas_temps - d_min + 1):
                    somme_cycles = Expr.add([
                        Expr.add(s_var.index(t + n), f_var.index(t + n))
                        for n in range(d_min)
                    ])
                    M.constraint(
                        f"duree_min_ON_{client}_{appareil}_{t}",
                        somme_cycles,
                        Domain.lessThan(1.0)
                    )
                
                # (3.6) Contraintes de duty-cycle
                # Période de pointe
                if len(self.donnees['T_POINTE']) > 0:
                    indices_pointe = [int(t) for t in self.donnees['T_POINTE'] 
                                     if t < self.n_pas_temps]
                    if indices_pointe:
                        somme_pointe = Expr.add([o_var.index(int(t)) for t in indices_pointe])
                        M.constraint(
                            f"duty_cycle_pointe_{client}_{appareil}",
                            somme_pointe,
                            Domain.greaterThan(len(indices_pointe) / self.parametres['x_DUTY'])
                        )
                
                # Période hors-pointe
                if len(self.donnees['T_HORS_POINTE']) > 0:
                    indices_hors_pointe = [int(t) for t in self.donnees['T_HORS_POINTE'] 
                                          if t < self.n_pas_temps]
                    if indices_hors_pointe:
                        somme_hors_pointe = Expr.add([o_var.index(int(t)) 
                                                      for t in indices_hors_pointe])
                        M.constraint(
                            f"duty_cycle_hors_pointe_{client}_{appareil}",
                            somme_hors_pointe,
                            Domain.lessThan(len(indices_hors_pointe) / self.parametres['x_DUTY_prime'])
                        )
        
        print("Modèle construit avec succès!")
        
        self.modele = M
        self.p_BASE = p_BASE
        
        return M
    
    def resoudre(self):
        """
        Résout le problème d'optimisation.
        
        Retourne :
        ---------
        dict : Dictionnaire contenant tous les résultats
        """
        if self.modele is None:
            self.construire_modele()
        
        print("\n" + "="*70)
        print("Résolution du problème d'optimisation avec MOSEK...")
        print("="*70)
        
        # Résoudre le modèle
        self.modele.solve()
        
        # Vérifier le statut de la solution
        statut = self.modele.getPrimalSolutionStatus()
        print(f"Statut de la solution : {statut}")
        
        # Extraire les résultats
        self.resultats = self._extraire_resultats()
        
        print("="*70)
        print("Optimisation terminée!")
        print("="*70 + "\n")
        
        return self.resultats
    
    def _extraire_resultats(self):
        """
        Extrait et organise les résultats du modèle résolu.
        
        Retourne :
        ---------
        dict : Dictionnaire contenant toutes les variables de solution
        """
        print("Extraction des résultats...")
        
        resultats = {
            'p_BASE': {},
            'o': {},
            'x': {},
            's': {},
            'f': {},
            'u': {},
            'valeur_objectif': self.modele.primalObjValue()
        }
        
        # Extraire la charge de base
        valeurs_p_BASE = self.p_BASE.level().reshape(self.n_clients, self.n_pas_temps)
        for i_idx, client in enumerate(self.donnees['I']):
            resultats['p_BASE'][client] = valeurs_p_BASE[i_idx, :]
        
        # Extraire les variables des appareils
        for i_idx, client in enumerate(self.donnees['I']):
            for appareil in enumerate(self.donnees['A']):
                
                cle = f"{client}_{appareil}"
                
                # États ON/OFF
                resultats['o'][cle] = self.variables[f'o_{client}_{appareil}'].level()
                
                # Indicateurs de démarrage
                resultats['s'][cle] = self.variables[f's_{client}_{appareil}'].level()
                
                # Indicateurs d'arrêt
                resultats['f'][cle] = self.variables[f'f_{client}_{appareil}'].level()
                
                # Autorisations thermiques
                resultats['u'][cle] = self.variables[f'u_{client}_{appareil}'].level()
                
                # Sélection des niveaux de puissance
                n_niveaux = len(self.donnees['L_a'][appareil])
                x_vals = self.variables[f'x_{client}_{appareil}'].level()
                resultats['x'][cle] = x_vals.reshape(self.n_pas_temps, n_niveaux)
        
        print("Résultats extraits avec succès!")
        return resultats
