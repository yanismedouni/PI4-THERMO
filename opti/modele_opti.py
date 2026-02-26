"""
Modele d'optimisation MICQP pour la desagregation des TCLs.
Base sur la formulation du cahier section 6.5.4.1.1.4.

Supporte deux appareils simultanes :
  - climatisation : air1 + air2 + air3
  - chauffage     : furnace1 + furnace2 + heater1 + heater2 + heater3

Les parametres thermiques et duty-cycle sont lus par appareil
depuis la structure 'thermique' et 'duty_cycle' de parametres.py.

CORRECTIONS (v3) :
  - eq17/18 : ajout de eq18 pour climatisation ET chauffage.
  - eq15/16 : reactivation des contraintes duty-cycle, conditionnees
              aux pas de temps thermiquement autorises pour eviter
              tout conflit avec les contraintes thermiques hors-saison.
  - Vectorisation des contraintes eq9/10/11/12/13/17/18 pour accelerer
              la construction du modele.
"""

import cvxpy as cp
import numpy as np


def creer_modele_optimisation(donnees, parametres):
    """
    Cree le modele d'optimisation MICQP.

    donnees :
        'T'       : int   — nombre de pas de temps
        'P_total' : array — puissance agregee mesuree (kW)
        'T_ext'   : array — temperature exterieure (degC)
        'heures'  : array — heure locale pour chaque pas (0-23)

    parametres : voir parametres.py
    """
    print("=" * 70)
    print("CREATION DU MODELE D'OPTIMISATION")
    print("=" * 70)

    # ── Donnees ───────────────────────────────────────────────────────────────
    T       = donnees['T']
    P_total = donnees['P_total']
    T_ext   = donnees['T_ext']
    heures  = donnees['heures']

    # ── Parametres globaux ────────────────────────────────────────────────────
    lambda1           = parametres['lambda1']
    d_min             = parametres['d_min']
    M                 = parametres['M']
    appareils         = parametres['appareils']
    niveaux_puissance = parametres['niveaux_puissance']
    heures_pointe     = parametres['heures_pointe']
    thermique_par_app = parametres['thermique']
    duty_par_app      = parametres['duty_cycle']

    print(f"\n  Pas de temps : {T}")
    print(f"  Appareils    : {appareils}")
    print(f"  lambda1      : {lambda1}  |  d_min : {d_min} pas ({d_min*15} min)")
    for app in appareils:
        th   = thermique_par_app[app]
        duty = duty_par_app[app]
        print(f"  [{app}] niveaux={niveaux_puissance[app]} kW"
              f" | mode={th['mode']}"
              f" | T_MIN={th['T_ext_MIN']} T_MAX={th['T_ext_MAX']}"
              f" | DUTY={duty['x_DUTY']}/{duty['x_DUTY_prime']}")

    # =========================================================================
    # VARIABLES D'OPTIMISATION
    # =========================================================================
    print("\nCreation des variables...")

    # Baseload continu >= 0  (eq 7)
    p_BASE = cp.Variable(T, nonneg=True)

    variables_appareils = {}
    for appareil in appareils:
        niveaux = niveaux_puissance[appareil]
        L_a     = len(niveaux)

        variables_appareils[appareil] = {
            'o':      cp.Variable(T,        boolean=True),   # eq (1) ON/OFF
            'x':      cp.Variable((T, L_a), boolean=True),   # eq (2) niveau l
            's':      cp.Variable(T,        boolean=True),   # eq (3) start
            'f':      cp.Variable(T,        boolean=True),   # eq (4) finish
            'u':      cp.Variable(T,        boolean=True),   # eq (6) autorisation thermique
            'niveaux': niveaux,
        }

    n_bin = sum(
        v['o'].size + v['x'].size + v['s'].size + v['f'].size + v['u'].size
        for v in variables_appareils.values()
    )
    print(f"  Variables binaires  : {n_bin}")
    print(f"  Variables continues : {p_BASE.size}")

    # =========================================================================
    # FONCTION OBJECTIF
    # =========================================================================
    print("\nConstruction de la fonction objectif...")

    # Puissance estimee par appareil : x_a_l @ niveaux => vecteur (T,)
    P_app_list = []
    for appareil in appareils:
        va      = variables_appareils[appareil]
        niveaux = np.array(va['niveaux'], dtype=float)
        P_app_list.append(va['x'] @ niveaux)

    P_appareils_total = cp.sum(P_app_list, axis=0) if len(P_app_list) > 1 else P_app_list[0]

    # Erreur quadratique de reconstruction
    erreur            = P_total - p_BASE - P_appareils_total
    terme_quadratique = cp.sum_squares(erreur)

    # Regularisation L1 sur les variations du baseload (eq 8)
    terme_L1 = lambda1 * cp.norm1(cp.diff(p_BASE))

    objectif = cp.Minimize(terme_quadratique + terme_L1)

    # =========================================================================
    # CONTRAINTES
    # =========================================================================
    print("Construction des contraintes...")
    contraintes = []

    # Periodes de pointe et hors-pointe (communes a tous les appareils)
    indices_pointe      = [i for i in range(T) if heures[i] in heures_pointe]
    indices_hors_pointe = [i for i in range(T) if heures[i] not in heures_pointe]

    for appareil in appareils:
        va     = variables_appareils[appareil]
        o_a    = va['o']
        x_a_l  = va['x']
        s_a    = va['s']
        f_a    = va['f']
        u_a    = va['u']

        th     = thermique_par_app[appareil]
        mode   = th['mode']
        T_MIN  = th['T_ext_MIN']
        T_MAX  = th['T_ext_MAX']

        duty          = duty_par_app[appareil]
        x_DUTY        = duty['x_DUTY']
        x_DUTY_prime  = duty['x_DUTY_prime']

        # ── Pas de transition au premier pas ──────────────────────────────────
        contraintes.append(s_a[0] == 0)
        contraintes.append(f_a[0] == 0)

        # ── eq9 : o_a(t) <= u_a(t) — vectorise ───────────────────────────────
        contraintes.append(o_a <= u_a)

        # ── eq10 : sum_l x_a_l[t,l] == 1 — vectorise ─────────────────────────
        contraintes.append(cp.sum(x_a_l, axis=1) == 1)

        # ── eq11 : sum_{l>0} x_a_l[t,l] == o_a(t) — vectorise ───────────────
        contraintes.append(cp.sum(x_a_l[:, 1:], axis=1) == o_a)

        # ── eq17/18 : contraintes thermiques Big-M — vectorise ────────────────
        #
        # CLIMATISATION :
        #   3 zones :
        #   T_ext < T_MIN           => u_a force a 0 (clim interdite)
        #   T_MIN <= T_ext <= T_MAX => u_a libre (optimiseur decide)
        #   T_ext > T_MAX           => u_a force a 1 (clim toujours autorisee)
        #
        # CHAUFFAGE (logique inverse) :
        #   T_ext > T_MAX           => u_a force a 0 (chauffage interdit)
        #   T_MIN <= T_ext <= T_MAX => u_a libre (optimiseur decide)
        #   T_ext < T_MIN           => u_a force a 1 (chauffage toujours autorise)

        if mode == 'climatisation':
            contraintes.append(T_ext - T_MAX <= M * u_a)            # eq17
            contraintes.append(T_ext - T_MIN >= -M * (1 - u_a))     # eq18
        else:  # chauffage
            contraintes.append(T_ext - T_MAX <= M * (1 - u_a))      # eq17
            contraintes.append(T_ext - T_MIN >= -M * u_a)           # eq18

        # ── eq12/13 : transitions ON/OFF — vectorise ──────────────────────────
        # eq12 : o_a(t) - o_a(t-1) = s_a(t) - f_a(t)
        contraintes.append(o_a[1:] - o_a[:-1] == s_a[1:] - f_a[1:])
        # eq13 : s_a(t) + f_a(t) <= 1
        contraintes.append(s_a[1:] + f_a[1:] <= 1)

        # ── eq14 : duree minimale de cycle ────────────────────────────────────
        #if d_min > 1:
        #    for t in range(T - d_min + 1):
        #        contraintes.append(
        #            cp.sum(s_a[t:t + d_min]) + cp.sum(f_a[t:t + d_min]) <= 1
        #        )

        # ── eq15/16 : duty-cycle conditionne a la temperature ─────────────────
        #
        # On filtre les indices ou l'appareil est thermiquement autorisable
        # pour eviter tout conflit avec les contraintes thermiques hors-saison.
        #
        # Ex: en hiver, si aucun pas de pointe n'a T_ext >= T_MIN pour la clim,
        # les contraintes duty-cycle ne sont pas ajoutees => pas de conflit.

        if mode == 'climatisation':
            pointe_ok      = [i for i in indices_pointe      if T_ext[i] >= T_MIN]
            hors_pointe_ok = [i for i in indices_hors_pointe if T_ext[i] >= T_MIN]
        else:  # chauffage
            pointe_ok      = [i for i in indices_pointe      if T_ext[i] <= T_MAX]
            hors_pointe_ok = [i for i in indices_hors_pointe if T_ext[i] <= T_MAX]

        print(f"  [{appareil}] pas pointe autorises      : "
              f"{len(pointe_ok)}/{len(indices_pointe)}")
        print(f"  [{appareil}] pas hors-pointe autorises : "
              f"{len(hors_pointe_ok)}/{len(indices_hors_pointe)}")

        # eq15 — activation minimale en pointe
        if len(pointe_ok) >= d_min:
            contraintes.append(
                cp.sum(o_a[pointe_ok]) >= len(pointe_ok) / x_DUTY
            )
            print(f"  [{appareil}] eq15 activee : ON >= "
                  f"{len(pointe_ok) / x_DUTY:.1f} pas en pointe")
        else:
            print(f"  [{appareil}] eq15 desactivee : pas assez de pas autorises en pointe")

        # eq16 — activation maximale hors-pointe
        if len(hors_pointe_ok) > 0:
            contraintes.append(
                cp.sum(o_a[hors_pointe_ok]) <= len(hors_pointe_ok) / x_DUTY_prime
            )
            print(f"  [{appareil}] eq16 activee : ON <= "
                  f"{len(hors_pointe_ok) / x_DUTY_prime:.1f} pas hors-pointe")
        else:
            print(f"  [{appareil}] eq16 desactivee : aucun pas autorise hors-pointe")

    print(f"\n  Contraintes totales : {len(contraintes)}")

    # =========================================================================
    # PROBLEME
    # =========================================================================
    probleme = cp.Problem(objectif, contraintes)
    print("\n" + "=" * 70)
    print("MODELE CREE AVEC SUCCES")
    print("=" * 70)

    return {
        'probleme':  probleme,
        'variables': {
            'p_BASE':    p_BASE,
            'appareils': variables_appareils,
        },
    }


def resoudre_optimisation(modele, verbose=False):
    """Resout le probleme avec MOSEK. Retourne None si echec."""
    print("\n" + "=" * 70)
    print("RESOLUTION DU PROBLEME")
    print("=" * 70)

    probleme = modele['probleme']
    try:
        probleme.solve(
            solver=cp.MOSEK,
            verbose=verbose,
            mosek_params={
                'MSK_DPAR_MIO_TOL_REL_GAP': 0.05,  # 5% gap acceptable
                'MSK_DPAR_MIO_MAX_TIME':     240.0, # max 4 minutes
            }
        )
    except Exception as e:
        print(f"ERREUR MOSEK : {e}")
        return None

    print(f"Statut          : {probleme.status}")
    if probleme.status not in ("optimal", "optimal_inaccurate"):
        print("ATTENTION : pas de solution optimale trouvee.")
        return None
    print(f"Valeur objectif : {probleme.value:.6f}")

    variables = modele['variables']
    resultats = {
        'statut':          probleme.status,
        'valeur_optimale': probleme.value,
        'p_BASE':          variables['p_BASE'].value,
        'appareils':       {},
    }

    for appareil, va in variables['appareils'].items():
        niveaux   = np.array(va['niveaux'], dtype=float)
        x_val     = va['x'].value
        resultats['appareils'][appareil] = {
            'o':         va['o'].value,
            'x':         x_val,
            's':         va['s'].value,
            'f':         va['f'].value,
            'u':         va['u'].value,
            'niveaux':   va['niveaux'],
            'P_estimee': x_val @ niveaux,
        }

    print("=" * 70)
    print("RESOLUTION TERMINEE")
    print("=" * 70)
    return resultats