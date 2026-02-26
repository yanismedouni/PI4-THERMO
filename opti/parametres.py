"""
Configuration des parametres de l'algorithme de desagregation.
Deux appareils virtuels etudies simultanement :
  - climatisation : air1 + air2 + air3
  - chauffage     : furnace1 + furnace2 + heater1 + heater2 + heater3

Les niveaux de puissance sont fixes manuellement (constants).

CORRECTIONS (v2) :
  - lambda1 : 0.5 -> 2.0
        Le baseload etait trop libre et absorbait toute la consommation.
        Une valeur plus elevee penalise davantage ses variations et force
        l'optimiseur a utiliser la climatisation pour expliquer les pics.

  - T_ext_MAX climatisation : 28.0 -> 45.0
        Avec T_MAX=28, presque toute la saison estivale tombait dans la
        zone [T_MIN, T_MAX] ou u_a est libre. L'optimiseur mettait alors
        u_a=0 sans contrainte. En montant T_MAX a 45, toute la plage de
        temperatures estivales devient une zone ou la clim PEUT s'allumer,
        et l'optimiseur doit choisir en fonction des donnees.

  - niveaux_puissance climatisation : [0.0, 2.5] -> [0.0, 1.5, 2.5]
        La clim reelle oscille entre 0 et ~2.8 kW de facon continue.
        Un seul niveau ON force une desagregation tout-ou-rien qui ne
        correspond pas a la realite. Deux niveaux ON permettent de mieux
        capturer les regimes bas et haut de la climatisation.
"""


def obtenir_parametres_defaut():
    """
    Retourne les parametres par defaut de l'algorithme.

    Niveaux de puissance :
        - Valeurs en kW, fixes manuellement par appareil.
        - Le premier niveau doit toujours etre 0.0 (etat OFF).
        - Les niveaux suivants representent les modes ON.
        - A calibrer a partir des resultats du clustering K-means.

    Contraintes thermiques :
        Chaque appareil a ses propres seuils et son mode thermique.
        - 'climatisation' : autorisee quand T_ext est HAUTE
              u_a = 1 si T_ext >= T_ext_MIN  (assez chaud)
              u_a = 0 si T_ext <  T_ext_MIN  (trop froid, clim interdite)
        - 'chauffage' : autorise quand T_ext est BASSE
              u_a = 1 si T_ext <= T_ext_MAX  (assez froid)
              u_a = 0 si T_ext >  T_ext_MAX  (trop chaud, chauffage interdit)

    Duty-cycle :
        - x_DUTY      : sum(o_a, t in T_POINTE)      >= |T_POINTE| / x_DUTY
        - x_DUTY_prime: sum(o_a, t in T_HORS_POINTE) <= |T_HORS_POINTE| / x_DUTY_prime
        Ces parametres sont definis separement par appareil.
    """
    return {
        # Regularisation
        # Penalise les variations du baseload entre deux pas de temps consecutifs.
        # Trop faible => baseload absorbe tout, clim reste a 0.
        # Trop eleve  => baseload completement plat, faux positifs sur la clim.
        # Plage suggeree pour des donnees en kW : 1.0 a 5.0
        'lambda1': 1.5,             # CORRECTION : etait 2.0, reduit pour permettre plus de flexibilité au baseload

        # Contraintes operationnelles communes
        # d_min = 2 => duree minimale ON/OFF de 30 minutes (2 x 15 min)
        'd_min': 3,

        # Big-M : doit etre >> max(|T_ext|) pour ne pas contraindre
        # artificiellement les variables thermiques.
        'M': 1000.0,

        # Appareils etudies simultanement
        'appareils': ['climatisation'],

        # Niveaux de puissance fixes (kW)
        # Le niveau 0.0 represente l'etat OFF (doit toujours etre en premier).
        # CORRECTION climatisation : [0.0, 2.5] -> [0.0, 1.5, 2.5]
        #   Deux niveaux ON pour capturer les regimes bas et haut de la clim.
        'niveaux_puissance': {
            # air1 + air2 + air3 agrege
            'climatisation': [0.0, 0.5, 1.5, 2.5],  # ajout de 0.5 kW            # furnace1 + furnace2 + heater1 + heater2 + heater3 agrege
            'chauffage':     [0.0, 1.5],
        },

        # Parametres thermiques par appareil
        # CORRECTION T_ext_MAX climatisation : 28.0 -> 45.0
        #   Avec T_MAX=28, presque toute la saison estivale (T_ext entre 18
        #   et 28) tombait dans la zone libre ou u_a peut etre 0 sans
        #   contrainte. En montant a 45, toute la plage estivale devient une
        #   zone ou la clim peut s'allumer et l'optimiseur doit decider selon
        #   les donnees plutot que d'ignorer la clim.
        'thermique': {
            'climatisation': {
                'mode':      'climatisation',
                'T_ext_MIN': 18.0,
                'T_ext_MAX': 45.0,  # CORRECTION : etait 28.0
            },
            'chauffage': {
                'mode':      'chauffage',
                'T_ext_MIN': -5.0,
                'T_ext_MAX': 15.0,
            },
        },

        # Duty-cycle par appareil
        # x_DUTY=4       => ON >= 25% du temps en pointe
        # x_DUTY_prime=2 => ON <= 50% du temps hors-pointe
        'duty_cycle': {
            'climatisation': {
                'x_DUTY':       4.0,
                'x_DUTY_prime': 2.0,
            },
            'chauffage': {
                'x_DUTY':       4.0,
                'x_DUTY_prime': 2.0,
            },
        },

        # Heures de pointe
        'heures_pointe': list(range(16, 21)),  # 16h-20h
    }


def afficher_parametres(params):
    print("=" * 70)
    print("PARAMETRES DE L'ALGORITHME")
    print("=" * 70)
    print(f"  lambda1          : {params['lambda1']}")
    print(f"  d_min            : {params['d_min']} pas ({params['d_min']*15} min)")
    print(f"  Appareils        : {params['appareils']}")
    print(f"  Heures de pointe : {params['heures_pointe'][0]}h a {params['heures_pointe'][-1]}h")

    for app in params['appareils']:
        niveaux = params['niveaux_puissance'][app]
        therm   = params['thermique'][app]
        duty    = params['duty_cycle'][app]
        print(f"\n  [{app}]")
        print(f"    Niveaux          : {niveaux} kW")
        print(f"    Mode thermique   : {therm['mode']}")
        print(f"    T_ext_MIN        : {therm['T_ext_MIN']} C")
        print(f"    T_ext_MAX        : {therm['T_ext_MAX']} C")
        print(f"    x_DUTY           : {duty['x_DUTY']}  => ON >= {100/duty['x_DUTY']:.0f}% en pointe")
        print(f"    x_DUTY_prime     : {duty['x_DUTY_prime']}  => ON <= {100/duty['x_DUTY_prime']:.0f}% hors-pointe")

    print("=" * 70)


if __name__ == "__main__":
    params = obtenir_parametres_defaut()
    afficher_parametres(params)