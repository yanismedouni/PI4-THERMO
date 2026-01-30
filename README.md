# Algorithme de Désagrégation de Charge

Implémentation d'un algorithme d'optimisation mixte (MIP) pour la désagrégation de charge électrique.

## Structure du Projet

```
.
├── modele_optimisation.py    # Modèle d'optimisation MOSEK
├── donnees.py                 # Gestion et validation des données
├── parametres.py              # Configuration des paramètres
├── main.py                    # Script principal d'exécution
└── requirements.txt           # Dépendances Python
```

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

**Important** : MOSEK nécessite une licence (gratuite pour usage académique).

## Utilisation

```bash
python main.py
```

## Description

L'algorithme résout un problème d'optimisation pour estimer :
- La charge de base (baseload)
- Les états ON/OFF des appareils
- Les niveaux de puissance sélectionnés
- Les autorisations thermiques basées sur la température

### Données d'entrée

- `T` : Pas de temps
- `T_POINTE` : Périodes de pointe
- `T_HORS_POINTE` : Périodes hors-pointe
- `A` : Liste des appareils
- `L_a` : Niveaux de puissance par appareil
- `I` : Liste des clients
- `p_total_i_d` : Puissance agrégée mesurée
- `T_ext` : Température extérieure

### Paramètres

- `lambda1` : Régularisation L1 (lissage du baseload)
- `d_min` : Durée minimale ON
- `x_DUTY` : Duty-cycle en pointe
- `x_DUTY_prime` : Duty-cycle hors-pointe
- `T_ext_MAX` / `T_ext_MIN` : Seuils de température

## Exemple

```python
from modele_optimisation import ModeleDesagregation
from donnees import creer_donnees_exemple
from parametres import obtenir_parametres_defaut

# Créer des données
donnees = creer_donnees_exemple(n_pas_temps=48)

# Obtenir les paramètres
parametres = obtenir_parametres_defaut()

# Résoudre
modele = ModeleDesagregation(donnees, parametres)
modele.construire_modele()
resultats = modele.resoudre()
```

## Auteur

