"""
Script principal pour tester le module donnees.py
"""


from donnees import (
    charger_donnees_csv,
    creer_donnees_exemple,
    valider_donnees,
    creer_ensembles
)

from pathlib import Path

SRC_DIR = Path(__file__).parent
CSV_PATH = SRC_DIR / "donnees_finales.csv"

donnees_csv = charger_donnees_csv(
    CSV_PATH,  # <-- fichier CSV ici
    liste_appareils=['Climatisation', 'Chauffe-eau'],
    niveaux_puissance_defaut=[0.0, 0.3, 0.6, 1.0, 1.5]
)

def main():
    print("=" * 70)
    print("TEST DU MODULE DONNEES")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Définition des chemins (ROBUSTE)
    # ------------------------------------------------------------------
    SRC_DIR = Path(__file__).parent
    CSV_PATH = SRC_DIR / "donnees_finales.csv"

    # Test 1 : Chargement depuis CSV
    print("\nTest 1 : Chargement depuis CSV")
    print("-" * 70)
    try:
        print(f"Chargement des données depuis : {CSV_PATH}")

        donnees_csv = charger_donnees_csv(
            CSV_PATH,
            liste_appareils=['Climatisation', 'Chauffe-eau'],
            niveaux_puissance_defaut=[0.0, 0.3, 0.6, 1.0, 1.5]
        )

        print("\nRésumé des données chargées :")
        print(f"  - Nombre de pas de temps : {len(donnees_csv['T'])}")
        print(f"  - Appareils : {donnees_csv['A']}")
        print(f"  - Clients : {donnees_csv['I']}")
        print(f"  - Pas de temps en pointe : {len(donnees_csv['T_POINTE'])}")
        print(f"  - Pas de temps hors-pointe : {len(donnees_csv['T_HORS_POINTE'])}")
        print(f"  - Forme p_total_i_d : {donnees_csv['p_total_i_d'].shape}")
        print(f"  - Longueur T_ext : {len(donnees_csv['T_ext'])}")

        # Validation
        print("\nValidation des données CSV...")
        valider_donnees(donnees_csv)

    except FileNotFoundError:
        print("  Avertissement : fichier 'donnees_finales.csv' introuvable")
        print("  → Placez-le dans le dossier src/")
    except Exception as e:
        print(f"  Erreur : {e}")

    # Test 2 : Création de données exemple
    print("\n\nTest 2 : Création de données exemple (synthétiques)")
    print("-" * 70)
    try:
        donnees_synth = creer_donnees_exemple(
            n_pas_temps=96,
            n_clients=1,
            n_appareils=2
        )

        print("\nDonnées synthétiques créées")
        print(f"  - Nombre de pas de temps : {len(donnees_synth['T'])}")
        print(f"  - Appareils : {donnees_synth['A']}")
        print(f"  - Clients : {donnees_synth['I']}")
        print(f"  - Pas de temps en pointe : {len(donnees_synth['T_POINTE'])}")
        print(f"  - Forme p_total_i_d : {donnees_synth['p_total_i_d'].shape}")

        print("\nValidation des données synthétiques...")
        valider_donnees(donnees_synth)

    except Exception as e:
        print(f"  Erreur : {e}")

    # Test 3 : Test de creer_ensembles
    print("\n\nTest 3 : Fonction creer_ensembles")
    print("-" * 70)
    try:
        T, T_POINTE, T_HORS_POINTE = creer_ensembles(
            n_pas_temps=96,
            heures_pointe=[(8, 12), (18, 22)]
        )

        print("\nEnsembles créés")
        print(f"  - Total : {len(T)} pas de temps")
        print(f"  - Pointe : {len(T_POINTE)} pas de temps")
        print(f"  - Hors-pointe : {len(T_HORS_POINTE)} pas de temps")

    except Exception as e:
        print(f"  Erreur : {e}")

    print("\n" + "=" * 70)
    print("Tests terminés")
    print("=" * 70)


if __name__ == "__main__":
    main()
