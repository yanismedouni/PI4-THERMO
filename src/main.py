"""
Point d'entrée principal du projet.
Lance dans l'ordre :
  1. csv/csv_grid.py   → génère output/grid_interp.csv
  2. csv/csv_solar.py  → génère output/solar_interp.csv
  3. src/pretraitement.py → génère output/processed_energy_data.csv
"""

import sys
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────
# Chemins du projet  (main.py est dans src/)
# ─────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent          # .../src/
PROJECT    = SRC_DIR.parent                 # .../  (racine du projet)
CSV_DIR    = PROJECT / "csv"
OUTPUT_DIR = PROJECT / "output"
DATA_DIR   = PROJECT / "data"
CSV_OUTPUT_DIR = PROJECT / "csv" /"output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA   = DATA_DIR   / "15minute_data_austin.csv"
GRID_OUT   = CSV_OUTPUT_DIR / "grid_interp.csv"
SOLAR_OUT  = CSV_OUTPUT_DIR / "solar_interp.csv"
FINAL_OUT  = OUTPUT_DIR / "processed_energy_data.csv"


# ─────────────────────────────────────────────
# Utilitaire : lancer un script dans son dossier
# ─────────────────────────────────────────────
def run_script(script_path: Path, description: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  Script : {script_path.relative_to(PROJECT)}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),   # exécuté depuis son propre dossier
        capture_output=False,          # affiche stdout/stderr en direct
    )

    if result.returncode != 0:
        print(f"\n Échec de {script_path.name} (code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\n {script_path.name} terminé avec succès.")


# ─────────────────────────────────────────────
# Patch dynamique des chemins dans csv_grid / csv_solar
# (évite de modifier ces fichiers à la main)
# ─────────────────────────────────────────────
def _patch_and_run_csv_script(script_path: Path, output_path: Path, description: str) -> None:
    """
    Lit le script, remplace les chemins relatifs/absolus codés en dur
    par les chemins réels du projet, puis l'exécute via exec().
    """
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  Script : {script_path.relative_to(PROJECT)}")
    print(f"{'=' * 60}")

    source = script_path.read_text(encoding="utf-8")

    # Remplace le chemin d'entrée (toujours le même fichier brut)
    source = source.replace(
        '"../data/15minute_data_austin.csv"',
        repr(str(RAW_DATA))
    )

    # Remplace n'importe quel chemin de sortie codé en dur
    # (csv_grid.py a un chemin Windows absolu ; csv_solar.py a un chemin relatif)
    for old_out in [
        '"C:/Users/yanis/Documents/PI4-THERMO/csv/output/grid_interp.csv"',
        '"../csv/output/grid_interp.csv"',
        '"../csv/output/solar_interp.csv"',
    ]:
        source = source.replace(old_out, repr(str(output_path)))

    exec(compile(source, str(script_path), "exec"), {"__file__": str(script_path)})
    print(f"\n✔ {script_path.name} terminé avec succès.")


# ─────────────────────────────────────────────
# Étape 3 : prétraitement
# ─────────────────────────────────────────────
def run_pretraitement() -> None:
    print(f"\n{'=' * 60}")
    print("  ÉTAPE 3 — Prétraitement final")
    print(f"  Script : src/pretraitement.py")
    print(f"{'=' * 60}")

    # Import direct (même dossier que main.py)
    from pretraitement import process_energy_data

    process_energy_data(
        grid_file=str(GRID_OUT),
        solar_file=str(SOLAR_OUT),
        raw_file=str(RAW_DATA),
        output_file=str(FINAL_OUT),
        ev_threshold_kw=3.0,
    )

    print(f"\n✔ Prétraitement terminé — résultat : {FINAL_OUT.relative_to(PROJECT)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  PIPELINE DE TRAITEMENT DES DONNÉES ÉNERGÉTIQUES")
    print("=" * 60)

    # Vérification préalable
    if not RAW_DATA.exists():
        print(f"\n✗ Fichier brut introuvable : {RAW_DATA}")
        print("  Placez 15minute_data_austin.csv dans le dossier data/")
        sys.exit(1)

    # ── Étape 1 : interpolation grille ──
    _patch_and_run_csv_script(
        script_path=CSV_DIR / "csv_grid.py",
        output_path=GRID_OUT,
        description="ÉTAPE 1 — Interpolation consommation réseau (grid)",
    )

    # ── Étape 2 : interpolation solaire ──
    _patch_and_run_csv_script(
        script_path=CSV_DIR / "csv_solar.py",
        output_path=SOLAR_OUT,
        description="ÉTAPE 2 — Interpolation production solaire",
    )

    # ── Étape 3 : prétraitement final ──
    run_pretraitement()

    print("\n" + "=" * 60)
    print("  Pipeline terminé avec succès.")
    print(f"  Résultat final : {FINAL_OUT.relative_to(PROJECT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()