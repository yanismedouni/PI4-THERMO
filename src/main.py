"""
Point d'entrée principal du projet.
Lance dans l'ordre pour chaque région :
  1. csv/csv_grid.py      → génère csv/output/<region>_grid_interp.csv
  2. csv/csv_solar.py     → génère csv/output/<region>_solar_interp.csv
  3. src/pretraitement.py → génère output/<region>_processed_energy_data.csv
  4. src/temperature.py   → remplace la colonne 'temp' par open-meteo (sur place)

Régions traitées :
  - austin      → data/15minute_data_austin.csv
  - california  → data/15minute_data_california.csv
  - newyork     → data/15minute_data_newyork.csv
  - puertorico  → data/pr_realpower_09-2023_15min.csv
                  data/pr_realpower_10-2023_15min.csv
                  data/pr_realpower_11-2023_15min.csv
                  (merged automatically before processing)

Fichiers open-meteo attendus dans data/ :
  - open-meteo-austin.csv
  - open-meteo-california.csv
  - open-meteo-newyork.csv
  - open-meteo-puertorico.csv
"""

import sys
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Chemins du projet  (main.py est dans src/)
# ─────────────────────────────────────────────
SRC_DIR        = Path(__file__).parent          # .../src/
PROJECT        = SRC_DIR.parent                 # racine du projet
CSV_DIR        = PROJECT / "csv"
OUTPUT_DIR     = PROJECT / "output"
DATA_DIR       = PROJECT / "data"
CSV_OUTPUT_DIR = PROJECT / "csv" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Définition des régions à traiter
# ─────────────────────────────────────────────
REGIONS = {
    "austin": {
        "raw": DATA_DIR / "15minute_data_austin.csv",
    },
    "california": {
        "raw": DATA_DIR / "15minute_data_california.csv",
    },
    "newyork": {
        "raw": DATA_DIR / "15minute_data_newyork.csv",
    },
    "puertorico": {
        # These three files will be merged automatically
        "raw_parts": [
            DATA_DIR / "pr_realpower_09-2023_15min.csv",
            DATA_DIR / "pr_realpower_10-2023_15min.csv",
            DATA_DIR / "pr_realpower_11-2023_15min.csv",
        ],
        # Merged file written here before processing
        "raw": DATA_DIR / "pr_merged.csv",
    },
}


# ─────────────────────────────────────────────
# Puerto Rico: merge 3 monthly files into one
# ─────────────────────────────────────────────
def merge_puerto_rico_files(parts: list[Path], output_path: Path) -> None:
    print(f"  → Fusion des fichiers Puerto Rico...")

    frames = []
    for p in parts:
        if not p.exists():
            print(f"  ✗ File not found: {p}")
            sys.exit(1)
        frames.append(pd.read_csv(p))

    df_merged = pd.concat(frames, ignore_index=True)

    if "local_15min" in df_merged.columns:
        df_merged = df_merged.sort_values(["dataid", "local_15min"]).reset_index(drop=True)

    df_merged = df_merged.drop_duplicates(subset=["dataid", "local_15min"], keep="first")
    df_merged.to_csv(output_path, index=False)
    print(f"  ✔ Fichier fusionné → {output_path.name}  ({len(df_merged):,} rows)")


# ─────────────────────────────────────────────
# Patch & run csv_grid.py / csv_solar.py
# ─────────────────────────────────────────────
def _patch_and_run_csv_script(
    script_path: Path,
    raw_data: Path,
    output_path: Path,
    description: str,
) -> None:
    """
    Read the CSV helper script, replace hard-coded paths with the real ones,
    then execute it via exec() so it runs in the current process.
    """
    print(f"  → {description}...")

    source = script_path.read_text(encoding="utf-8")

    # Replace any hard-coded input path
    for old_in in [
        '"../data/15minute_data_austin.csv"',
        '"../data/15minute_data_california.csv"',
        '"../data/15minute_data_newyork.csv"',
        '"../data/pr_merged.csv"',
    ]:
        source = source.replace(old_in, repr(str(raw_data)))

    # Replace any hard-coded output path
    for old_out in [
        '"C:/Users/yanis/Documents/PI4-THERMO/csv/output/grid_interp.csv"',
        '"../csv/output/grid_interp.csv"',
        '"../csv/output/solar_interp.csv"',
    ]:
        source = source.replace(old_out, repr(str(output_path)))

    exec(compile(source, str(script_path), "exec"), {"__file__": str(script_path)})


# ─────────────────────────────────────────────
# Étape 3 : prétraitement
# ─────────────────────────────────────────────
def run_pretraitement(
    region: str,
    raw: Path,
    grid_out: Path,
    solar_out: Path,
    final_out: Path,
) -> None:
    print(f"  → Prétraitement final...")

    from pretraitement import process_energy_data

    process_energy_data(
        grid_file=str(grid_out),
        solar_file=str(solar_out),
        raw_file=str(raw),
        output_file=str(final_out),
        ev_threshold_kw=3.0,
    )


# ─────────────────────────────────────────────
# Étape 4 : température open-meteo
# ─────────────────────────────────────────────
def run_temperature(region: str, final_out: Path) -> None:
    print(f"  → Fusion température open-meteo...")

    from temperature import merge_temperature_file

    merge_temperature_file(
        csv_path=final_out,
        region=region,
        data_dir=DATA_DIR,
        output_path=final_out,
    )


# ─────────────────────────────────────────────
# Pipeline pour une région
# ─────────────────────────────────────────────
def process_region(region: str, config: dict) -> None:
    print(f"\n[ {region.upper()} ]")

    raw: Path = config["raw"]

    if "raw_parts" in config:
        merge_puerto_rico_files(config["raw_parts"], raw)

    if not raw.exists():
        print(f"  ✗ Fichier brut introuvable : {raw}")
        sys.exit(1)

    grid_out  = CSV_OUTPUT_DIR / f"{region}_grid_interp.csv"
    solar_out = CSV_OUTPUT_DIR / f"{region}_solar_interp.csv"
    final_out = OUTPUT_DIR     / f"{region}_processed_energy_data.csv"

    _patch_and_run_csv_script(
        script_path=CSV_DIR / "csv_grid.py",
        raw_data=raw,
        output_path=grid_out,
        description="Interpolation grid",
    )

    _patch_and_run_csv_script(
        script_path=CSV_DIR / "csv_solar.py",
        raw_data=raw,
        output_path=solar_out,
        description="Interpolation solaire",
    )

    run_pretraitement(region, raw, grid_out, solar_out, final_out)
    run_temperature(region, final_out)

    print(f"  ✔ {region} terminé → {final_out.name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    print("Pipeline de traitement des données énergétiques")
    print(f"Régions : {', '.join(REGIONS.keys())}\n")

    for region, config in REGIONS.items():
        process_region(region, config)

    print("\n✔ Pipeline terminé. Résultats dans output/")


if __name__ == "__main__":
    main()