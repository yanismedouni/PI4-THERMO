"""
Point d'entrée principal du projet.
Lance dans l'ordre pour chaque région :
  1. csv/csv_grid.py   → génère csv/output/<region>_grid_interp.csv
  2. csv/csv_solar.py  → génère csv/output/<region>_solar_interp.csv
  3. src/pretraitement.py → génère output/<region>_processed_energy_data.csv

Régions traitées :
  - austin      → data/15minute_data_austin.csv
  - california  → data/15minute_data_california.csv
  - newyork     → data/15minute_data_newyork.csv
  - puertorico  → data/pr_realpower_09-2023_15min.csv
                  data/pr_realpower_10-2023_15min.csv
                  data/pr_realpower_11-2023_15min.csv
                  (merged automatically before processing)
"""

import sys
import pandas as pd
import subprocess
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
    """
    Concatenate the three Puerto Rico monthly CSVs into a single file.
    Duplicate rows (same dataid + local_15min) are dropped, keeping the first.
    """
    print(f"\n  Merging {len(parts)} Puerto Rico files...")

    frames = []
    for p in parts:
        if not p.exists():
            print(f"  ✗ File not found: {p}")
            sys.exit(1)
        df = pd.read_csv(p)
        print(f"    Loaded {len(df):>8,} rows from {p.name}")
        frames.append(df)

    df_merged = pd.concat(frames, ignore_index=True)

    # Sort chronologically and drop duplicates
    if "local_15min" in df_merged.columns:
        df_merged = df_merged.sort_values(["dataid", "local_15min"]).reset_index(drop=True)

    before = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=["dataid", "local_15min"], keep="first")
    dropped = before - len(df_merged)
    if dropped:
        print(f"    Dropped {dropped:,} duplicate rows.")

    df_merged.to_csv(output_path, index=False)
    print(f"  ✔ Merged Puerto Rico data saved → {output_path.name}  ({len(df_merged):,} rows)")


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
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  Script : {script_path.relative_to(PROJECT)}")
    print(f"{'=' * 60}")

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
    print(f"\n✔ {script_path.name} terminé avec succès.")


# ─────────────────────────────────────────────
# Étape 3 : prétraitement
# ─────────────────────────────────────────────
def run_pretraitement(region: str, raw: Path, grid_out: Path, solar_out: Path, final_out: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"  PRÉTRAITEMENT — {region.upper()}")
    print(f"{'=' * 60}")

    from pretraitement import process_energy_data

    process_energy_data(
        grid_file=str(grid_out),
        solar_file=str(solar_out),
        raw_file=str(raw),
        output_file=str(final_out),
        ev_threshold_kw=3.0,
    )

    print(f"\n✔ Prétraitement terminé — résultat : {final_out.relative_to(PROJECT)}")


# ─────────────────────────────────────────────
# Pipeline pour une région
# ─────────────────────────────────────────────
def process_region(region: str, config: dict) -> None:
    print(f"\n{'#' * 60}")
    print(f"#  RÉGION : {region.upper()}")
    print(f"{'#' * 60}")

    raw: Path = config["raw"]

    # ── Puerto Rico: merge monthly files first ──
    if "raw_parts" in config:
        merge_puerto_rico_files(config["raw_parts"], raw)

    # ── Verify raw file exists ──
    if not raw.exists():
        print(f"\n✗ Fichier brut introuvable : {raw}")
        sys.exit(1)

    grid_out  = CSV_OUTPUT_DIR / f"{region}_grid_interp.csv"
    solar_out = CSV_OUTPUT_DIR / f"{region}_solar_interp.csv"
    final_out = OUTPUT_DIR     / f"{region}_processed_energy_data.csv"

    # ── Step 1: grid interpolation ──
    _patch_and_run_csv_script(
        script_path=CSV_DIR / "csv_grid.py",
        raw_data=raw,
        output_path=grid_out,
        description=f"ÉTAPE 1 — Interpolation grid ({region})",
    )

    # ── Step 2: solar interpolation ──
    _patch_and_run_csv_script(
        script_path=CSV_DIR / "csv_solar.py",
        raw_data=raw,
        output_path=solar_out,
        description=f"ÉTAPE 2 — Interpolation solaire ({region})",
    )

    # ── Step 3: preprocessing ──
    run_pretraitement(region, raw, grid_out, solar_out, final_out)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  PIPELINE DE TRAITEMENT DES DONNÉES ÉNERGÉTIQUES")
    print(f"  Régions : {', '.join(REGIONS.keys())}")
    print("=" * 60)

    for region, config in REGIONS.items():
        process_region(region, config)

    print("\n" + "=" * 60)
    print("  Pipeline terminé avec succès.")
    print(f"  Résultats dans : {OUTPUT_DIR.relative_to(PROJECT)}/")
    for region in REGIONS:
        final = OUTPUT_DIR / f"{region}_processed_energy_data.csv"
        print(f"    → {final.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()