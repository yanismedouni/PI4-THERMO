#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcul de la MAE et RMSE à partir des CSV de désagrégation.

Lit tous les fichiers resultats_desagregation_*.csv dans output/
et produit un tableau récapitulatif + un fichier Excel.

Usage :
    python src/calcul_metriques.py
    python src/calcul_metriques.py --dossier output/mes_resultats
    python src/calcul_metriques.py --rapport output/metriques.xlsx
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = PROJECT_DIR / "output"

COL_REEL    = "P_reel_clim"
COL_ESTIME  = "P_estime_clim"
COL_ON_PRED = "o_climatisation"
COL_ON_REEL = "P_reel_clim"     # binarisé avec seuil


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def calculer_metriques(csv_path: str, seuil_on: float = 0.05) -> dict:
    """
    Calcule MAE, RMSE, Energy Fraction et Norm Error
    à partir d'un CSV de désagrégation.
    """
    df = pd.read_csv(csv_path)

    colonnes_requises = [COL_REEL, COL_ESTIME, COL_ON_PRED]
    manquantes = [c for c in colonnes_requises if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {Path(csv_path).name} : {manquantes}")

    y_reel   = pd.to_numeric(df[COL_REEL],   errors="coerce").fillna(0).to_numpy()
    y_estime = pd.to_numeric(df[COL_ESTIME], errors="coerce").fillna(0).to_numpy()
    o_pred   = pd.to_numeric(df[COL_ON_PRED],errors="coerce").fillna(0).to_numpy()
    o_reel   = (y_reel > seuil_on).astype(int)

    n = len(df)

    # ── Métriques de puissance ────────────────────────────────────────
    rmse = float(np.sqrt(np.mean((y_reel - y_estime) ** 2)))
    mae  = float(np.mean(np.abs(y_reel - y_estime)))

    # ── Fraction d'énergie ────────────────────────────────────────────
    energie_reelle  = y_reel.sum()
    energie_estimee = y_estime.sum()
    energy_frac = float(energie_estimee / max(energie_reelle, 1e-9))

    # ── Erreur normalisée L1 ──────────────────────────────────────────
    norm_err = float(np.abs(y_reel - y_estime).sum() / max(np.abs(y_reel).sum(), 1e-9))

    # ── Métriques ON/OFF ─────────────────────────────────────────────
    o_pred_bin = (np.round(o_pred)).astype(int)
    tp = int(((o_reel == 1) & (o_pred_bin == 1)).sum())
    tn = int(((o_reel == 0) & (o_pred_bin == 0)).sum())
    fp = int(((o_reel == 0) & (o_pred_bin == 1)).sum())
    fn = int(((o_reel == 1) & (o_pred_bin == 0)).sum())

    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if not (np.isnan(precision) or np.isnan(recall))
          and (precision + recall) > 0
          else float("nan"))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    # ── Info sur le fichier ───────────────────────────────────────────
    # Extraire dataid et date depuis le nom du fichier
    # Format : resultats_desagregation_<dataid>_<date>_7jours.csv
    nom = Path(csv_path).stem   # ex. resultats_desagregation_3864_2015-07-02_7jours
    parties = nom.split("_")
    try:
        dataid     = parties[2]
        date_debut = parties[3]
    except IndexError:
        dataid     = "—"
        date_debut = "—"

    return {
        "Fichier"       : Path(csv_path).name,
        "Client (dataid)": dataid,
        "Date début"    : date_debut,
        "N pas"         : n,
        "RMSE (kW)"     : round(rmse, 4),
        "MAE (kW)"      : round(mae,  4),
        "Energy Frac"   : round(energy_frac, 4),
        "Norm Err"      : round(norm_err, 4),
        "Recall"        : round(recall,    4) if not np.isnan(recall)    else float("nan"),
        "Precision"     : round(precision, 4) if not np.isnan(precision) else float("nan"),
        "F1"            : round(f1,        4) if not np.isnan(f1)        else float("nan"),
        "FPR"           : round(fpr,       4) if not np.isnan(fpr)       else float("nan"),
        "TP"            : tp,
        "TN"            : tn,
        "FP"            : fp,
        "FN"            : fn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT EXCEL
# ─────────────────────────────────────────────────────────────────────────────

def sauvegarder_excel(resultats: list[dict], out_path: str) -> None:
    """Génère un rapport Excel formaté à partir des métriques calculées."""

    wb = Workbook()
    ws = wb.active
    ws.title = "Métriques"
    ws.sheet_view.showGridLines = False

    # ── Styles ────────────────────────────────────────────────────────
    BLUE_DARK  = "1F3864"
    BLUE_MID   = "2E75B6"
    BLUE_LIGHT = "D6E4F0"
    WHITE      = "FFFFFF"
    GREY       = "F2F2F2"

    def fill(c): return PatternFill("solid", start_color=c)
    def font_(bold=False, color="000000", size=10):
        return Font(name="Arial", bold=bold, color=color, size=size)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left   = Alignment(horizontal="left",   vertical="center", wrap_text=True)
    thin   = Side(style="thin", color="BBBBBB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # ── Titre ─────────────────────────────────────────────────────────
    n_cols = 16
    ws.merge_cells(f"A1:{get_column_letter(n_cols)}1")
    ws["A1"] = "THERMO — Métriques de désagrégation (Climatisation)"
    ws["A1"].font      = font_(bold=True, color=WHITE, size=13)
    ws["A1"].fill      = fill(BLUE_DARK)
    ws["A1"].alignment = center
    ws.row_dimensions[1].height = 28

    # ── En-têtes ──────────────────────────────────────────────────────
    headers = [
        "Fichier CSV", "Client", "Date début", "N pas",
        "RMSE (kW)", "MAE (kW)", "Energy Frac", "Norm Err",
        "Recall", "Precision", "F1", "FPR",
        "TP", "TN", "FP", "FN",
    ]
    ws.row_dimensions[2].height = 32
    for col_idx, h in enumerate(headers, start=1):
        cell           = ws.cell(row=2, column=col_idx, value=h)
        cell.font      = font_(bold=True, color=WHITE, size=10)
        cell.fill      = fill(BLUE_MID)
        cell.alignment = center
        cell.border    = border

    # ── Lignes de données ─────────────────────────────────────────────
    keys = [
        "Fichier", "Client (dataid)", "Date début", "N pas",
        "RMSE (kW)", "MAE (kW)", "Energy Frac", "Norm Err",
        "Recall", "Precision", "F1", "FPR",
        "TP", "TN", "FP", "FN",
    ]
    for i, row_data in enumerate(resultats, start=3):
        shade = GREY if i % 2 == 0 else WHITE
        ws.row_dimensions[i].height = 18

        for col_idx, key in enumerate(keys, start=1):
            val  = row_data.get(key, "")
            cell = ws.cell(row=i, column=col_idx, value=val)
            cell.fill      = fill(shade)
            cell.border    = border
            cell.font      = font_(size=10)
            cell.alignment = left if col_idx == 1 else center

    # ── Ligne de résumé (moyennes) ────────────────────────────────────
    if resultats:
        row_moy = len(resultats) + 3
        ws.row_dimensions[row_moy].height = 20

        ws.cell(row=row_moy, column=1, value="MOYENNE").font = font_(bold=True, color=WHITE, size=10)
        ws.cell(row=row_moy, column=1).fill      = fill(BLUE_DARK)
        ws.cell(row=row_moy, column=1).alignment = center
        ws.cell(row=row_moy, column=1).border    = border

        for col_idx, key in enumerate(keys[1:], start=2):
            vals = [r[key] for r in resultats
                    if isinstance(r.get(key), (int, float))
                    and not np.isnan(float(r[key]))]
            val  = round(float(np.mean(vals)), 4) if vals else "—"
            cell = ws.cell(row=row_moy, column=col_idx, value=val)
            cell.fill      = fill(BLUE_LIGHT)
            cell.border    = border
            cell.font      = font_(bold=True, size=10)
            cell.alignment = center

    # ── Largeurs colonnes ─────────────────────────────────────────────
    col_widths = [42, 10, 12, 8, 10, 10, 12, 10, 10, 11, 10, 10, 7, 7, 7, 7]
    for i, w in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.freeze_panes = "A3"

    wb.save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcul MAE / RMSE à partir des CSV de désagrégation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python src/calcul_metriques.py
  python src/calcul_metriques.py --dossier output/california
  python src/calcul_metriques.py --rapport output/metriques_final.xlsx
  python src/calcul_metriques.py --seuil_on 0.10
        """,
    )
    parser.add_argument(
        "--dossier", type=str, default=str(OUTPUT_DIR),
        help="Dossier contenant les CSV de désagrégation (défaut : output/)",
    )
    parser.add_argument(
        "--rapport", type=str,
        default=str(OUTPUT_DIR / "metriques_desagregation.xlsx"),
        help="Chemin du rapport Excel de sortie (défaut : output/metriques_desagregation.xlsx)",
    )
    parser.add_argument(
        "--seuil_on", type=float, default=0.05,
        help="Seuil de binarisation P_reel_clim → ON en kW (défaut : 0.05)",
    )
    args = parser.parse_args()

    Path(args.rapport).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CALCUL DES MÉTRIQUES — THERMO NILM")
    print("=" * 60)
    print(f"  Dossier CSV  : {args.dossier}")
    print(f"  Rapport Excel: {args.rapport}")
    print(f"  Seuil ON     : {args.seuil_on} kW")

    pattern   = str(Path(args.dossier) / "resultats_desagregation_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"\n⚠️  Aucun fichier 'resultats_desagregation_*.csv' trouvé dans : {args.dossier}")
        print("   Lancez d'abord optiUneSemaine.py pour générer les CSV.")
        return

    print(f"\n  {len(csv_files)} fichier(s) trouvé(s)\n")

    resultats = []
    for csv_path in csv_files:
        nom = Path(csv_path).name
        print(f"  {nom} ... ", end="", flush=True)
        try:
            metriques = calculer_metriques(csv_path, seuil_on=args.seuil_on)
            resultats.append(metriques)
            print(f"RMSE={metriques['RMSE (kW)']:.4f}  MAE={metriques['MAE (kW)']:.4f}  F1={metriques['F1']}")
        except Exception as e:
            print(f"ERREUR : {e}")

    if not resultats:
        print("\n⚠️  Aucun résultat valide — rapport non généré.")
        return

    # ── Affichage du tableau récapitulatif ────────────────────────────
    print("\n" + "=" * 60)
    print("RÉCAPITULATIF")
    print("=" * 60)
    df_res = pd.DataFrame(resultats)[[
        "Client (dataid)", "Date début",
        "RMSE (kW)", "MAE (kW)", "Energy Frac", "Norm Err",
        "Recall", "F1", "FPR",
    ]]
    print(df_res.to_string(index=False))

    # ── Moyennes ──────────────────────────────────────────────────────
    print("\n  Moyennes :")
    for col in ["RMSE (kW)", "MAE (kW)", "Energy Frac", "Norm Err", "Recall", "F1", "FPR"]:
        vals = pd.to_numeric(df_res[col], errors="coerce").dropna()
        if len(vals) > 0:
            print(f"    {col:<15} : {vals.mean():.4f}")

    # ── Sauvegarde Excel ──────────────────────────────────────────────
    sauvegarder_excel(resultats, args.rapport)
    print(f"\n✔ Rapport sauvegardé : {args.rapport}")


if __name__ == "__main__":
    main()