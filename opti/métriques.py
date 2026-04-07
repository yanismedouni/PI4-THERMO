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

    # ── Gap relatif MOSEK ─────────────────────────────────────────────
    gap_reel = float("nan")
    if "gap_reel" in df.columns:
        vals = pd.to_numeric(df["gap_reel"], errors="coerce").dropna()
        if not vals.empty:
            gap_reel = float(vals.iloc[0])

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
        "Gap réel (%)" : round(gap_reel * 100, 2) if not np.isnan(gap_reel) else float("nan"),
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
    n_cols = 17
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
        "Gap réel (%)",
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
        "Gap réel (%)",
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
    col_widths = [42, 10, 12, 8, 10, 10, 12, 10, 10, 11, 10, 10, 12, 7, 7, 7, 7]
    for i, w in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.freeze_panes = "A3"

    wb.save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION SUR UNE PÉRIODE COMPLÈTE (résultats_ID_annee_####_complet.csv)
# ─────────────────────────────────────────────────────────────────────────────

def evaluer_periode(
    csv_path: str,
    seuil_on: float = 0.05,
    par_saison: bool = True,
    par_mois: bool = True,
) -> dict:
    """
    Évalue les performances sur un fichier consolidé annuel/été
    (format : resultats_<dataid>_<periode>_<annee>_complet.csv).

    Retourne un dict avec :
      - metriques_globales  : mêmes métriques que calculer_metriques()
      - metriques_par_saison: dict {saison: métriques} si par_saison=True
      - metriques_par_mois  : dict {mois: métriques}  si par_mois=True
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    colonnes_requises = [COL_REEL, COL_ESTIME, COL_ON_PRED, "timestamp"]
    manquantes = [c for c in colonnes_requises if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {Path(csv_path).name} : {manquantes}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["mois"]   = df["timestamp"].dt.month
    df["saison"] = df["mois"].map({
        12: "Hiver",     1: "Hiver",     2: "Hiver",
         3: "Printemps", 4: "Printemps", 5: "Printemps",
         6: "Été",       7: "Été",       8: "Été",
         9: "Automne",  10: "Automne",  11: "Automne",
    })

    # Extraire dataid/période/année depuis le nom du fichier
    # Format : resultats_<dataid>_<periode>_<annee>_complet.csv
    nom     = Path(csv_path).stem   # ex. resultats_3864_annee_2015_complet
    parties = nom.split("_")
    try:
        dataid  = parties[1]
        periode = parties[2]
        annee   = parties[3]
    except IndexError:
        dataid, periode, annee = "—", "—", "—"

    def _metriques_df(sous_df: pd.DataFrame) -> dict:
        """Calcule les métriques sur un sous-ensemble du DataFrame."""
        if sous_df.empty:
            return {}
        y_reel   = pd.to_numeric(sous_df[COL_REEL],    errors="coerce").fillna(0).to_numpy()
        y_estime = pd.to_numeric(sous_df[COL_ESTIME],  errors="coerce").fillna(0).to_numpy()
        o_pred   = pd.to_numeric(sous_df[COL_ON_PRED], errors="coerce").fillna(0).to_numpy()
        o_reel   = (y_reel > seuil_on).astype(int)

        rmse        = float(np.sqrt(np.mean((y_reel - y_estime) ** 2)))
        mae         = float(np.mean(np.abs(y_reel - y_estime)))
        energy_frac = float(y_estime.sum() / max(y_reel.sum(), 1e-9))
        norm_err    = float(np.abs(y_reel - y_estime).sum() / max(np.abs(y_reel).sum(), 1e-9))

        o_pred_bin = np.round(o_pred).astype(int)
        tp = int(((o_reel == 1) & (o_pred_bin == 1)).sum())
        tn = int(((o_reel == 0) & (o_pred_bin == 0)).sum())
        fp = int(((o_reel == 0) & (o_pred_bin == 1)).sum())
        fn = int(((o_reel == 1) & (o_pred_bin == 0)).sum())

        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if not (np.isnan(precision) or np.isnan(recall))
              and (precision + recall) > 0 else float("nan"))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

        return {
            "N pas"       : len(sous_df),
            "RMSE (kW)"   : round(rmse,        4),
            "MAE (kW)"    : round(mae,          4),
            "Energy Frac" : round(energy_frac,  4),
            "Norm Err"    : round(norm_err,      4),
            "Recall"      : round(recall,        4) if not np.isnan(recall)    else float("nan"),
            "Precision"   : round(precision,     4) if not np.isnan(precision) else float("nan"),
            "F1"          : round(f1,            4) if not np.isnan(f1)        else float("nan"),
            "FPR"         : round(fpr,           4) if not np.isnan(fpr)       else float("nan"),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        }

    resultats = {
        "fichier"           : Path(csv_path).name,
        "dataid"            : dataid,
        "periode"           : periode,
        "annee"             : annee,
        "metriques_globales": _metriques_df(df),
    }

    if par_saison:
        resultats["metriques_par_saison"] = {
            s: _metriques_df(df[df["saison"] == s])
            for s in ["Hiver", "Printemps", "Été", "Automne"]
            if (df["saison"] == s).any()
        }

    if par_mois:
        noms_mois = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Jun",
                     7:"Jul", 8:"Aoû", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"}
        resultats["metriques_par_mois"] = {
            noms_mois[m]: _metriques_df(df[df["mois"] == m])
            for m in sorted(df["mois"].unique())
        }

    return resultats


def afficher_evaluation_periode(resultats: dict) -> None:
    """Affiche les métriques d'une évaluation de période de façon lisible."""

    print(f"\n{'=' * 65}")
    print(f"ÉVALUATION — Client {resultats['dataid']}  |  "
          f"{resultats['periode'].upper()} {resultats['annee']}")
    print(f"{'=' * 65}")

    g = resultats["metriques_globales"]
    print(f"\n  Métriques globales ({g.get('N pas', '?')} pas) :")
    for k, v in g.items():
        if k != "N pas":
            print(f"    {k:<15} : {v}")

    if "metriques_par_saison" in resultats:
        print(f"\n  Par saison :")
        df_s = pd.DataFrame(resultats["metriques_par_saison"]).T[
            ["N pas", "RMSE (kW)", "MAE (kW)", "Energy Frac", "F1"]
        ]
        print(df_s.to_string())

    if "metriques_par_mois" in resultats:
        print(f"\n  Par mois :")
        df_m = pd.DataFrame(resultats["metriques_par_mois"]).T[
            ["N pas", "RMSE (kW)", "MAE (kW)", "Energy Frac", "F1"]
        ]
        print(df_m.to_string())


def sauvegarder_excel_periode(resultats: dict, out_path: str) -> None:
    """
    Génère un rapport Excel multi-onglets pour une évaluation de période.
    Onglets : Globale | Par saison | Par mois
    """
    wb = Workbook()

    BLUE_DARK  = "1F3864"
    BLUE_MID   = "2E75B6"
    BLUE_LIGHT = "D6E4F0"
    WHITE      = "FFFFFF"
    GREY       = "F2F2F2"

    def fill(c):   return PatternFill("solid", start_color=c)
    def font_(bold=False, color="000000", size=10):
        return Font(name="Arial", bold=bold, color=color, size=size)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin   = Side(style="thin", color="BBBBBB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    COLS_AFFICHAGE = [
        "N pas", "RMSE (kW)", "MAE (kW)", "Energy Frac", "Norm Err",
        "Recall", "Precision", "F1", "FPR", "TP", "TN", "FP", "FN",
    ]

    def _ecrire_onglet(ws, titre: str, donnees: dict) -> None:
        """Écrit un onglet avec une ligne de titre + en-têtes + données."""
        n_cols = 1 + len(COLS_AFFICHAGE)

        ws.merge_cells(f"A1:{get_column_letter(n_cols)}1")
        ws["A1"] = titre
        ws["A1"].font      = font_(bold=True, color=WHITE, size=12)
        ws["A1"].fill      = fill(BLUE_DARK)
        ws["A1"].alignment = center
        ws.row_dimensions[1].height = 24

        en_tetes = ["Groupe"] + COLS_AFFICHAGE
        for col_idx, h in enumerate(en_tetes, start=1):
            c           = ws.cell(row=2, column=col_idx, value=h)
            c.font      = font_(bold=True, color=WHITE, size=10)
            c.fill      = fill(BLUE_MID)
            c.alignment = center
            c.border    = border
        ws.row_dimensions[2].height = 28

        for i, (label, metriques) in enumerate(donnees.items(), start=3):
            shade = GREY if i % 2 == 0 else WHITE
            ws.row_dimensions[i].height = 18
            ws.cell(row=i, column=1, value=label).font      = font_(bold=True, size=10)
            ws.cell(row=i, column=1).fill      = fill(shade)
            ws.cell(row=i, column=1).alignment = center
            ws.cell(row=i, column=1).border    = border
            for col_idx, col in enumerate(COLS_AFFICHAGE, start=2):
                val  = metriques.get(col, "—")
                c    = ws.cell(row=i, column=col_idx, value=val)
                c.fill      = fill(shade)
                c.border    = border
                c.font      = font_(size=10)
                c.alignment = center

        for col_idx, w in enumerate([18] + [12] * len(COLS_AFFICHAGE), start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = w
        ws.freeze_panes = "A3"

    # Onglet 1 — Globale
    ws_global = wb.active
    ws_global.title = "Globale"
    _ecrire_onglet(
        ws_global,
        f"Métriques globales — Client {resultats['dataid']} "
        f"({resultats['periode'].upper()} {resultats['annee']})",
        {"Global": resultats["metriques_globales"]},
    )

    # Onglet 2 — Par saison
    if "metriques_par_saison" in resultats:
        ws_s = wb.create_sheet("Par saison")
        _ecrire_onglet(
            ws_s,
            f"Métriques par saison — Client {resultats['dataid']}",
            resultats["metriques_par_saison"],
        )

    # Onglet 3 — Par mois
    if "metriques_par_mois" in resultats:
        ws_m = wb.create_sheet("Par mois")
        _ecrire_onglet(
            ws_m,
            f"Métriques par mois — Client {resultats['dataid']}",
            resultats["metriques_par_mois"],
        )

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
    parser.add_argument(
        "--periode", type=str, default=None,
        help=(
            "Si fourni, évalue les fichiers '*_complet.csv' au lieu des "
            "fichiers hebdomadaires. Ex : --periode annee ou --periode ete"
        ),
    )
    args = parser.parse_args()

    Path(args.rapport).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CALCUL DES MÉTRIQUES — THERMO NILM")
    print("=" * 60)
    print(f"  Dossier CSV  : {args.dossier}")
    print(f"  Rapport Excel: {args.rapport}")
    print(f"  Seuil ON     : {args.seuil_on} kW")

    # ── Mode période : fichiers *_complet.csv ────────────────────────────
    if args.periode is not None:
        pattern_complet = str(
            Path(args.dossier) / f"resultats_*_{args.periode}_*_complet.csv"
        )
        csv_complets = sorted(glob.glob(pattern_complet))

        if not csv_complets:
            print(f"\n⚠️  Aucun fichier '*_{args.periode}_*_complet.csv' trouvé dans : {args.dossier}")
            print("   Lancez d'abord optiPeriode.py pour générer les CSV consolidés.")
            return

        print(f"\n  {len(csv_complets)} fichier(s) complet(s) trouvé(s)\n")

        for csv_path in csv_complets:
            print(f"  {Path(csv_path).name}")
            try:
                res = evaluer_periode(csv_path, seuil_on=args.seuil_on)
                afficher_evaluation_periode(res)

                rapport_xlsx = str(
                    Path(args.rapport).parent
                    / f"metriques_{res['dataid']}_{res['periode']}_{res['annee']}.xlsx"
                )
                sauvegarder_excel_periode(res, rapport_xlsx)
                print(f"  ✔ Rapport sauvegardé : {Path(rapport_xlsx).name}")
            except Exception as e:
                print(f"  ✗ Erreur : {e}")
        return

    # ── Mode hebdomadaire : fichiers resultats_desagregation_*.csv ───────
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
        "Recall", "F1", "FPR", "Gap réel (%)",   # ← ajouter
    ]]
    print(df_res.to_string(index=False))

    # ── Moyennes ──────────────────────────────────────────────────────
    print("\n  Moyennes :")
    for col in ["RMSE (kW)", "MAE (kW)", "Energy Frac", "Norm Err", "Recall", "F1", "FPR", "Gap réel (%)"]:
        vals = pd.to_numeric(df_res[col], errors="coerce").dropna()
        if len(vals) > 0:
            print(f"    {col:<15} : {vals.mean():.4f}")

    # ── Sauvegarde Excel ──────────────────────────────────────────────
    sauvegarder_excel(resultats, args.rapport)
    print(f"\n✔ Rapport sauvegardé : {args.rapport}")


if __name__ == "__main__":
    main()