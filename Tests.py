#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procédure de test.

Created on Fri Feb 26 2026
@author: catherinehenri
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import time


def load_results_csv(
    csv_path: str,
    encoding: str = "utf-8",
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """
    Charge un CSV de résultats.

    Paramètres
    ----------
    csv_path : str
        Chemin du CSV contenant les résultats de l'algorithme.
    encoding : str
        Encodage (utf-8 par défaut).
    sep : str | None
        Séparateur. None = autodétection via engine="python".

    Retours
    -------
    pd.DataFrame
        Données du CSV.
    """
    if sep is None:
        return pd.read_csv(csv_path, encoding=encoding, sep=None, engine="python")
    return pd.read_csv(csv_path, encoding=encoding, sep=sep)


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """
    Valide la présence de colonnes.

    Paramètres
    ----------
    df : pd.DataFrame
        Données.
    cols : Sequence[str]
        Colonnes requises.

    Retours
    -------
    None
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")


def _to_binary(x: pd.Series) -> np.ndarray:
    """
    Convertit une série en binaire 0/1.

    Paramètres
    ----------
    x : pd.Series
        Série bool/int/float.

    Retours
    -------
    np.ndarray
        Tableau binaire (0/1).
    """
    v = pd.to_numeric(x, errors="coerce").fillna(0)
    return (v != 0).astype(int).to_numpy()


def confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Calcule une matrice de confusion binaire.

    Paramètres
    ----------
    y_true : np.ndarray
        Vérité terrain (0/1).
    y_pred : np.ndarray
        Prédiction (0/1).

    Retours
    -------
    dict[str, int]
        tp, tn, fp, fn.
    """
    yt = (np.asarray(y_true).astype(int) != 0).astype(int)
    yp = (np.asarray(y_pred).astype(int) != 0).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcule recall, precision, f1, fpr.

    Paramètres
    ----------
    y_true : np.ndarray
        Vérité terrain binaire.
    y_pred : np.ndarray
        Prédiction binaire.

    Retours
    -------
    dict[str, float]
        recall, precision, f1, fpr.
    """
    c = confusion_binary(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan

    return {
        "recall": float(recall) if not np.isnan(recall) else np.nan,
        "precision": float(precision) if not np.isnan(precision) else np.nan,
        "f1": float(f1) if not np.isnan(f1) else np.nan,
        "fpr": float(fpr) if not np.isnan(fpr) else np.nan,
    }


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calcule la MAPE.

    Paramètres
    ----------
    y_true : np.ndarray
        Série vraie.
    y_pred : np.ndarray
        Série prédite.
    eps : float
        Protection division par zéro.

    Retours
    -------
    float
        MAPE.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yp - yt) / denom))


def energy_fraction(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """
    Calcule la fraction d'énergie reconstruite.

    Paramètres
    ----------
    y_true : np.ndarray
        Série vraie.
    y_pred : np.ndarray
        Série prédite.
    eps : float
        Protection division par zéro.

    Retours
    -------
    float
        sum(y_pred)/sum(y_true).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(yp.sum() / max(yt.sum(), eps))


def norm_l1_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """
    Calcule ||y_pred - y_true||_1 / ||y_true||_1.

    Paramètres
    ----------
    y_true : np.ndarray
        Série vraie.
    y_pred : np.ndarray
        Série prédite.
    eps : float
        Protection division par zéro.

    Retours
    -------
    float
        Erreur normalisée L1.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.abs(yp - yt).sum() / max(np.abs(yt).sum(), eps))


def _result(test_id: str, passed: bool, metrics: Dict[str, float], details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construit un retour standard.

    Paramètres
    ----------
    test_id : str
        Identifiant du test.
    passed : bool
        Résultat global.
    metrics : dict[str, float]
        Métriques du test.
    details : dict[str, Any]
        Détails (par région, confusion, etc.).

    Retours
    -------
    dict[str, Any]
        Résultat standard.
    """
    return {"test_id": test_id, "passed": passed, "metrics": metrics, "details": details}


def test_T02_detection(
    csv_path: str,
    *,
    true_col: str,
    pred_col: str,
    filter_expr: Optional[str] = None,
    recall_min: float = 0.70,
    f1_min: float = 0.70,
) -> Dict[str, Any]:
    """
    T-02 — Détection (recall et F1).

    Paramètres
    ----------
    csv_path : str
        Chemin du CSV résultats.
    true_col : str
        Colonne ON vraie (0/1).
    pred_col : str
        Colonne ON prédite (0/1).
    filter_expr : str | None
        Filtre pandas query (ex: "region == 'Austin' and equipment == 'ac'").
    recall_min : float
        Seuil minimal de recall.
    f1_min : float
        Seuil minimal de F1.

    Retours
    -------
    dict[str, Any]
        test_id, passed, metrics, details.
    """
    df = load_results_csv(csv_path)
    _require_cols(df, [true_col, pred_col])

    if filter_expr:
        df = df.query(filter_expr).copy()

    y_true = _to_binary(df[true_col])
    y_pred = _to_binary(df[pred_col])

    met = detection_metrics(y_true, y_pred)
    conf = confusion_binary(y_true, y_pred)

    passed = (not np.isnan(met["recall"])) and (not np.isnan(met["f1"])) and (met["recall"] >= recall_min) and (met["f1"] >= f1_min)

    return _result(
        "T-02",
        passed,
        met,
        {"confusion": conf, "n": int(len(df)), "filter": filter_expr, "targets": {"recall_min": recall_min, "f1_min": f1_min}},
    )


def test_T03_false_positives(
    csv_path: str,
    *,
    true_col: str,
    pred_col: str,
    filter_expr: Optional[str] = None,
    fpr_max: float = 0.15,
) -> Dict[str, Any]:
    """
    T-03 — Faux positifs (FPR).

    Paramètres
    ----------
    csv_path : str
        Chemin du CSV résultats.
    true_col : str
        Colonne ON vraie.
    pred_col : str
        Colonne ON prédite.
    filter_expr : str | None
        Filtre pandas query.
    fpr_max : float
        Seuil maximal FPR.

    Retours
    -------
    dict[str, Any]
        test_id, passed, metrics, details.
    """
    df = load_results_csv(csv_path)
    _require_cols(df, [true_col, pred_col])

    if filter_expr:
        df = df.query(filter_expr).copy()

    y_true = _to_binary(df[true_col])
    y_pred = _to_binary(df[pred_col])

    met = detection_metrics(y_true, y_pred)
    conf = confusion_binary(y_true, y_pred)

    passed = (not np.isnan(met["fpr"])) and (met["fpr"] <= fpr_max)

    return _result(
        "T-03",
        passed,
        {"fpr": met["fpr"]},
        {"confusion": conf, "n": int(len(df)), "filter": filter_expr, "target": {"fpr_max": fpr_max}},
    )


def test_T05_prediction(
    csv_path: str,
    *,
    true_power_col: str,
    pred_power_col: str,
    filter_expr: Optional[str] = None,
    mape_max: float = 0.25,
    energy_frac_min: float = 0.85,
    norm_err_max: float = 0.15,
) -> Dict[str, Any]:
    """
    T-05 — Prédiction (MAPE, fraction énergie, erreur normalisée).

    Paramètres
    ----------
    csv_path : str
        Chemin du CSV résultats.
    true_power_col : str
        Colonne puissance/énergie vraie.
    pred_power_col : str
        Colonne puissance/énergie prédite.
    filter_expr : str | None
        Filtre pandas query.
    mape_max : float
        Seuil maximal MAPE.
    energy_frac_min : float
        Seuil minimal fraction énergie.
    norm_err_max : float
        Seuil maximal erreur normalisée.

    Retours
    -------
    dict[str, Any]
        test_id, passed, metrics, details.
    """
    df = load_results_csv(csv_path)
    _require_cols(df, [true_power_col, pred_power_col])

    if filter_expr:
        df = df.query(filter_expr).copy()

    yt = pd.to_numeric(df[true_power_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    yp = pd.to_numeric(df[pred_power_col], errors="coerce").fillna(0).to_numpy(dtype=float)

    met = {
        "mape": mape(yt, yp),
        "energy_frac": energy_fraction(yt, yp),
        "norm_err": norm_l1_error(yt, yp),
    }

    passed = (met["mape"] <= mape_max) and (met["energy_frac"] >= energy_frac_min) and (met["norm_err"] <= norm_err_max)

    return _result(
        "T-05",
        passed,
        met,
        {"n": int(len(df)), "filter": filter_expr, "targets": {"mape_max": mape_max, "energy_frac_min": energy_frac_min, "norm_err_max": norm_err_max}},
    )


def test_T04_fpr_three_regions(
    csv_path: str,
    *,
    region_col: str,
    true_col: str,
    pred_col: str,
    regions: Sequence[str],
    base_filter_expr: Optional[str] = None,
    fpr_max: float = 0.15,
) -> Dict[str, Any]:
    """
    T-04 — Faux positifs sur trois régions.

    Paramètres
    ----------
    csv_path : str
        Chemin du CSV résultats.
    region_col : str
        Colonne région.
    true_col : str
        Colonne ON vraie.
    pred_col : str
        Colonne ON prédite.
    regions : Sequence[str]
        Liste des régions à évaluer.
    base_filter_expr : str | None
        Filtre commun (ex: "equipment == 'ac'").
    fpr_max : float
        Seuil maximal FPR par région.

    Retours
    -------
    dict[str, Any]
        Résultat global + métriques par région.
    """
    df = load_results_csv(csv_path)
    _require_cols(df, [region_col, true_col, pred_col])

    if base_filter_expr:
        df = df.query(base_filter_expr).copy()

    per_region: Dict[str, Any] = {}
    passed_all = True

    for r in regions:
        sub = df[df[region_col] == r].copy()
        y_true = _to_binary(sub[true_col]) if len(sub) else np.array([], dtype=int)
        y_pred = _to_binary(sub[pred_col]) if len(sub) else np.array([], dtype=int)

        met = detection_metrics(y_true, y_pred) if len(sub) else {"fpr": np.nan, "recall": np.nan, "precision": np.nan, "f1": np.nan}
        ok = (not np.isnan(met["fpr"])) and (met["fpr"] <= fpr_max)

        per_region[r] = {
            "n": int(len(sub)),
            "fpr": float(met["fpr"]) if not np.isnan(met["fpr"]) else np.nan,
            "passed_region": bool(ok),
        }
        passed_all = passed_all and ok

    fprs = [v["fpr"] for v in per_region.values()]
    metrics = {"fpr_max_observed": float(np.nanmax(fprs)) if fprs else np.nan}

    return _result(
        "T-04",
        passed_all,
        metrics,
        {"per_region": per_region, "regions": list(regions), "base_filter": base_filter_expr, "target": {"fpr_max": fpr_max}},
    )


def save_test_report(
    results: Sequence[Dict[str, Any]],
    out_csv_path: str,
) -> None:
    """
    Sauvegarde un rapport de tests sous forme tabulaire.

    Paramètres
    ----------
    results : Sequence[dict]
        Liste des résultats (retours des fonctions test_Txx_*).
    out_csv_path : str
        Chemin du CSV de sortie.

    Retours
    -------
    None
    """
    rows = []
    for r in results:
        row = {"test_id": r.get("test_id"), "passed": r.get("passed")}
        metrics = r.get("metrics", {})
        for k, v in metrics.items():
            row[f"metric_{k}"] = v
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv_path, index=False)