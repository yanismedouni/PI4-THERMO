"""
Code pour importer les données et les prétraiter.

Created on Fri Feb 12 2026
@author: catherinehenri
"""

from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_proba_curve(df_proba: pd.DataFrame, title: str, ylabel: str) -> None:
    """
    Trace une courbe de probabilité P(ON) en fonction de la température.

    Paramètres
    ----------
    df_proba : pd.DataFrame
        Doit contenir 'temp_center' et 'p_on_mean'.
    title : str
        Titre du graphique.
    ylabel : str
        Libellé de l'axe Y.

    Retours
    -------
    None
    """
    plt.figure(figsize=(9, 5))
    plt.plot(df_proba["temp_center"], df_proba["p_on_mean"], marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Température extérieure (°C)")
    plt.ylabel(ylabel)
    plt.yticks(np.arange(0, 1.0001, 0.05))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_hourly_curve(
    df_hourly: pd.DataFrame,
    title: str,
    ylabel: str = "P(ON)",
    hour_col: str = "hour",
    p_col: str = "p_on_mean",
    n_col: Optional[str] = "n_points_total",
    show_n_as_text: bool = False,
) -> None:
    """
    Trace une courbe de probabilité d'utilisation en fonction de l'heure (0–23).

    Paramètres
    ----------
    df_hourly : pd.DataFrame
        DataFrame contenant au minimum `hour_col` et `p_col`.
        Typiquement la sortie de `estimate_hourly_usage_multi_region(...).season_hourly`
        ou `.peak_week_hourly`.
    title : str
        Titre du graphique.
    ylabel : str
        Étiquette de l'axe Y.
    hour_col : str
        Nom de la colonne contenant l'heure (0–23).
    p_col : str
        Nom de la colonne contenant la probabilité (0–1).
    n_col : str | None
        Colonne optionnelle pour le nombre de points (sert au diagnostic).
    show_n_as_text : bool
        Si True et si `n_col` existe, affiche N dans le coin du graphe.

    Retours
    -------
    None
    """
    if hour_col not in df_hourly.columns or p_col not in df_hourly.columns:
        raise ValueError(f"df_hourly doit contenir '{hour_col}' et '{p_col}'.")

    work = df_hourly[[hour_col, p_col] + ([n_col] if (n_col and n_col in df_hourly.columns) else [])].copy()
    work[hour_col] = pd.to_numeric(work[hour_col], errors="coerce")
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")

    work = work.dropna(subset=[hour_col, p_col]).sort_values(hour_col)

    plt.figure(figsize=(9, 5))
    plt.plot(work[hour_col], work[p_col], marker="o", linewidth=2)

    plt.title(title)
    plt.xlabel("Heure de la journée")
    plt.ylabel(ylabel)

    plt.xticks(np.arange(0, 24, 1))
    plt.yticks(np.arange(0, 1.0001, 0.05))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if show_n_as_text and n_col and n_col in work.columns:
        n_total = int(pd.to_numeric(work[n_col], errors="coerce").sum())
        plt.gca().text(
            0.99, 0.02,
            f"N total = {n_total:,}".replace(",", " "),
            transform=plt.gca().transAxes,
            ha="right", va="bottom"
        )

    plt.show()


def plot_hourly_curves_by_month(
    df_monthly_hourly: pd.DataFrame,
    title: str,
    ylabel: str = "P(ON)",
    month_col: str = "month",
    hour_col: str = "hour",
    p_col: str = "p_on_mean",
    months: Optional[Iterable[int]] = None,
) -> None:
    """
    Trace P(ON|heure) avec une courbe par mois (dans la saison).

    Paramètres
    ----------
    df_monthly_hourly : pd.DataFrame
        Doit contenir `month_col`, `hour_col`, `p_col`.
        Typiquement `HourlyUsageResult.monthly_hourly`.
    title : str
        Titre du graphique.
    ylabel : str
        Étiquette Y.
    month_col : str
        Colonne du mois (1–12).
    hour_col : str
        Colonne heure (0–23).
    p_col : str
        Colonne probabilité.
    months : Iterable[int] | None
        Si fourni, filtre sur ces mois.

    Retours
    -------
    None
    """
    required = {month_col, hour_col, p_col}
    if not required.issubset(df_monthly_hourly.columns):
        raise ValueError(f"df_monthly_hourly doit contenir {required}")

    work = df_monthly_hourly.copy()
    work[month_col] = pd.to_numeric(work[month_col], errors="coerce")
    work[hour_col] = pd.to_numeric(work[hour_col], errors="coerce")
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work = work.dropna(subset=[month_col, hour_col, p_col])

    if months is not None:
        months_set = set(months)
        work = work[work[month_col].isin(months_set)].copy()

    plt.figure(figsize=(10, 5))
    for m, g in work.groupby(month_col):
        g = g.sort_values(hour_col)
        plt.plot(g[hour_col], g[p_col], marker="o", linewidth=2, label=f"Mois {int(m)}")

    plt.title(title)
    plt.xlabel("Heure de la journée")
    plt.ylabel(ylabel)

    plt.xticks(np.arange(0, 24, 1))
    plt.yticks(np.arange(0, 1.0001, 0.05))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()