"""
Code pour tracer les graphiques des données analysées.

Created on Fri Feb 12 2026
@author: catherinehenri
"""

from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_proba_curve(
    df_proba: pd.DataFrame,
    equipment_name: str,
    region_name: Optional[str] = None,
    season_label: Optional[str] = None,
    ylabel: str = "Probabilité d'utilisation",
    show_context: bool = False,
) -> None:
    """
    Trace P(ON | T_ext) en fonction de la température extérieure.

    Paramètres
    ----------
    df_proba : pd.DataFrame
        Doit contenir 'temp_center' et 'p_on_mean'.
    equipment_name : str
        Nom de l'équipement (ex: "Chauffage").
    region_name : str | None
        Région (optionnel; recommandé de le mettre plutôt dans la caption du rapport).
    season_label : str | None
        Période (optionnel; recommandé de le mettre plutôt dans la caption du rapport).
    ylabel : str
        Libellé de l'axe Y.
    show_context : bool
        Si True, ajoute une 2e ligne de contexte (région/période) au titre.

    Retours
    -------
    None
    """
    required = {"temp_center", "p_on_mean"}
    if not required.issubset(df_proba.columns):
        raise ValueError(f"df_proba doit contenir {required}")

    work = df_proba.copy()
    work["temp_center"] = pd.to_numeric(work["temp_center"], errors="coerce")
    work["p_on_mean"] = pd.to_numeric(work["p_on_mean"], errors="coerce")
    work = work.dropna(subset=["temp_center", "p_on_mean"]).sort_values("temp_center")

    title = f"P(ON | T_ext) – {equipment_name}"
    if show_context and (region_name or season_label):
        ctx = " – ".join([x for x in [region_name, season_label] if x])
        title = f"{title}\n{ctx}"

    plt.figure(figsize=(9, 5))
    plt.plot(work["temp_center"], work["p_on_mean"], marker="o", linewidth=2)

    plt.title(title, fontsize=12)
    plt.xlabel("Température extérieure (°C)")
    plt.ylabel(ylabel)

    plt.yticks(np.arange(0, 1.0001, 0.05))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_hourly_curve(
    df_hourly: pd.DataFrame,
    equipment_name: str,
    region_name: Optional[str] = None,
    period_label: Optional[str] = None,
    ylabel: str = "Probabilité d'utilisation",
    hour_col: str = "hour",
    p_col: str = "p_on_mean",
    n_col: Optional[str] = "n_points_total",
    show_n_as_text: bool = False,
    show_context: bool = False,
) -> None:
    """
    Trace P(ON | h) en fonction de l'heure (0–23).

    Paramètres
    ----------
    df_hourly : pd.DataFrame
        Doit contenir au minimum `hour_col` et `p_col`.
    equipment_name : str
        Nom de l'équipement.
    region_name : str | None
        Région (optionnel; recommandé plutôt dans la caption).
    period_label : str | None
        Libellé période (optionnel; ex: "Saison", "Semaine pic").
    ylabel : str
        Libellé de l'axe Y.
    hour_col : str
        Nom de la colonne contenant l'heure (0–23).
    p_col : str
        Nom de la colonne contenant P(ON).
    n_col : str | None
        Colonne optionnelle avec le nombre de points.
    show_n_as_text : bool
        Si True, affiche N total dans le coin du graphe (diagnostic).
    show_context : bool
        Si True, ajoute une 2e ligne de contexte (région/période) au titre.

    Retours
    -------
    None
    """
    if hour_col not in df_hourly.columns or p_col not in df_hourly.columns:
        raise ValueError(f"df_hourly doit contenir '{hour_col}' et '{p_col}'.")

    work = df_hourly.copy()
    work[hour_col] = pd.to_numeric(work[hour_col], errors="coerce")
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work = work.dropna(subset=[hour_col, p_col]).sort_values(hour_col)

    title = f"P(ON | h) – {equipment_name}"
    if show_context and (region_name or period_label):
        ctx = " – ".join([x for x in [region_name, period_label] if x])
        title = f"{title}\n{ctx}"

    plt.figure(figsize=(9, 5))
    plt.plot(work[hour_col], work[p_col], marker="o", linewidth=2)

    plt.title(title, fontsize=12)
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
            f"N = {n_total:,}".replace(",", " "),
            transform=plt.gca().transAxes,
            ha="right", va="bottom",
        )

    plt.show()


def plot_hourly_curves_by_month(
    df_monthly_hourly: pd.DataFrame,
    equipment_name: str,
    region_name: Optional[str] = None,
    season_label: Optional[str] = None,
    ylabel: str = "Probabilité d'utilisation",
    month_col: str = "month",
    hour_col: str = "hour",
    p_col: str = "p_on_mean",
    months: Optional[Iterable[int]] = None,
    show_context: bool = False,
) -> None:
    """
    Trace P(ON | h, mois) avec une courbe par mois.

    Paramètres
    ----------
    df_monthly_hourly : pd.DataFrame
        Doit contenir `month_col`, `hour_col`, `p_col`.
    equipment_name : str
        Nom de l'équipement.
    region_name : str | None
        Région (optionnel; recommandé plutôt dans la caption).
    season_label : str | None
        Période (optionnel; recommandé plutôt dans la caption).
    ylabel : str
        Libellé Y.
    month_col : str
        Colonne du mois (1–12).
    hour_col : str
        Colonne heure (0–23).
    p_col : str
        Colonne P(ON).
    months : Iterable[int] | None
        Si fourni, filtre sur ces mois.
    show_context : bool
        Si True, ajoute une 2e ligne de contexte (région/période) au titre.

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

    title = f"P(ON | h, mois) – {equipment_name}"
    if show_context and (region_name or season_label):
        ctx = " – ".join([x for x in [region_name, season_label] if x])
        title = f"{title}\n{ctx}"

    plt.figure(figsize=(10, 5))
    for m, g in work.groupby(month_col, observed=True):
        g = g.sort_values(hour_col)
        plt.plot(g[hour_col], g[p_col], marker="o", linewidth=2, label=f"Mois {int(m)}")

    plt.title(title, fontsize=12)
    plt.xlabel("Heure de la journée")
    plt.ylabel(ylabel)

    plt.xticks(np.arange(0, 24, 1))
    plt.yticks(np.arange(0, 1.0001, 0.05))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()