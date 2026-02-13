"""
Code pour importer les données et les prétraiter.

Created on Fri Feb 12 2026
@author: catherinehenri
"""

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