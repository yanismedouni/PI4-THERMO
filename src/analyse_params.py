# =============================================================================
# Module      : analyse_params.py
# Description : Analyse des paramètres optimaux à partir des résultats OAT.
#               Calcule un score combiné pondéré et identifie les meilleures
#               configurations par client et globalement.
#
#               Score combiné (à minimiser) :
#                 score = 0.5 * RMSE + (1/3) * MAE + (1/6) * (1 - F1)
#
# Usage :
#   python src/analyse_params.py
#   python src/analyse_params.py --input output/resultats_sensibilite_juillet.csv
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Poids du score combiné
W_RMSE = 0.5
W_MAE  = 1/3
W_F1   = 1/6  # appliqué sur (1 - F1) pour minimiser


# =============================================================================
# SCORE COMBINÉ
# =============================================================================

def calculer_score(rmse, mae, f1):
    """Score combiné à minimiser. Plus bas = meilleur."""
    return W_RMSE * rmse + W_MAE * mae + W_F1 * (1 - f1)


# =============================================================================
# CHARGEMENT
# =============================================================================

def charger_csv(chemin):
    df = pd.read_csv(chemin)
    df = df[df['statut_solveur'] != 'echec'].copy()
    df['score'] = calculer_score(df['rmse'], df['mae'], df['f1'])
    print(f"  {len(df)} runs chargés | "
          f"{df['dataid'].nunique()} clients | "
          f"{df['param_varie'].nunique()} paramètres")
    return df


# =============================================================================
# BASELINE
# =============================================================================

def analyser_baseline(df):
    """Extrait les scores de la baseline par client."""
    df_b = df[df['param_varie'] == 'baseline'].copy()
    df_b = df_b[['dataid', 'date', 'rmse', 'mae', 'f1', 'score']].copy()
    df_b = df_b.sort_values('dataid').reset_index(drop=True)
    df_b['rmse']  = df_b['rmse'].round(4)
    df_b['mae']   = df_b['mae'].round(4)
    df_b['f1']    = df_b['f1'].round(4)
    df_b['score'] = df_b['score'].round(4)
    return df_b


# =============================================================================
# MEILLEUR PARAMÈTRE PAR CLIENT
# =============================================================================

def meilleur_par_client(df):
    """
    Pour chaque client et chaque paramètre varié,
    trouve la valeur qui minimise le score combiné.
    """
    params = [p for p in df['param_varie'].unique() if p != 'baseline']
    lignes = []

    for dataid in sorted(df['dataid'].unique()):
        df_c = df[df['dataid'] == dataid]

        # Score baseline de ce client
        score_baseline = df_c[df_c['param_varie'] == 'baseline']['score'].values
        score_baseline = score_baseline[0] if len(score_baseline) > 0 else float('nan')

        for param in params:
            df_cp = df_c[df_c['param_varie'] == param]
            if df_cp.empty:
                continue

            idx_opt    = df_cp['score'].idxmin()
            score_opt  = df_cp.loc[idx_opt, 'score']
            valeur_opt = df_cp.loc[idx_opt, 'valeur_testee']
            mae_opt    = df_cp.loc[idx_opt, 'mae']
            rmse_opt   = df_cp.loc[idx_opt, 'rmse']
            f1_opt     = df_cp.loc[idx_opt, 'f1']

            gain_abs = score_baseline - score_opt
            gain_pct = (gain_abs / score_baseline * 100) if score_baseline > 0 else float('nan')

            lignes.append({
                'dataid':          dataid,
                'param_varie':     param,
                'valeur_opt':      valeur_opt,
                'score_opt':       round(score_opt,      4),
                'score_baseline':  round(score_baseline, 4),
                'gain_abs':        round(gain_abs,       4),
                'gain_pct':        round(gain_pct,       2),
                'mae_opt':         round(mae_opt,        4),
                'rmse_opt':        round(rmse_opt,       4),
                'f1_opt':          round(f1_opt,         4),
            })

    return pd.DataFrame(lignes)


# =============================================================================
# MEILLEUR PARAMÈTRE GLOBAL
# =============================================================================

def meilleur_global(df, baseline_score_moyen):
    """
    Pour chaque paramètre, calcule le score moyen sur tous les clients
    pour chaque valeur testée. Retourne la valeur optimale globale.
    """
    params = [p for p in df['param_varie'].unique() if p != 'baseline']
    lignes = []

    for param in params:
        df_p = df[df['param_varie'] == param]

        # Moyenne du score par valeur testée sur tous les clients
        agg = df_p.groupby('valeur_testee').agg(
            score_moy  = ('score', 'mean'),
            score_std  = ('score', 'std'),
            mae_moy    = ('mae',   'mean'),
            rmse_moy   = ('rmse',  'mean'),
            f1_moy     = ('f1',    'mean'),
            n_clients  = ('dataid', 'nunique'),
        ).reset_index()

        idx_opt    = agg['score_moy'].idxmin()
        score_opt  = agg.loc[idx_opt, 'score_moy']
        valeur_opt = agg.loc[idx_opt, 'valeur_testee']
        mae_opt    = agg.loc[idx_opt, 'mae_moy']
        rmse_opt   = agg.loc[idx_opt, 'rmse_moy']
        f1_opt     = agg.loc[idx_opt, 'f1_moy']
        score_std  = agg.loc[idx_opt, 'score_std']

        gain_abs = baseline_score_moyen - score_opt
        gain_pct = (gain_abs / baseline_score_moyen * 100) if baseline_score_moyen > 0 else float('nan')

        lignes.append({
            'param_varie':          param,
            'valeur_opt_globale':   valeur_opt,
            'score_moy_opt':        round(score_opt,              4),
            'score_std_opt':        round(score_std,              4),
            'score_baseline_moy':   round(baseline_score_moyen,   4),
            'gain_abs':             round(gain_abs,               4),
            'gain_pct':             round(gain_pct,               2),
            'mae_moy_opt':          round(mae_opt,                4),
            'rmse_moy_opt':         round(rmse_opt,               4),
            'f1_moy_opt':           round(f1_opt,                 4),
            'n_clients':            int(agg.loc[idx_opt, 'n_clients']),
        })

    df_global = pd.DataFrame(lignes)
    df_global = df_global.sort_values('gain_abs', ascending=False).reset_index(drop=True)
    return df_global


# =============================================================================
# AFFICHAGE CONSOLE
# =============================================================================

def afficher_rapport(baseline, par_client, global_params):

    print("\n" + "=" * 70)
    print("RAPPORT D'ANALYSE — SCORE COMBINÉ")
    print(f"  Score = {W_RMSE}×RMSE + {W_MAE:.4f}×MAE + {W_F1:.4f}×(1-F1)")
    print("=" * 70)

    # ── Baseline ──────────────────────────────────────────────────────────────
    score_moy = baseline['score'].mean()
    print("\n── 1. BASELINE PAR CLIENT ────────────────────────────────────────")
    print(baseline.to_string(index=False))
    print(f"\n  Score baseline moyen : {score_moy:.4f}")

    # ── Meilleur par client ────────────────────────────────────────────────────
    print("\n── 2. MEILLEURE VALEUR PAR PARAMÈTRE PAR CLIENT ─────────────────")
    for dataid in sorted(par_client['dataid'].unique()):
        df_c = par_client[par_client['dataid'] == dataid].copy()
        score_b = df_c['score_baseline'].iloc[0]
        print(f"\n  Client {dataid}  (baseline score={score_b:.4f})")
        print(f"  {'Paramètre':<22} {'Valeur optimale':<30} "
              f"{'Score':>8} {'Gain abs':>10} {'Gain %':>8} "
              f"{'MAE':>8} {'RMSE':>8} {'F1':>8}")
        print("  " + "-" * 106)
        for _, row in df_c.iterrows():
            print(f"  {row['param_varie']:<22} {str(row['valeur_opt']):<30} "
                  f"{row['score_opt']:>8.4f} {row['gain_abs']:>10.4f} "
                  f"{row['gain_pct']:>7.1f}% "
                  f"{row['mae_opt']:>8.4f} {row['rmse_opt']:>8.4f} "
                  f"{row['f1_opt']:>8.4f}")

    # ── Global ─────────────────────────────────────────────────────────────────
    print("\n── 3. MEILLEURE VALEUR PAR PARAMÈTRE — GLOBAL ───────────────────")
    print(f"  (score baseline moyen de référence : {score_moy:.4f})\n")
    print(f"  {'Paramètre':<22} {'Valeur optimale':<30} "
          f"{'Score moy':>10} {'±std':>8} {'Gain abs':>10} {'Gain %':>8} "
          f"{'MAE':>8} {'RMSE':>8} {'F1':>8}")
    print("  " + "-" * 116)
    for _, row in global_params.iterrows():
        print(f"  {row['param_varie']:<22} {str(row['valeur_opt_globale']):<30} "
              f"{row['score_moy_opt']:>10.4f} {row['score_std_opt']:>8.4f} "
              f"{row['gain_abs']:>10.4f} {row['gain_pct']:>7.1f}% "
              f"{row['mae_moy_opt']:>8.4f} {row['rmse_moy_opt']:>8.4f} "
              f"{row['f1_moy_opt']:>8.4f}")

    print("\n── 4. PARAMÈTRES OPTIMAUX RECOMMANDÉS ───────────────────────────")
    for _, row in global_params.iterrows():
        print(f"  {row['param_varie']:<22} → {row['valeur_opt_globale']}  "
              f"(gain {row['gain_pct']:.1f}% vs baseline)")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse des paramètres optimaux — score combiné OAT"
    )
    parser.add_argument(
        '--input', type=str,
        default=str(OUTPUT_DIR / "resultats_sensibilite_juillet.csv"),
        help="CSV produit par script_param.py"
    )
    parser.add_argument(
        '--output', type=str,
        default=str(OUTPUT_DIR / "analyse_params.csv"),
        help="Chemin du CSV de sortie"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ANALYSE DES PARAMÈTRES OPTIMAUX")
    print("=" * 70)

    df         = charger_csv(args.input)
    baseline   = analyser_baseline(df)
    par_client = meilleur_par_client(df)
    global_p   = meilleur_global(df, baseline['score'].mean())

    afficher_rapport(baseline, par_client, global_p)

    # Export CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(  args.output.replace('.csv', '_baseline.csv'),   index=False)
    par_client.to_csv(args.output.replace('.csv', '_par_client.csv'), index=False)
    global_p.to_csv(  args.output.replace('.csv', '_global.csv'),     index=False)

    print(f"\n  Fichiers exportés :")
    print(f"    {args.output.replace('.csv', '_baseline.csv')}")
    print(f"    {args.output.replace('.csv', '_par_client.csv')}")
    print(f"    {args.output.replace('.csv', '_global.csv')}")


if __name__ == "__main__":
    main()