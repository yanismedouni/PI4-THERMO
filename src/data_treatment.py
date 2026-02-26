"""
Script pour creer un fichier CSV avec:
- Consommation totale (Grid - Solar)
- Consommation des furnaces (1, 2, 3 et total)
- Consommation des heaters (1, 2, 3 et total)
- Consommation totale chauffage (furnaces + heaters)
- Consommation air conditionne (air1, air2, air3 et total)
- Consommation TCLs totale (chauffage + climatisation)
- Temperature exterieure (avec interpolation lineaire si necessaire)
- Timestamps a chaque 15 minutes
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

def load_consumption_data(filepath):
    """
    Charge les donnees de consommation depuis un fichier Excel ou CSV
    """
    print("=" * 70)
    print("CHARGEMENT DES DONNEES DE CONSOMMATION")
    print("=" * 70)
    
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    
    print(f"* {len(df)} lignes chargees")
    print(f"* Colonnes disponibles: {df.columns.tolist()}")
    
    # Convertir la colonne de temps en datetime
    if 'local_15min' in df.columns:
        df['local_15min'] = pd.to_datetime(df['local_15min'])
        time_col = 'local_15min'
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        time_col = 'time'
    else:
        print("! Aucune colonne de temps trouvee!")
        return None, None
    
    print(f"* Periode: {df[time_col].min()} a {df[time_col].max()}")
    
    return df, time_col

def load_meteo_data(filepath):
    """
    Charge les donnees meteo depuis un fichier CSV
    Le fichier Open-Meteo a 3 lignes de metadonnees a sauter
    """
    print("\n" + "=" * 70)
    print("CHARGEMENT DES DONNEES METEO")
    print("=" * 70)
    
    # Essayer de detecter si le fichier a des metadonnees
    try:
        # Lire les premieres lignes pour detecter la structure
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if 'latitude' in first_line.lower() or 'elevation' in first_line.lower():
                # C'est un fichier Open-Meteo avec metadonnees
                df = pd.read_csv(filepath, skiprows=3)
            else:
                df = pd.read_csv(filepath)
    except:
        df = pd.read_csv(filepath)
    
    print(f"* {len(df)} lignes chargees")
    print(f"* Colonnes: {df.columns.tolist()}")
    
    # Renommer les colonnes si necessaire
    if len(df.columns) == 2:
        df.columns = ['time', 'temperature']
    elif 'temperature_2m' in df.columns:
        df.rename(columns={'temperature_2m': 'temperature'}, inplace=True)
    
    # Convertir le temps en datetime
    df['time'] = pd.to_datetime(df['time'])
    
    print(f"* Periode: {df['time'].min()} a {df['time'].max()}")
    
    return df

def interpolate_temperature(df_consumption, df_meteo, time_col):
    """
    Interpole les temperatures pour correspondre aux timestamps de consommation
    """
    print("\n" + "=" * 70)
    print("INTERPOLATION DES TEMPERATURES")
    print("=" * 70)
    
    # S'assurer que les deux dataframes sont tries par temps
    df_consumption = df_consumption.sort_values(time_col)
    df_meteo = df_meteo.sort_values('time')
    
    # Creer un index temporel pour l'interpolation
    df_consumption_indexed = df_consumption.set_index(time_col)
    df_meteo_indexed = df_meteo.set_index('time')
    
    # Obtenir la plage temporelle complete
    all_times = df_consumption_indexed.index.union(df_meteo_indexed.index).sort_values()
    
    # Reindexer les donnees meteo sur tous les temps
    df_meteo_reindexed = df_meteo_indexed.reindex(all_times)
    
    # Interpolation lineaire
    df_meteo_reindexed['temperature'] = df_meteo_reindexed['temperature'].interpolate(method='linear')
    
    # Extraire seulement les temperatures aux timestamps de consommation
    temperatures = df_meteo_reindexed.loc[df_consumption_indexed.index, 'temperature']
    
    print(f"* {len(temperatures)} valeurs de temperature interpolees")
    print(f"* Temperature min: {temperatures.min():.2f}C")
    print(f"* Temperature max: {temperatures.max():.2f}C")
    print(f"* Temperature moyenne: {temperatures.mean():.2f}C")
    
    return temperatures.values

def process_data(consumption_file, meteo_file, output_file):
    """
    Traite les donnees et cree le fichier CSV final
    """
    # Charger les données
    df_cons, time_col = load_consumption_data(consumption_file)
    if df_cons is None:
        return
    
    df_meteo = load_meteo_data(meteo_file)
    
    # Creer le dataframe de sortie
    print("\n" + "=" * 70)
    print("CREATION DU FICHIER DE SORTIE")
    print("=" * 70)
    
    df_output = pd.DataFrame()
    
    # Colonne de temps
    df_output['temps'] = df_cons[time_col]
    
    # Consommation totale (Grid - Solar)
    if 'grid' in df_cons.columns and 'solar' in df_cons.columns:
        df_output['consommation_totale'] = df_cons['grid'] - df_cons['solar']
        print("* Consommation totale = Grid - Solar")
    elif 'grid' in df_cons.columns:
        df_output['consommation_totale'] = df_cons['grid']
        print("* Consommation totale = Grid (pas de Solar)")
    else:
        print("! Colonne 'grid' non trouvee!")
        return
    
    # Furnaces
    furnace_cols = []
    for i in [1, 2, 3]:
        col_name = f'furnace{i}'
        if col_name in df_cons.columns:
            df_output[col_name] = df_cons[col_name]
            furnace_cols.append(col_name)
        else:
            df_output[col_name] = 0.0
            print(f"! {col_name} non trouvee, remplie avec 0")
    
    df_output['furnace_total'] = df_output[furnace_cols].sum(axis=1) if furnace_cols else 0
    print(f"* Furnaces ajoutees ({len(furnace_cols)} colonnes trouvees)")
    
    # Heaters
    heater_cols = []
    for i in [1, 2, 3]:
        col_name = f'heater{i}'
        if col_name in df_cons.columns:
            df_output[col_name] = df_cons[col_name]
            heater_cols.append(col_name)
        else:
            df_output[col_name] = 0.0
            print(f"! {col_name} non trouvee, remplie avec 0")
    
    df_output['heater_total'] = df_output[heater_cols].sum(axis=1) if heater_cols else 0
    print(f"* Heaters ajoutees ({len(heater_cols)} colonnes trouvees)")
    
    # Consommation totale de chauffage
    df_output['chauffage_total'] = df_output['furnace_total'] + df_output['heater_total']
    print("* Consommation totale chauffage calculee")
    
    # Air conditionne
    air_cols = []
    for i in [1, 2, 3]:
        col_name = f'air{i}'
        if col_name in df_cons.columns:
            df_output[col_name] = df_cons[col_name]
            air_cols.append(col_name)
        else:
            df_output[col_name] = 0.0
            print(f"! {col_name} non trouvee, remplie avec 0")
    
    df_output['air_total'] = df_output[air_cols].sum(axis=1) if air_cols else 0
    print(f"* Air conditionne ajoute ({len(air_cols)} colonnes trouvees)")
    
    # Consommation TCLs totale (chauffage + climatisation)
    df_output['TCLs_total'] = df_output['chauffage_total'] + df_output['air_total']
    print("* Consommation TCLs totale calculee")
    
    # Interpoler et ajouter les températures
    temperatures = interpolate_temperature(df_cons, df_meteo, time_col)
    df_output['temperature_exterieure'] = temperatures
    
    # Formater la colonne temps
    df_output['temps'] = df_output['temps'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Sauvegarder
    df_output.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("RESUME DU FICHIER CREE")
    print("=" * 70)
    print(f"* Fichier sauvegarde: {output_file}")
    print(f"* Nombre de lignes: {len(df_output)}")
    print(f"* Nombre de colonnes: {len(df_output.columns)}")
    print(f"\nColonnes creees:")
    for col in df_output.columns:
        print(f"  - {col}")
    
    print(f"\nApercu des donnees:")
    print(df_output.head(10))
    
    print(f"\nStatistiques de consommation:")
    print(f"  - Consommation totale moyenne: {df_output['consommation_totale'].mean():.2f} kW")
    print(f"  - Chauffage total moyen: {df_output['chauffage_total'].mean():.2f} kW")
    print(f"  - Air conditionne total moyen: {df_output['air_total'].mean():.2f} kW")
    print(f"  - TCLs total moyen: {df_output['TCLs_total'].mean():.2f} kW")
    
    print("\n[OK] Traitement termine avec succes!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_consumption_data.py <fichier_consommation> <fichier_meteo> <fichier_sortie>")
        print("\nExemple:")
        print("  python process_consumption_data.py Short_data.xlsx open-meteo.csv donnees_finales.csv")
        sys.exit(1)
    
    consumption_file = sys.argv[1]
    meteo_file = sys.argv[2]
    output_file = sys.argv[3]
    
    process_data(consumption_file, meteo_file, output_file)