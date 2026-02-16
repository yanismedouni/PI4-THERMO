"""
Energy Consumption Data Processing Script

This script processes energy consumption data from a CSV/Excel file:
1. Adds solar production to grid consumption
2. Removes EV consumption from grid if it exceeds 3kW threshold
3. Outputs a clean CSV with only relevant columns
"""

import pandas as pd


def add_solar_to_grid(df):
    """
    Add solar production (solar + solar2) to grid consumption.
    """
    df['solar'] = df['solar'].fillna(0) + df['solar2'].fillna(0)
    df['grid'] = df['grid'].fillna(0) + df['solar']
    return df


def remove_ev_consumption_above_threshold(df, threshold_kw=3.0):
    """
    Remove EV consumption (car1 + car2) from grid only if total exceeds threshold.
    """
    df['car'] = df['car1'].fillna(0) + df['car2'].fillna(0)

    df.loc[df['car'] > threshold_kw, 'grid'] = (
        df['grid'] - df['car']
    )

    return df


def process_energy_data(input_file, output_file, ev_threshold_kw=3.0):
    """
    Main function to process energy consumption data.
    """

    # Automatically detect file type
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)

    print(f"Loaded {len(df)} rows from {input_file}")
    print(f"Original columns: {df.columns.tolist()}")

    # SAVE ORIGINAL GRID BEFORE MODIFICATIONS (FIXED!)
    df['grid_original'] = df['grid'].values.copy()  # ← .values.copy()
    
    # Apply transformations
    df = add_solar_to_grid(df)
    df = remove_ev_consumption_above_threshold(df, threshold_kw=ev_threshold_kw)

    # Keep only relevant columns (including original grid)
    output_columns = ['dataid', 'local_15min', 'solar', 'car', 'grid_original', 'grid']
    df_output = df[output_columns].copy()

    # Save output
    try:
        df_output.to_csv(output_file, index=False)
        print(f"\n✓ Successfully saved to {output_file}")
    except PermissionError:
        print(f"\n✗ ERROR: Cannot write to {output_file}")
        print("  The file may be open in Excel. Please close it and try again.")
        return None

    print("\nFirst few rows:")
    print(df_output.head())
    print("\nSummary statistics:")
    print(df_output.describe())

    return df_output


# ==========================
#        MAIN PROGRAM
# ==========================
if __name__ == "__main__":

    input_path = "../data/15minute_data_austin.csv"
    output_path = "../output/processed_energy_data.csv"

    process_energy_data(input_path, output_path, ev_threshold_kw=3.0)
