
import pandas as pd
import os

# --- Configuration ---
INPUT_FILE = os.path.join('data', 'processed_energy_data_final.csv')
OUTPUT_FILE = os.path.join('data', 'processed_energy_data_long.csv')

# --- Load the dataset ---
print(f"Loading data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: The file {INPUT_FILE} was not found. Please make sure you have run the previous scripts to create it.")
    exit()

print("Dataset loaded successfully.")
print(f"Original shape of the dataset: {df.shape}")

# --- Convert from Wide to Long Format ---
print("\n--- Converting data from wide to long format ---")
df_long = pd.melt(df, id_vars=['Year'], var_name='Indicator', value_name='Value')

print("Conversion complete.")
print(f"New shape of the dataset: {df_long.shape}")


# --- Save the long format data ---
print(f"\n--- Saving long format data to {OUTPUT_FILE} ---")
df_long.to_csv(OUTPUT_FILE, index=False)
print("Script finished successfully.")

