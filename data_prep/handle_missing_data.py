
import pandas as pd
import os

# --- Configuration ---
INPUT_FILE = os.path.join('data', 'processed_energy_data_cleaned.csv')
OUTPUT_FILE = os.path.join('data', 'processed_energy_data_final.csv')
MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with more than 50% missing values

# --- Load the dataset ---
print(f"Loading data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: The file {INPUT_FILE} was not found. Please make sure you have run the previous script to create it.")
    exit()

print("Dataset loaded successfully.")
print(f"Original shape of the dataset: {df.shape}")

# --- Step 1: Assess Missing Data ---
print("\n--- Assessing missing data ---")
missing_percentage = df.isnull().sum() / len(df)

# --- Step 2: Drop Columns with High Missingness ---
print(f"\n--- Dropping columns with more than {MISSING_VALUE_THRESHOLD:.0%} missing values ---")
cols_to_drop = missing_percentage[missing_percentage > MISSING_VALUE_THRESHOLD].index
df_dropped = df.drop(columns=cols_to_drop)

print(f"Dropped {len(cols_to_drop)} columns.")
print(f"New shape of the dataset: {df_dropped.shape}")

# --- Step 3: Impute Remaining Missing Values ---
print("\n--- Imputing remaining missing values using forward fill ---")
# First, sort by year to ensure correct forward fill
df_dropped = df_dropped.sort_values(by='Year').reset_index(drop=True)
df_imputed = df_dropped.ffill()

# Check if there are any remaining missing values (for columns that had missing values at the beginning)
remaining_missing = df_imputed.isnull().sum().sum()
if remaining_missing > 0:
    print(f"There are still {remaining_missing} missing values. Applying backfill to handle these cases.")
    df_imputed = df_imputed.bfill()


print("Imputation complete.")

# --- Save the cleaned data ---
print(f"\n--- Saving cleaned data to {OUTPUT_FILE} ---")
df_imputed.to_csv(OUTPUT_FILE, index=False)
print("Script finished successfully.")
