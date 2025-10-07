import pandas as pd
import os

INPUT_FILE = os.path.join('data', 'processed_energy_data_long.csv')
OUTPUT_DIR = './custom_forecasting_model/output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'INDICATORS.md')

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    unique_indicators = df['Indicator'].unique()

    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Unique Indicators in the Dataset\n\n")
        for indicator in sorted(unique_indicators):
            f.write(f"- `{indicator}`\n")

    print(f"Successfully saved the list of indicators to {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"Error: The file {INPUT_FILE} was not found.")