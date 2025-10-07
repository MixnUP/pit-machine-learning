
# Data Preparation for AI Model

This document outlines the steps taken to clean and prepare the dataset for training an AI model.

## Data Source

The initial datasets were sourced from the Humanitarian Data Exchange (HDX): [data.humdata.org](https://data.humdata.org/)

The raw data is located in the `/resources` directory and consists of the following files:
- `economy-and-growth_phl.csv`
- `energy-and-mining_phl.csv`

## Step 1: Initial Preprocessing

The first step in the data preparation process was to run the `energy_data_preprocess.py` script. This script performs the following actions:

1.  **Reads Data:** It reads all CSV files from the `/resources` directory.
2.  **Combines Datasets:** It combines the data from all CSV files into a single dataset.
3.  **Pivots to Wide Format:** It transforms the data into a wide format, where each row represents a year and each column represents a unique indicator.
4.  **Saves Processed Data:** The resulting dataset is saved as `processed_energy_data.csv` in the `/data` directory.

## Step 2: Data Cleaning

The `clean_data.py` script was used to perform an initial cleaning of the `processed_energy_data.csv` file.

-   **Action:** It removes a malformed header row (the second line of the file) that does not contain valid data.
-   **Output:** A new file named `processed_energy_data_cleaned.csv` is created in the `/data` directory.

## Step 3: Handling Missing Data

The `handle_missing_data.py` script was used to address the large number of missing values in the dataset.

1.  **Column Removal:** It identifies and removes any column that has more than 50% missing values.
2.  **Imputation:** It fills the remaining missing values using a "forward fill" method, which propagates the last valid observation forward. A "backward fill" is also used to handle any missing values at the beginning of the dataset.
-   **Output:** The cleaned and imputed data is saved to `processed_energy_data_final.csv` in the `/data` directory.

## Step 4: Reshaping the Data

The `convert_to_long.py` script was used to transform the data into a format that is more suitable for machine learning.

-   **Action:** It converts the wide-format data into a long format. The resulting dataset has three columns: `Year`, `Indicator`, and `Value`.
-   **Output:** The final, long-format data is saved to `processed_energy_data_long.csv` in the `/data` directory.

## Final Dataset

The final dataset, `processed_energy_data_long.csv`, is now cleaned, imputed, and structured in a long format, making it ready for feature engineering and model training.
