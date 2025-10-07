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

## Custom Forecasting Model (from Scratch)

This section details the custom linear regression model built from scratch for forecasting.

### 1. Training and Model Saving (`custom_linear_regression_forecast.py`)

-   **Purpose:** This script trains the custom linear regression model, evaluates its performance, generates a forecast, and saves the trained model parameters.
-   **Input:** `data/processed_energy_data_long.csv`
-   **Process:**
    -   Loads the processed data and filters for the chosen indicator (`Electric power consumption (kWh per capita)`).
    -   Splits the data into training and testing sets (80/20 split, with a minimum test set size).
    -   Calculates the slope (`m`) and intercept (`c`) of the linear regression line using only the training data.
    -   Saves the calculated `m` and `c` coefficients to `custom_forecasting_model/output/linear_regression_model.pkl`.
    -   Evaluates the model's accuracy on the test set using Root Mean Squared Error (RMSE).
    -   Generates a forecast for the next 10 years.
    -   Creates and saves a plot (`custom_forecasting_model/output/custom_forecast_plot.png`) visualizing the historical data, model fit, predictions on test data, and future forecast.
-   **Output:**
    -   `custom_forecasting_model/output/linear_regression_model.pkl` (saved model parameters)
    -   `custom_forecasting_model/output/custom_forecast_plot.png` (visualization of forecast)
    -   RMSE value printed to console.
    -   Future forecast values printed to console.

### 2. Making Predictions (`predict.py`)

-   **Purpose:** This script loads the saved model and uses it to make predictions for a specified year.
-   **Input:** `custom_forecasting_model/output/linear_regression_model.pkl` (the saved model) and a year provided as a command-line argument.
-   **Process:**
    -   Loads the `m` and `c` coefficients from `linear_regression_model.pkl`.
    -   Calculates the predicted value for the input year using the loaded coefficients.
-   **Output:** The predicted value for the given year printed to the console.

### 3. Interactive Application (`streamlit_app.py`)

-   **Purpose:** Provides a user-friendly web interface to interact with the custom forecasting model.
-   **Input:** User-entered year in the web interface.
-   **Process:**
    -   Loads the `m` and `c` coefficients from `linear_regression_model.pkl`.
    -   Takes a year input from the user.
    -   Displays the predicted value.
    -   Displays the `custom_forecast_plot.png` for visual context.
-   **Output:** Predicted value and forecast plot displayed in the web browser.