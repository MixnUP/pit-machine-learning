
# PIT: Machine Learning Project Proposal and Implementation

## Project Overview

This project proposes, designs, and implements a machine learning solution for forecasting electric power consumption per capita in the Philippines, addressing a critical aspect of energy and economic development.

### Data Requirements

The project utilizes a dataset sourced from a Philippine context, specifically focusing on economic and energy indicators relevant to the Philippines.

**Goal:** To demonstrate the application of machine learning techniques to solve a real-world problem in the local Philippine environment, providing insights into future energy consumption trends.

## Project Proposal

### 1. Title

**Forecasting Electric Power Consumption Per Capita in the Philippines Using a Custom Linear Regression Model**

### 2. Problem Statement

The Philippines, a rapidly developing nation, faces increasing demands for electric power. Accurate forecasting of electric power consumption per capita is crucial for effective energy planning, infrastructure development, and policy formulation. Unreliable forecasts can lead to energy shortages, inefficient resource allocation, and hinder sustainable economic growth.

This project aims to address the problem of predicting future electric power consumption per capita in the Philippines. By developing a robust forecasting model, we can provide stakeholders with valuable insights into future energy needs, enabling proactive decision-making to ensure energy security and support national development goals.

### 3. Methodology

#### Machine Learning Approach

*   **Type:** Regression (specifically, Time Series Forecasting)
*   **Algorithm:** Custom-built Simple Linear Regression Model
*   **Justification:** A simple linear regression model is chosen for its interpretability and as a foundational "from scratch" implementation to demonstrate the core principles of machine learning. While more complex models exist, this approach allows for a clear understanding of model training, parameter estimation, and forecasting mechanics without relying on high-level libraries for the model itself. It effectively captures the underlying linear trend often present in long-term time series data.

#### Implementation Plan

1.  **Data Acquisition & Collection:**
    *   Raw data (economic and energy indicators for the Philippines) acquired from data.humdata.org.
    *   Files: `economy-and-growth_phl.csv`, `energy-and-mining_phl.csv` located in the `/resources` directory.

2.  **Data Preprocessing & Cleaning:**
    *   **Initial Preprocessing (`data_prep/energy_data_preprocess.py`):** Reads raw CSVs, combines them, pivots to a wide format, and saves as `data/processed_energy_data.csv`.
    *   **Data Cleaning (`data_prep/clean_data.py`):** Removes a malformed header row from `data/processed_energy_data.csv`, saving as `data/processed_energy_data_cleaned.csv`.
    *   **Handling Missing Data (`data_prep/handle_missing_data.py`):** Drops columns with >50% missing values and imputes remaining missing values using forward and backward fill, saving as `data/processed_energy_data_final.csv`.
    *   **Reshaping Data (`data_prep/convert_to_long.py`):** Converts the data from wide to long format, saving as `data/processed_energy_data_long.csv`.

3.  **Feature Engineering/Selection:**
    *   For this simple linear regression model, the primary feature is the `Year`. No complex feature engineering or selection is performed beyond filtering for the target indicator.

4.  **Model Selection & Training:**
    *   **Model:** Custom-built Simple Linear Regression.
    *   **Training (`custom_forecasting_model/custom_linear_regression_forecast.py`):**
        *   Loads `data/processed_energy_data_long.csv`.
        *   Filters for `Electric power consumption (kWh per capita)`.
        *   Splits data into training (80%) and testing (20%) sets chronologically.
        *   Calculates slope (`m`) and intercept (`c`) using standard linear regression formulas on the training data.
        *   Saves `m` and `c` to `custom_forecasting_model/output/linear_regression_model.pkl`.

5.  **Model Evaluation:**
    *   **Metric:** Root Mean Squared Error (RMSE).
    *   **Process:** The `custom_forecasting_model/custom_linear_regression_forecast.py` script calculates RMSE on the test set by comparing predicted values against actual values.

6.  **Interpretation of Results:**
    *   The RMSE provides a quantitative measure of the model's average prediction error.
    *   The generated plot (`custom_forecasting_model/output/custom_forecast_plot.png`) visually represents the model's fit to historical data, its predictions on the test set, and the future forecast, allowing for qualitative assessment of trends.

### 4. Data Description

#### Source

*   **URL:** [data.humdata.org](https://data.humdata.org/)
*   **Philippine Origin:** The datasets (`economy-and-growth_phl.csv`, `energy-and-mining_phl.csv`) are specifically filtered or provided for the Philippines (indicated by `_phl` in filenames), ensuring relevance to the local context.

#### Dataset Details

*   **Features (Columns):**
    *   `Year`: The year of the observation (temporal feature).
    *   `Indicator`: The name of the economic or energy indicator (categorical, used for filtering).
    *   `Value`: The numerical value of the indicator for that year (target variable when filtered).
*   **Target Variable:** For this project, the target variable is `Electric power consumption (kWh per capita)`.
*   **Number of Samples:** Approximately 65 years of data (after cleaning) for each indicator.
*   **Number of Features:** The final long-format dataset (`processed_energy_data_long.csv`) has 3 columns (`Year`, `Indicator`, `Value`) and approximately 22,000 rows (individual observations of indicators over years). For the specific forecasting task, the model uses `Year` as the single feature and `Electric power consumption (kWh per capita)` as the target.

### 5. Expected Output and Impact

*   **Intended Output:**
    *   A custom-built linear regression model (`linear_regression_model.pkl`) capable of forecasting `Electric power consumption (kWh per capita)`.
    *   A visual forecast plot (`custom_forecast_plot.png`) showing historical trends and future predictions.
    *   An interactive Streamlit application (`streamlit_app.py`) for user-friendly prediction input and display.
*   **Potential Use by Stakeholders in the Philippines:**
    *   **Energy Policy Makers:** Can use the forecasts to anticipate future electricity demand, informing decisions on power plant construction, renewable energy investments, and grid expansion.
    *   **Utility Companies:** Can optimize resource allocation, plan maintenance schedules, and manage supply chains more effectively based on predicted consumption patterns.
    *   **Economic Planners:** Can integrate energy consumption forecasts into broader economic models to assess growth potential and identify areas for sustainable development.
    *   **Researchers:** The custom model serves as a foundational example for building more complex time series models tailored to Philippine data.
