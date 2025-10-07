# PyMathematics - Machine Learning PIT

This project contains scripts to process and clean energy and economy data for the Philippines, preparing it for machine learning tasks.

## Prerequisites

- Python 3.x
- `pip` for installing Python packages

## Project Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**

    *   **Windows:**
        ```bash
        python -m venv ml-pit
        ml-pit\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv ml-pit
        source ml-pit/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The data preparation process is divided into several scripts that need to be run in order. These scripts are located in the `data_prep` directory.

1.  **Initial Preprocessing:**
    This script reads the raw data from the `/resources` directory, combines them, and pivots the data into a wide format.
    ```bash
    python data_prep/energy_data_preprocess.py
    ```

2.  **Data Cleaning:**
    This script removes a malformed header row from the processed data.
    ```bash
    python data_prep/clean_data.py
    ```

3.  **Handling Missing Data:**
    This script drops columns with a high percentage of missing values and imputes the rest.
    ```bash
    python data_prep/handle_missing_data.py
    ```

4.  **Reshaping the Data:**
    This script converts the data from a wide to a long format, which is more suitable for machine learning.
    ```bash
    python data_prep/convert_to_long.py
    ```

## Final Dataset

After running all the data preparation scripts, the final, cleaned dataset will be available at `data/processed_energy_data_long.csv`. This dataset is ready to be used for training an AI model.

## Custom Forecasting Model (from Scratch)

This project also includes a simple linear regression model for forecasting, built from scratch to demonstrate the underlying principles.

1.  **Train the Custom Model:**
    This script trains the model and saves it to a `.pkl` file.
    ```bash
    cd custom_forecasting_model
    python custom_linear_regression_forecast.py
    ```
    This will also save a plot of the forecast as `custom_forecast_plot.png`.

2.  **Make Predictions with the Custom Model:**
    Use the `predict.py` script to make predictions for a specific year using the saved model.
    ```bash
    # Navigate to the directory if you are not already there
    cd custom_forecasting_model

    # Run a prediction for a specific year
    python predict.py 2035
    ```

## Streamlit Application

A Streamlit application is provided to interact with the custom forecasting model.

1.  **Install all dependencies:**
    Make sure you have installed all the required packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit app:**
    Navigate to the `custom_forecasting_model` directory and run the app:
    ```bash
    cd custom_forecasting_model
    streamlit run streamlit_app.py
    ```
    This will open the application in your web browser, allowing you to input a year and get predictions, as well as view the forecast plot.
