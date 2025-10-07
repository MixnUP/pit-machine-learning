import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import math

# --- Configuration ---
INPUT_FILE = os.path.join('data', 'processed_energy_data_long.csv')
INDICATOR_TO_FORECAST = 'Electric power consumption (kWh per capita)'
OUTPUT_DIR = './custom_forecasting_model/output'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'linear_regression_model.pkl')
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, 'custom_forecast_plot.png')
FORECAST_YEARS = 10
MIN_DATA_POINTS_FOR_TEST = 5 # Minimum number of data points to hold out for testing

# --- Load and Prepare Data ---
print(f"Loading data from {INPUT_FILE}...")
try:
    df_long = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: The file {INPUT_FILE} was not found. Please make sure you have run the data preparation scripts.")
    exit()

print("Filtering data for the indicator:", INDICATOR_TO_FORECAST)
df_indicator = df_long[df_long['Indicator'] == INDICATOR_TO_FORECAST].copy()

# Ensure data is sorted by year
df_indicator.sort_values('Year', inplace=True)

# --- Train-Test Split ---
print("\n--- Splitting data into training and testing sets ---")
if len(df_indicator) >= MIN_DATA_POINTS_FOR_TEST * 2: # Ensure there's enough data for a meaningful split
    test_size = MIN_DATA_POINTS_FOR_TEST
    train_df = df_indicator.iloc[:-test_size]
    test_df = df_indicator.iloc[-test_size:]
else:
    # Not enough data, use all for training and skip evaluation
    train_df = df_indicator
    test_df = pd.DataFrame()

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

# --- Model Training (on training data only) ---
print("\n--- Training custom Linear Regression model on the training set ---")

x_train = train_df['Year'].values
y_train = train_df['Value'].values
n_train = len(x_train)

# Calculate sum of x, y, xy, x^2 for training data
if n_train > 1:
    sum_x_train = sum(x_train)
    sum_y_train = sum(y_train)
    sum_xy_train = sum(x_train * y_train)
    sum_x_sq_train = sum(x_train**2)

    # Calculate slope (m) and intercept (c)
    try:
        m = (n_train * sum_xy_train - sum_x_train * sum_y_train) / (n_train * sum_x_sq_train - sum_x_train**2)
        c = (sum_y_train - m * sum_x_train) / n_train
    except ZeroDivisionError:
        print("Error: Cannot calculate linear regression, division by zero.")
        m, c = 0, 0
else:
    print("Not enough data to train the model.")
    m, c = 0, 0


print(f"Training complete. Model parameters: slope (m) = {m}, intercept (c) = {c}")

# --- Save the Model ---
print(f"\n--- Saving model to {MODEL_FILE} ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(MODEL_FILE, 'wb') as f:
    pickle.dump({'m': m, 'c': c}, f)
print("Model saved successfully.")

# --- Evaluation on Test Set ---
if not test_df.empty:
    print("\n--- Evaluating model on the testing set ---")
    x_test = test_df['Year'].values
    y_test = test_df['Value'].values
    predictions = [m * year + c for year in x_test]

    # Calculate RMSE from scratch
    squared_errors = [(pred - actual)**2 for pred, actual in zip(predictions, y_test)]
    mse = sum(squared_errors) / len(squared_errors)
    rmse = math.sqrt(mse)

    print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.2f}")
else:
    print("\n--- Skipping evaluation: Not enough data for a test set ---")
    predictions = [] # for plotting

# --- Forecasting ---
print(f"\n--- Generating forecast for the next {FORECAST_YEARS} years ---")
last_year = int(df_indicator['Year'].max())
future_years = range(last_year + 1, last_year + 1 + FORECAST_YEARS)
forecasted_values = [m * year + c for year in future_years]

for year, value in zip(future_years, forecasted_values):
    print(f"Year: {year}, Forecasted Value: {value:.2f}")

# --- Visualization ---
print(f"\n--- Plotting and saving the forecast to {OUTPUT_PLOT_FILE} ---")
plt.figure(figsize=(12, 7))
plt.scatter(train_df['Year'], train_df['Value'], label='Training Data', color='blue')
if not test_df.empty:
    plt.scatter(test_df['Year'], test_df['Value'], label='Testing Data (Actual)', color='orange')
    plt.plot(test_df['Year'], predictions, color='purple', linestyle='--', label='Predictions on Test Data')

plt.plot(train_df['Year'], m * train_df['Year'] + c, color='red', label='Linear Regression Fit')
plt.plot(future_years, forecasted_values, color='green', linestyle='--', label='Future Forecast')
plt.title(f'Forecast for: {INDICATOR_TO_FORECAST}')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_PLOT_FILE)
print("Plot saved successfully.")