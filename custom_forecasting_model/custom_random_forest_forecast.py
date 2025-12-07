import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
INPUT_FILE = os.path.join('data', 'processed_energy_data_long.csv')
INDICATOR_TO_FORECAST = 'Electric power consumption (kWh per capita)'
OUTPUT_DIR = './custom_forecasting_model/output'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'random_forest_model.pkl')
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, 'random_forest_forecast_plot.png')
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
print("\n--- Training Random Forest Regressor on the training set ---")

# Prepare the data
X_train = train_df[['Year']].values
y_train = train_df['Value'].values

# Create and train the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    min_samples_split=2,
    min_samples_leaf=1
)
model.fit(X_train, y_train)

# Get feature importance
feature_importance = model.feature_importances_[0]
print(f"Training complete. Feature importance (Year): {feature_importance:.4f}")

# --- Save the Model and Metrics ---
print(f"\n--- Saving model and metrics to {MODEL_FILE} ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize metrics
rmse = None
mae = None
r2 = None

try:
    if not test_df.empty:
        print("\n--- Evaluating model on the testing set ---")
        X_test = test_df[['Year']].values
        y_test = test_df['Value'].values
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("\n--- Model Evaluation on Test Set ---")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared (R²) Score: {r2:.4f}")
        print("\nNote: R² ranges from 0 to 1, where 1 indicates perfect fit.")
        print("Lower RMSE and MAE values indicate better model performance.")
    else:
        print("\n--- Skipping evaluation: Not enough data for a test set ---")
        predictions = []  # for plotting

except Exception as e:
    print(f"Error during model evaluation: {str(e)}")

# Create a dictionary to store model and metrics
model_data = {
    'model': model,
    'metrics': {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'test_data_available': not test_df.empty
    }
}

# Save the model and metrics
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model_data, f)
print("Model and metrics saved successfully.")

# --- Forecasting ---
print(f"\n--- Generating forecast for the next {FORECAST_YEARS} years ---")
last_year = int(df_indicator['Year'].max())
future_years = np.array(range(last_year + 1, last_year + 1 + FORECAST_YEARS)).reshape(-1, 1)
forecasted_values = model.predict(future_years)

for year, value in zip(future_years.flatten(), forecasted_values):
    print(f"Year: {int(year)}, Forecasted Value: {value:.2f}")

# --- Visualization ---
print(f"\n--- Plotting and saving the forecast to {OUTPUT_PLOT_FILE} ---")
plt.figure(figsize=(12, 7))

# Plot training and test data
plt.scatter(train_df['Year'], train_df['Value'], label='Training Data', color='blue', alpha=0.5)
if not test_df.empty:
    plt.scatter(test_df['Year'], test_df['Value'], label='Testing Data (Actual)', color='orange', alpha=0.7)
    plt.scatter(test_df['Year'], predictions, color='purple', label='Predictions on Test Data', alpha=0.7)

# Plot model's predictions for training range and future
all_years = np.concatenate([
    train_df['Year'].values, 
    test_df['Year'].values if not test_df.empty else np.array([]),
    future_years.flatten()
])
all_years = np.unique(all_years).reshape(-1, 1)
pred_all = model.predict(all_years)

plt.plot(all_years, pred_all, 'r-', label='Random Forest Prediction')
plt.plot(future_years, forecasted_values, 'g--', label='Future Forecast')
plt.title(f'Forecast for: {INDICATOR_TO_FORECAST}')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_PLOT_FILE)
print("Plot saved successfully.")