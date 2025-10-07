
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# --- Configuration ---
INPUT_FILE = os.path.join('data', 'processed_energy_data_long.csv')
INDICATOR_TO_FORECAST = 'Access to electricity (% of population)'
FORECAST_PERIOD_YEARS = 10
OUTPUT_PLOT_FILE = 'electricity_access_forecast.png'

# --- Load and Prepare Data ---
print(f"Loading data from {INPUT_FILE}...")
try:
    df_long = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: The file {INPUT_FILE} was not found. Please make sure you have run the previous scripts.")
    exit()

print("Filtering data for the indicator:", INDICATOR_TO_FORECAST)
df_indicator = df_long[df_long['Indicator'] == INDICATOR_TO_FORECAST]

# Prepare data for Prophet
df_prophet = df_indicator[['Year', 'Value']].copy()
df_prophet.rename(columns={'Year': 'ds', 'Value': 'y'}, inplace=True)
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')

print("Data prepared for Prophet:")
print(df_prophet.head())

# --- Train the Prophet Model ---
print("\n--- Training the Prophet model ---")
model = Prophet()
model.fit(df_prophet)

# --- Make Future Predictions ---
print(f"\n--- Making a {FORECAST_PERIOD_YEARS}-year forecast ---")
future = model.make_future_dataframe(periods=FORECAST_PERIOD_YEARS, freq='Y')
forecast = model.predict(future)

# --- Display and Save the Forecast ---
print("\nForecasted values:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(FORECAST_PERIOD_YEARS))

print(f"\n--- Plotting and saving the forecast to {OUTPUT_PLOT_FILE} ---")
fig = model.plot(forecast)
plt.title(f'Forecast for: {INDICATOR_TO_FORECAST}')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.savefig(OUTPUT_PLOT_FILE)
print("Plot saved successfully.")
