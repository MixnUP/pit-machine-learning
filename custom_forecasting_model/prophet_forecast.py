import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import pickle
import os

# ================================
# CONFIGURATION
# ================================
INPUT_FILE = os.path.join('data', 'processed_energy_data_long.csv')
INDICATOR_TO_FORECAST = 'Electric power consumption (kWh per capita)'
OUTPUT_DIR = './custom_forecasting_model/output'
MODEL_FILE = os.path.join(OUTPUT_DIR, 'prophet_model.pkl')
FORECAST_YEARS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# LOAD & PREPARE DATA
# ================================
print(f"Loading data from {INPUT_FILE}...")

df = pd.read_csv(INPUT_FILE)
df = df[df['Indicator'] == INDICATOR_TO_FORECAST].copy()

# Prophet requires columns: ds (date) and y (value)
df_prophet = df[['Year', 'Value']].copy()
df_prophet.rename(columns={'Year': 'ds', 'Value': 'y'}, inplace=True)

# Convert integer year → datetime
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')

print("Data prepared for Prophet:")
print(df_prophet.head())

# ================================
# TRAIN PROPHET MODEL
# ================================
print("\nTraining Prophet model...")

model = Prophet(
    yearly_seasonality=False,   # No yearly patterns for per-capita electricity
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.2,  # smoother curve
)

model.fit(df_prophet)

print("Prophet training complete.")

# ================================
# SAVE PROPHET MODEL (.pkl)
# ================================
print(f"Saving model to {MODEL_FILE}...")

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")

# ================================
# FORECAST FUTURE YEARS
# ================================
print(f"\nGenerating forecast for next {FORECAST_YEARS} years...")

future = model.make_future_dataframe(periods=FORECAST_YEARS, freq='Y')
forecast = model.predict(future)

# Show last forecasted values
print("\nForecasted Values:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(FORECAST_YEARS))

# ================================
# PLOTS
# ================================
# Main forecast plot
fig1 = model.plot(forecast)
plt.title(f"Prophet Forecast: {INDICATOR_TO_FORECAST}")
plt.xlabel("Year")
plt.ylabel(INDICATOR_TO_FORECAST)
plt.grid(True)
plot_path = os.path.join(OUTPUT_DIR, "prophet_forecast_plot.png")
plt.savefig(plot_path)
print(f"Forecast plot saved to {plot_path}")

# Components plot (trend & changepoints)
fig2 = model.plot_components(forecast)
components_path = os.path.join(OUTPUT_DIR, "prophet_components_plot.png")
plt.savefig(components_path)
print(f"Components plot saved to {components_path}")

# ================================
# TEST LOADING THE MODEL AGAIN
# ================================
print("\nTesting saved model re-load...")

with open(MODEL_FILE, "rb") as f:
    loaded_model = pickle.load(f)

future2 = loaded_model.make_future_dataframe(periods=FORECAST_YEARS, freq='Y')
forecast2 = loaded_model.predict(future2)

print("\nReloaded model prediction (last values):")
print(forecast2[['ds', 'yhat']].tail(5))

print("\nAll done bai! Prophet forecasting system ready.")
