
import streamlit as st
import pickle
import os
import pandas as pd # For displaying the plot
from PIL import Image # For displaying the plot

# --- Configuration ---
MODEL_FILE = os.path.join('./custom_forecasting_model/output', 'linear_regression_model.pkl')
PLOT_FILE = os.path.join('./custom_forecasting_model/output', 'custom_forecast_plot.png')

# --- Load the Model ---
@st.cache_resource # Cache the model loading
def load_model():
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        return model['m'], model['c']
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_FILE}. Please run the training script first.")
        return None, None

m, c = load_model()

# --- Streamlit App ---
st.set_page_config(page_title="Electricity Consumption Forecast", layout="centered")

st.title("Electricity Consumption Forecast (Custom Linear Regression)")
st.write("This app uses a custom-built linear regression model to forecast electric power consumption per capita.")

if m is not None and c is not None:
    st.subheader("Make a Prediction")
    year_to_predict = st.number_input("Enter a year to predict:", min_value=1960, max_value=2100, value=2030, step=1)

    if st.button("Get Prediction"):
        prediction = m * year_to_predict + c
        st.success(f"Predicted Electric Power Consumption for {year_to_predict}: **{prediction:.2f} kWh per capita**")

    st.subheader("Forecast Visualization")
    if os.path.exists(PLOT_FILE):
        st.image(Image.open(PLOT_FILE), caption="Historical Data, Model Fit, and Forecast", use_container_width=True)
    else:
        st.warning(f"Forecast plot not found at {PLOT_FILE}. Please run the training script to generate the plot.")

else:
    st.info("Model not loaded. Please ensure the training script has been run successfully.")

