import streamlit as st
import pickle
import os
import pandas as pd
from PIL import Image

# --- Configuration ---
MODEL_FILE = os.path.join('custom_forecasting_model/output', 'random_forest_model.pkl')
PLOT_FILE = os.path.join('custom_forecasting_model/output', 'random_forest_forecast_plot.png')

# --- Load the Model and Metrics ---
@st.cache_resource  # Cache the model loading
def load_model():
    try:
        with open(MODEL_FILE, 'rb') as f:
            loaded_data = pickle.load(f)
            
        # Handle both old (model only) and new (model + metrics) formats
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            # New format with metrics
            return {
                'model': loaded_data['model'],
                'metrics': loaded_data.get('metrics', {})
            }
        else:
            # Old format - just the model
            return {
                'model': loaded_data,
                'metrics': {}
            }
            
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_FILE}. Please run the training script first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model and metrics
model_data = load_model()
model = model_data['model'] if model_data else None
metrics = model_data.get('metrics', {}) if model_data else {}

# --- Streamlit App ---
st.set_page_config(page_title="Electricity Consumption Forecast (Random Forest)", layout="centered")

st.title("Electricity Consumption Forecast (Random Forest)")
st.write("This app uses a Random Forest model to forecast electric power consumption per capita.")

if model is not None:
    # Display model metrics if available
    if metrics and metrics.get('test_data_available'):
        st.subheader("Model Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.2f}", 
                     help="Root Mean Squared Error - Lower is better")
        with col2:
            st.metric("MAE", f"{metrics.get('mae', 'N/A'):.2f}",
                     help="Mean Absolute Error - Lower is better")
        with col3:
            st.metric("R² Score", f"{metrics.get('r2', 'N/A'):.4f}",
                     help="R-squared - Closer to 1 is better")
    elif metrics:
        st.info("No test data was available for evaluation. Metrics are not available.")
    
    st.subheader("Make a Prediction")
    
    # Create input field for the year
    input_features = {}
    
    # Single input for year since we're working with time series
    year = st.number_input(
        "Enter Year:",
        min_value=1960,
        max_value=2100,
        value=2030,
        step=1
    )
    
    # Store the year in the input features
    input_features['Year'] = year
    
    if st.button("Get Prediction"):
        try:
            # Prepare input data for prediction
            # Assuming the model expects a 2D array with the year
            input_data = [[year]]
            prediction = model.predict(input_data)[0]
            
            st.success(
                f"Predicted Electric Power Consumption: **{prediction:.2f} kWh per capita**\n"
                f"for Year: {year}"
            )
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(f"Model expects {model.n_features_in_} features, but got 1")
    
    # Feature importance section removed as requested
    
    # Display forecast visualization if available
    st.subheader("Forecast Visualization")
    if os.path.exists(PLOT_FILE):
        st.image(Image.open(PLOT_FILE), 
                caption="Random Forest Model Forecast", 
                use_container_width=True)
    else:
        st.warning(f"Forecast plot not found at {PLOT_FILE}. Please run the training script to generate the plot.")
else:
    st.info("Model not loaded. Please ensure the training script has been run successfully.")

# Add some spacing at the bottom
st.markdown("\n\n---\n")
st.markdown("*Note: This app uses a Random Forest model for forecasting.*")
