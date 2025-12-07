import streamlit as st
import pandas as pd
import pickle
import os
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

# ================================
# CONFIG
# ================================
OUTPUT_DIR = "./custom_forecasting_model/output"
MODEL_FILE = os.path.join(OUTPUT_DIR, "prophet_model.pkl")

# ================================
# PAGE SETTINGS
# ================================
st.set_page_config(page_title="Prophet Forecast Viewer", layout="wide")
st.title("🔮 Energy Forecasting App (Prophet Model)")
st.write("Predict long-term values using a saved Prophet model with clear interpretation.")

# ================================
# LOAD PROPHET MODEL
# ================================
st.subheader("📁 Load Prophet Model")

if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Model file not found: {MODEL_FILE}")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

st.success("✅ Prophet model loaded successfully!")

# ================================
# DETERMINE LAST TRAINING YEAR
# ================================
temp_df = model.make_future_dataframe(periods=1, freq="Y")
temp_fc = model.predict(temp_df)

last_year_in_data = temp_fc['ds'].dt.year.max()

st.info(f"📌 Last available year in the training dataset: **{last_year_in_data}**")

# ================================
# USER INPUT: SELECT YEAR
# ================================
st.subheader("🧮 Predict a Specific Future Year")

future_year = st.number_input(
    "Enter a future year to predict:",
    min_value=last_year_in_data + 1,
    max_value=2100,
    value=2040,
    step=1,
    help="Prophet will forecast the value for this specific year."
)

years_ahead = future_year - last_year_in_data
st.write(f"➡️ Forecasting **{years_ahead} years** beyond the dataset.")

# ================================
# GENERATE FORECAST (+1 FIX)
# ================================
future_df = model.make_future_dataframe(periods=years_ahead + 1, freq="Y")
forecast = model.predict(future_df)

# ================================
# GET THE SPECIFIC FORECASTED ROW
# ================================
pred_row = forecast[forecast["ds"].dt.year == future_year]

st.subheader("📈 Forecast Result")

if pred_row.empty:
    st.error("❌ Forecast for selected year not found.")
else:
    pred_row = pred_row.iloc[-1]
    yhat = float(pred_row["yhat"])
    lower = float(pred_row["yhat_lower"])
    upper = float(pred_row["yhat_upper"])

    # MAIN FORECAST DISPLAY
    st.success(f"""
    ## 🔮 Forecast for **{future_year}**

    ### 📌 Predicted Value  
    **{yhat:,.2f}**  
    Expected value based on Prophet's trend model.

    ### 📉 Lower Bound (95% confidence)  
    **{lower:,.2f}**  
    Indicates the conservative scenario (slow growth or downturn).

    ### 📈 Upper Bound (95% confidence)  
    **{upper:,.2f}**  
    Indicates the optimistic scenario (rapid growth or expansion).
    """)

    # ================================
    # INTERPRETATION SECTION
    # ================================
    st.subheader("🧠 Interpretation of Results")

    st.info(f"""
    In **{future_year}**, the model predicts that the indicator will most likely be around  
    **{yhat:,.2f} units**.

    The true value could realistically fall anywhere between  
    **{lower:,.2f} → {upper:,.2f}**, depending on economic conditions, demand,
    population growth, and energy adoption rates.

    - The **Predicted Value** is the central estimate.
    - The **Lower Bound** represents a slow-growth or low-demand future.
    - The **Upper Bound** represents a high-demand or fast-growth future.

    These bounds come from Prophet's confidence interval and reflect
    uncertainty as we forecast farther into the future.
    """)

# ================================
# FORECAST TABLE
# ================================
st.subheader("📄 Forecast Table (Last Generated Years)")
st.dataframe(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20),
    use_container_width=True
)

# ================================
# INTERACTIVE FORECAST PLOT
# ================================
st.subheader("📊 Interactive Forecast Visualization")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig, use_container_width=True)

# ================================
# PNG IMAGES
# ================================
st.subheader("🖼 Saved Prophet Output Images")

forecast_png = os.path.join(OUTPUT_DIR, "prophet_forecast_plot.png")
components_png = os.path.join(OUTPUT_DIR, "prophet_components_plot.png")

if os.path.exists(forecast_png):
    st.image(forecast_png, caption="Forecast Plot")
else:
    st.warning("⚠️ Forecast PNG not found.")

if os.path.exists(components_png):
    st.image(components_png, caption="Trend Components Plot")
else:
    st.warning("⚠️ Components PNG not found.")
