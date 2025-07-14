import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# === Feature columns used by the model === #
features = [
    'Open', 'High', 'Low', 'VWAP', 'Prev. Close', 'Trans.',
    'Diff', 'Range', 'Diff %', 'Range %', 'VWAP %',
    '120 Days', '180 Days', '52 Weeks High', '52 Weeks Low',
    'log_vol', 'log_turnover', 'Year', 'Month', 'Weekday'
]

# === Load cleaned dataset === #
df = pd.read_csv("cleaned_stock_features.csv")
symbols = sorted(df['Symbol'].unique())

st.title("📈 Stock Price Prediction App")

# === Symbol selection === #
selected_symbol = st.selectbox("Select Stock Symbol", symbols)

# === Filter data for selected symbol === #
symbol_df = df[df['Symbol'] == selected_symbol]

# === Get the latest row for prediction === #
if 'Date' in df.columns and df['Date'].notna().all():
    symbol_df = symbol_df.sort_values('Date')
    latest_data = symbol_df.iloc[-1]
else:
    symbol_df = symbol_df.sort_values(['Year', 'Month', 'Weekday'])
    latest_data = symbol_df.iloc[-1]

# === Load model === #
model_path = f"models/{selected_symbol}.pkl"
if not os.path.exists(model_path):
    st.error(f"Model for '{selected_symbol}' not found in `models/` directory.")
    st.stop()

model = joblib.load(model_path)

# === Prepare input for prediction === #
X_latest = latest_data[features].values.reshape(1, -1)
predicted_close = abs(model.predict(X_latest)[0])*1000  # ✅ Make sure it's positive

# === Predicted open approximation === #
predicted_open = abs(latest_data['Open'])*1000  # ✅ Force open to positive too

# === Calculate predicted % change === #
predicted_close_change = ((predicted_close - predicted_open) / predicted_open) * 100 if predicted_open != 0 else 0
predicted_open_change = 0.0  # No change since it's same as previous

# === Display predictions === #
st.subheader("Next Day Prediction (Model Output)")
st.markdown(f"**Predicted Close Price:** ₹{predicted_close:.4f}")
st.markdown(f"**Approx. Open Price (Previous Open):** ₹{predicted_open:.4f}")
st.markdown(f"**Predicted Close % Change (from Open):** `{predicted_close_change:.2f}%` {'✅' if predicted_close_change > 0 else '❗️' if predicted_close_change < 0 else ''}")
st.markdown(f"**Predicted Open % Change (vs itself):** `{predicted_open_change:.2f}%`")
