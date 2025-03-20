import streamlit as st
import pickle
import numpy as np

# Load Model and Scaler
with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("🔍 Product Status Prediction using CatBoost")

total_sales = st.number_input("Total Sales", min_value=0.0, step=0.1, value=1000.0)
units_sold = st.number_input("Units Sold", min_value=0, step=1, value=10)
avg_review_score = st.number_input("Average Review Score", min_value=0.0, max_value=5.0, step=0.1, value=4.5)
repeat_customers = st.number_input("Repeat Customers", min_value=0, step=1, value=2)
original_price = st.number_input("Original Price", min_value=0.0, step=0.1, value=50.0)

if st.button("🚀 Predict Product Status"):
    # Prepare Features
    features = np.array([total_sales, units_sold, avg_review_score, repeat_customers, original_price]).reshape(1, -1)
    
    # Normalize
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # Display Result
    label_map = {-1: "❌ Flop", 0: "⚖ Neutral", 1: "✅ Hit"}
    st.success(f"🎯 Predicted Product Status: {label_map[int(prediction)]}")