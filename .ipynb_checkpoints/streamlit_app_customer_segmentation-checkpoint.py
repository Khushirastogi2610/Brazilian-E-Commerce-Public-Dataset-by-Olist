import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("rfm_kmeans.pkl", "rb") as model_file:
    scaler, kmeans = pickle.load(model_file)

# Define customer segments
segment_labels = {
    0: "Frequent Buyers",
    1: "Loyal Buyers",
    2: "Budget Buyers",
    3: "VIP Buyers"
}

# Streamlit UI
st.title("Customer Segmentation Using RFM Analysis")
st.write("Enter Recency, Frequency, and Monetary Values to Predict Customer Segment.")

# Input fields
recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)
frequency = st.number_input("Frequency (Total purchases)", min_value=0, value=5)
monetary_value = st.number_input("Monetary Value (Total amount spent)", min_value=0.0, value=1000.0)

# Predict button
if st.button("Predict Segment"):
    # Transform input using the scaler
    input_data = np.array([[recency, frequency, monetary_value]])
    scaled_input = scaler.transform(input_data)

    # Predict segment
    cluster = kmeans.predict(scaled_input)[0]
    segment = segment_labels[cluster]

    # Display result
    st.success(f"Predicted Customer Segment: **{segment}**")
