import streamlit as st
import requests

st.title("Sentiment Analysis with CatBoost")

user_input = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):
    if user_input:
        response = requests.post("http://127.0.0.1:5001/predict", json={"text": user_input})
        result = response.json()
        if "prediction" in result:
            st.success(f"Predicted Sentiment: {result['prediction']}")
        else:
            st.error(f"Error: {result['error']}")
    else:
        st.warning("Please enter some text.")