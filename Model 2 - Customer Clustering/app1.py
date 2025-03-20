from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the saved model
with open("rfm_kmeans.pkl", "rb") as model_file:
    scaler, kmeans = pickle.load(model_file)

# Define customer segments
segment_labels = {
    0: "Frequent Buyers",
    1: "Loyal Buyers",
    2: "Budget Buyers",
    3: "VIP Buyers"
}

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_segment():
    try:
        # Get JSON request data
        data = request.get_json()

        # Extract input values
        recency = float(data["recency"])
        frequency = float(data["frequency"])
        monetary_value = float(data["monetary_value"])

        # Transform input using the same scaler
        scaled_input = scaler.transform([[recency, frequency, monetary_value]])

        # Predict the cluster
        cluster = kmeans.predict(scaled_input)[0]

        # Get segment label
        segment = segment_labels[cluster]

        return jsonify({"segment": segment})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
