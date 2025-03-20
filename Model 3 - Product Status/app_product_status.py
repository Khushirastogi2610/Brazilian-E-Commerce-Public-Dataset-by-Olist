from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Extract Features
    features = np.array([
        data["total_sales"],
        data["units_sold"],
        data["avg_review_score"],
        data["repeat_customers"],
        data["original_price"]
    ]).reshape(1, -1)

    # Normalize Features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    
    # Map Predictions
    label_map = {-1: "Flop", 0: "Neutral", 1: "Hit"}
    result = {"product_status": label_map[int(prediction)]}
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
