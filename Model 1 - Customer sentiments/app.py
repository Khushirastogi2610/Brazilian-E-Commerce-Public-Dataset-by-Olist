from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model, vectorizer, and label encoder
with open("catboost_text_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("text_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("label_encoder.pkl", "rb") as enc_file:
    label_encoder = pickle.load(enc_file)

app = Flask(__name__)

@app.route('/')
def home():
    return "CatBoost Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if "text" not in data:
            return jsonify({"error": "Missing 'text' key in request"}), 400

        # Transform input text using the same vectorizer
        text_vectorized = vectorizer.transform([data["text"]])  # Convert text to TF-IDF vector
        text_dense = text_vectorized.toarray()  # Ensure it's in dense format

        # Make prediction
        prediction = model.predict(text_dense)
        sentiment_label = label_encoder.inverse_transform([int(prediction[0])])[0]  # Convert back to original labels

        return jsonify({"prediction": sentiment_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
