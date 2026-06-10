import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='.')

# Load the pre-trained Logistic Regression model
MODEL_PATH = "logistic_pkl.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features in the exact order expected by the model
        feature_order = [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
        ]
        
        input_features = []
        for feature in feature_order:
            val = request.form.get(feature)
            if val is None:
                return render_template("index.html", error=f"Missing field: {feature}")
            input_features.append(float(val))
            
        # Convert to numpy array and reshape for prediction
        final_features = np.array([input_features])
        
        # Make prediction and calculate probabilities
        prediction = model.predict(final_features)[0]
        prediction_proba = model.predict_proba(final_features)[0]
        confidence = round(prediction_proba[prediction] * 100, 2)

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {'Positive / High Risk' if prediction == 1 else 'Negative / Low Risk'}",
            confidence=f"Confidence: {confidence}%",
            prediction_code=int(prediction)
        )

    except Exception as e:
        return render_template("index.html", error=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    # Render assigns a port dynamically via environment variables
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
