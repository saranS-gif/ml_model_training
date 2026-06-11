from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# Logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load("tuned_diabetes_model_rf.joblib")
    scaler = joblib.load("scaler.joblib")
    print("✅ Model and Scaler Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading Files: {e}")
    raise

# Features used during training
FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

@app.route("/")
def home():
    return jsonify({
        "message": "Diabetes Prediction API Running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        # Check missing fields
        missing = [f for f in FEATURE_NAMES if f not in data]

        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}"
            }), 400

        # Create dataframe
        input_df = pd.DataFrame(
            [[data[col] for col in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = int(model.predict(scaled_input)[0])

        # Probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]

            diabetic_prob = float(probabilities[1])
            non_diabetic_prob = float(probabilities[0])
        else:
            diabetic_prob = None
            non_diabetic_prob = None

        result = {
            "prediction": prediction,
            "status": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "confidence_scores": {
                "non_diabetic": non_diabetic_prob,
                "diabetic": diabetic_prob
            }
        }

        return jsonify(result)

    except Exception as e:
        logging.error(str(e))

        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
