from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# Load the assets
model = joblib.load('tuned_diabetes_model_rf.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        logger.info(f"Incoming prediction request: {data}")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data], columns=feature_names)
        
        # Scale data
        scaled_data = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_data, columns=feature_names)
        
        # Predict
        prediction = model.predict(scaled_df)[0]
        probability = model.predict_proba(scaled_df)[0].tolist()
        
        result = {
            "prediction": int(prediction),
            "status": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "confidence_scores": {
                "non_diabetic": probability[0],
                "diabetic": probability[1]
            }
        }

        logger.info(f"Prediction result: {result['status']} (Confidence: {max(probability):.2%})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
