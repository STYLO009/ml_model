from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Global variable to track app status
app_status = {
    "status": "running",
    "model_loaded": False,
    "preprocessing_loaded": False,
    "start_time": datetime.now().isoformat()
}

try:
    # Load your trained model
    model = joblib.load('loan_model.pkl')
    app_status["model_loaded"] = True
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    # Load preprocessing info
    preprocessing_info = joblib.load('preprocessing_info.pkl')
    app_status["preprocessing_loaded"] = True
    print("✅ Preprocessing info loaded successfully!")
except Exception as e:
    print(f"❌ Error loading preprocessing info: {e}")
    preprocessing_info = None

def prepare_features(input_data):
    """Prepare features to match training data structure"""
    # Create a DataFrame with all expected columns
    feature_template = pd.DataFrame(columns=preprocessing_info['feature_names'])
    
    # Add the input data
    for col in ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']:
        if col in input_data:
            feature_template[col] = [input_data[col]]
    
    # Fill missing columns with 0 (for one-hot encoded features)
    feature_template = feature_template.fillna(0)
    
    return feature_template

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status_code = 200 if (app_status["model_loaded"] and app_status["preprocessing_loaded"]) else 500
    return jsonify({
        "status": app_status["status"],
        "model_loaded": app_status["model_loaded"],
        "preprocessing_loaded": app_status["preprocessing_loaded"],
        "timestamp": app_status["start_time"],
        "version": "1.0.0"
    }), status_code

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not app_status["model_loaded"] or not app_status["preprocessing_loaded"]:
            return jsonify({"error": "Model or preprocessing not loaded"}), 500
            
        # Get data from request
        data = request.json
        
        # Validate input
        required_fields = ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Prepare features to match training data
        input_features = prepare_features(data)
        
        # Make prediction
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)
        
        # Return results
        result = {
            'approved': bool(prediction[0]),
            'probability': float(probability[0][1]),
            'status': 'approved' if prediction[0] else 'rejected',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Loan Prediction API...")
    print("📊 Health check available at: http://localhost:5000/health")
    print("🔮 Prediction endpoint: http://localhost:5000/predict")
    print("🏠 Homepage: http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0')