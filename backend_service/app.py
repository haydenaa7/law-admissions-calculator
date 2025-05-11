# backend_service/app.py
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import joblib
import os
from .custom_transformers import cast_to_object, cast_to_bool, cast_to_string

app = Flask(__name__)
# Configure CORS: Allow requests from your Vercel domain and localhost for testing
# Replace 'https://your-vercel-app-xxxx.vercel.app' with your actual Vercel URL after deployment
# For development, you might use '*' but be more specific for production.
CORS(app, resources={
    r"/predict": {"origins": "https://law-admissions-calculator.vercel.app"},
    r"/schools": {"origins": "https://law-admissions-calculator.vercel.app"}
}) # Adjust port if your local Vercel dev server runs elsewhere

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# Correct path to model artifacts within the backend_service directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')

MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'law_school_admission_model.joblib')
MODEL_CLASSES_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'model_classes.joblib')
TOP_SCHOOLS_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'top_schools.joblib')

model = None
model_classes = None
top_schools_list = None

def load_dependencies():
    global model, model_classes, top_schools_list
    try:
        if model is None:
            print(f"Attempting to load model from {MODEL_PATH}...")
            if not os.path.exists(MODEL_PATH):
                print(f"CRITICAL: Model file not found at {MODEL_PATH}") # More prominent
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
            print("Model successfully loaded.") # Confirmation

        if model_classes is None:
            print(f"Attempting to load model classes from {MODEL_CLASSES_PATH}...")
            if not os.path.exists(MODEL_CLASSES_PATH):
                print(f"CRITICAL: Classes file not found at {MODEL_CLASSES_PATH}")
                raise FileNotFoundError(f"Classes file not found at {MODEL_CLASSES_PATH}")
            model_classes = joblib.load(MODEL_CLASSES_PATH)
            print("Model classes successfully loaded.")

        if top_schools_list is None:
            print(f"Attempting to load top schools from {TOP_SCHOOLS_PATH}...")
            if not os.path.exists(TOP_SCHOOLS_PATH):
                print(f"CRITICAL: Schools file not found at {TOP_SCHOOLS_PATH}")
                raise FileNotFoundError(f"Schools file not found at {TOP_SCHOOLS_PATH}")
            top_schools_list = joblib.load(TOP_SCHOOLS_PATH)
            print("Top schools list successfully loaded.")
        
        print("All dependencies in load_dependencies loaded successfully.")

    except Exception as e:
        print(f"FATAL ERROR during load_dependencies: {e}")
        import traceback
        traceback.print_exc() # Print full traceback
        raise

load_dependencies() # Load on application startup

def to_bool_api(val_str):
    return str(val_str).lower() in ['true', 'yes', '1', 'on']

@app.route('/predict', methods=['POST', 'OPTIONS']) # Add OPTIONS for CORS preflight
def predict():
    if request.method == 'OPTIONS': # Handle CORS preflight
        return _build_cors_preflight_response()
    if model is None or model_classes is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received or invalid content type.'}), 400
        print(f"Received data on backend: {data}")

        year = int(data.get('year'))
        month = int(data.get('month'))
        day_val = int(data.get('day'))
        dayofweek = pd.to_datetime(f"{year}-{month}-{day_val}").dayofweek
        
        gpa_val = data.get('gpa')
        years_out_val = data.get('years_out')
        international_gpa_val = data.get('international_gpa')

        indiv_df_data = {
            'year': [year], 'month': [month], 'day': [day_val], 'dayofweek': [dayofweek],
            'is_in_state': [to_bool_api(data.get('is_in_state', False))],
            'is_fee_waived': [to_bool_api(data.get('is_fee_waived', False))],
            'lsat': [int(data.get('lsat'))],
            'softs': [str(data.get('softs'))],
            'urm': [to_bool_api(data.get('urm', False))],
            'non_trad': [to_bool_api(data.get('non_trad', False))],
            'gpa': [float(gpa_val) if gpa_val and str(gpa_val).lower() not in ['nan', 'none', ''] else np.nan],
            'is_international': [to_bool_api(data.get('is_international', False))],
            'international_gpa': [str(international_gpa_val) if international_gpa_val and str(international_gpa_val).lower() not in ['nan', 'none', ''] else np.nan],
            'years_out': [int(years_out_val) if years_out_val and str(years_out_val).lower() not in ['nan', 'none', ''] else np.nan],
            'is_military': [to_bool_api(data.get('is_military', False))],
            'is_character_and_fitness_issues': [to_bool_api(data.get('is_character_and_fitness_issues', False))],
            'school_name': [str(data.get('school_name'))]
        }
        indiv_df = pd.DataFrame(indiv_df_data)
        probabilities = model.predict_proba(indiv_df)[0]
        
        output_dict = {class_label: probabilities[i] for i, class_label in enumerate(model_classes)}
        most_likely_outcome = max(output_dict, key=output_dict.get)

        response_payload = {
            'probabilities': output_dict,
            'most_likely_outcome': most_likely_outcome,
            'school_name': data.get('school_name')
        }
        return jsonify(response_payload)

    except KeyError as e:
        print(f"KeyError during prediction: {e}")
        return jsonify({'error': f'Missing field in input data: {e}'}), 400
    except ValueError as e:
        print(f"ValueError during prediction: {e}")
        return jsonify({'error': f'Invalid data type for a field: {e}'}), 400
    except FileNotFoundError as e:
        print(f"FileNotFoundError during prediction: {e}")
        return jsonify({'error': 'A required model file was not found on the server.'}), 500
    except Exception as e:
        print(f"Unhandled error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/schools', methods=['GET', 'OPTIONS']) # Add OPTIONS for CORS preflight
def get_schools():
    if request.method == 'OPTIONS': # Handle CORS preflight
        return _build_cors_preflight_response()
    if top_schools_list is None:
        return jsonify({'error': 'School list not available on server.'}), 500
    return jsonify({'available_schools': sorted(top_schools_list)})

def _build_cors_preflight_response():
    response = make_response()
    # These headers are set by flask_cors automatically for the actual request,
    # but for OPTIONS, we might need to be explicit if not handled by flask_cors for OPTIONS.
    # However, flask_cors should handle OPTIONS requests if configured properly.
    # If still issues, you might add:
    # response.headers.add("Access-Control-Allow-Origin", "*") # Or your specific Vercel domain
    # response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization")
    # response.headers.add('Access-Control-Allow-Methods', "GET,POST,OPTIONS")
    return response # Flask-CORS should handle the headers

# For Render, it will use a WSGI server like Gunicorn, so this __main__ block is for local testing.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
