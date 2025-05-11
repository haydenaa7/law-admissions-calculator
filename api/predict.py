# api/predict.py
import pandas as pd
import numpy as np
import joblib
import json
import os
from http.server import BaseHTTPRequestHandler # Using this for Vercel handler format

# Determine the correct path to model artifacts
# This works if 'api' is a subdir of the project root where 'model_artifacts' also is
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')

MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'law_school_admission_model.joblib')
MODEL_CLASSES_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'model_classes.joblib')
TOP_SCHOOLS_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'top_schools.joblib')


model = None
model_classes = None
top_schools_list = None

def load_dependencies():
    global model, model_classes, top_schools_list
    if model is None:
        print(f"Loading model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            # Fallback for Vercel's build environment where paths might be different
            # Vercel often puts everything from the root into the lambda environment.
            alt_model_path = 'model_artifacts/law_school_admission_model.joblib'
            if os.path.exists(alt_model_path):
                 print(f"Trying alternative model path: {alt_model_path}")
                 model = joblib.load(alt_model_path)
            else:
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH} or {alt_model_path}")
        else:
            model = joblib.load(MODEL_PATH)
        print("Model loaded.")

    if model_classes is None:
        print(f"Loading model classes from {MODEL_CLASSES_PATH}...")
        if not os.path.exists(MODEL_CLASSES_PATH):
            alt_classes_path = 'model_artifacts/model_classes.joblib'
            if os.path.exists(alt_classes_path):
                print(f"Trying alternative classes path: {alt_classes_path}")
                model_classes = joblib.load(alt_classes_path)
            else:
                raise FileNotFoundError(f"Classes file not found at {MODEL_CLASSES_PATH} or {alt_classes_path}")
        else:
            model_classes = joblib.load(MODEL_CLASSES_PATH)
        print("Model classes loaded.")
    
    if top_schools_list is None:
        print(f"Loading top schools from {TOP_SCHOOLS_PATH}...")
        if not os.path.exists(TOP_SCHOOLS_PATH):
            alt_schools_path = 'model_artifacts/top_schools.joblib'
            if os.path.exists(alt_schools_path):
                print(f"Trying alternative schools path: {alt_schools_path}")
                top_schools_list = joblib.load(alt_schools_path)
            else:
                # Fallback if file not found, use a default/empty list or raise error
                print(f"Warning: Schools file not found at {TOP_SCHOOLS_PATH} or {alt_schools_path}. Dropdown might be empty or use hardcoded values.")
                top_schools_list = [] # Or handle error appropriately
        else:
            top_schools_list = joblib.load(TOP_SCHOOLS_PATH)
        print("Top schools list loaded.")


load_dependencies() # Load on cold start

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data_bytes = self.rfile.read(content_length)
            data = json.loads(post_data_bytes.decode('utf-8'))

            print(f"Received data: {data}")

            # Ensure model and classes are loaded (might be needed if cold start failed partially)
            if model is None or model_classes is None or top_schools_list is None:
                load_dependencies()
                if model is None or model_classes is None: # top_schools_list can have a fallback
                    self._send_json_response({'error': 'Model or classes not loaded on server.'}, status_code=500)
                    return

            def to_bool(val):
                if isinstance(val, bool): return val
                return str(val).lower() in ['true', 'yes', '1', 'on']

            try:
                year = int(data.get('year'))
                month = int(data.get('month'))
                day_val = int(data.get('day')) # Renamed to avoid conflict
                dayofweek = pd.to_datetime(f"{year}-{month}-{day_val}").dayofweek
            except Exception as e:
                self._send_json_response({'error': f'Invalid date input: {e}'}, status_code=400)
                return

            gpa_val = data.get('gpa')
            years_out_val = data.get('years_out')
            international_gpa_val = data.get('international_gpa')

            indiv_df_data = {
                'year': [year],
                'month': [month],
                'day': [day_val], # Use the renamed variable
                'dayofweek': [dayofweek],
                'is_in_state': [to_bool(data.get('is_in_state', False))],
                'is_fee_waived': [to_bool(data.get('is_fee_waived', False))],
                'lsat': [int(data.get('lsat'))],
                'softs': [str(data.get('softs'))],
                'urm': [to_bool(data.get('urm', False))],
                'non_trad': [to_bool(data.get('non_trad', False))],
                'gpa': [float(gpa_val) if gpa_val and str(gpa_val).lower() not in ['nan', 'none', ''] else np.nan],
                'is_international': [to_bool(data.get('is_international', False))],
                'international_gpa': [str(international_gpa_val) if international_gpa_val and str(international_gpa_val).lower() not in ['nan', 'none', ''] else np.nan],
                'years_out': [int(years_out_val) if years_out_val and str(years_out_val).lower() not in ['nan', 'none', ''] else np.nan],
                'is_military': [to_bool(data.get('is_military', False))],
                'is_character_and_fitness_issues': [to_bool(data.get('is_character_and_fitness_issues', False))],
                'school_name': [str(data.get('school_name'))]
            }
            
            indiv_df = pd.DataFrame(indiv_df_data)
            print(f"DataFrame for prediction: \n{indiv_df.to_string()}")
            print(f"DataFrame dtypes: \n{indiv_df.dtypes}")

            probabilities = model.predict_proba(indiv_df)[0]
            
            output_dict = {}
            for i, class_label in enumerate(model_classes):
                output_dict[class_label] = probabilities[i]
            
            most_likely_outcome = max(output_dict, key=output_dict.get)

            response_payload = {
                'probabilities': output_dict,
                'most_likely_outcome': most_likely_outcome,
                'school_name': data.get('school_name'),
                'available_schools': top_schools_list # Send this for dynamic dropdown if needed, or on a GET request
            }
            self._send_json_response(response_payload)

        except json.JSONDecodeError:
            self._send_json_response({'error': 'Invalid JSON input'}, status_code=400)
        except KeyError as e:
            self._send_json_response({'error': f'Missing field: {e}'}, status_code=400)
        except ValueError as e:
            self._send_json_response({'error': f'Invalid data type for a field: {e}'}, status_code=400)
        except FileNotFoundError as e:
            print(f"File not found error during request: {e}")
            self._send_json_response({'error': f'Server configuration error: A required model file was not found. {e}'}, status_code=500)
        except Exception as e:
            import traceback
            print(f"Unhandled Exception: {e}\n{traceback.format_exc()}")
            self._send_json_response({'error': f'Internal server error: {str(e)}'}, status_code=500)
        return

    def do_GET(self): # Handler for GET requests, e.g., to fetch school list
        if self.path == '/api/predict' or self.path == '/api/predict/': # Or a more specific path like /api/schools
            if top_schools_list is None:
                load_dependencies() # Ensure it's loaded
            
            if top_schools_list is not None:
                 self._send_json_response({'available_schools': sorted(top_schools_list)})
            else:
                 self._send_json_response({'error': 'School list not available'}, status_code=500)
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Not Found'}).encode('utf-8'))
        return

    def do_OPTIONS(self): # Handle CORS preflight requests
        self.send_response(204) # No Content
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return

    def _send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*') # For development
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
