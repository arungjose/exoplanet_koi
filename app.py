import json
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd   # ✅ Added pandas

app = Flask(__name__)

# --- MODEL LOADING LOGIC ---
try:
    model = joblib.load("model2.pkl")   # Load joblib model
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model2.pkl: {e}. Prediction functionality will be disabled.")
    model = None
# ---------------------------

# The 17 required keys (MUST match training columns)
REQUIRED_KEYS = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth', 
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_tce_plnt_num', 
    'koi_steff', 'koi_slogg', 'koi_srad'
]

@app.route('/')
def index():
    """Renders the single-page application (index.html)."""
    return render_template('index.html')

@app.route('/api/submit_data', methods=['POST'])
def submit_data():
    """Receives JSON data from the frontend and processes it."""
    try:
        if model is None:
            return jsonify({"error": "Model failed to load on the server."}), 500

        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        # Validate that all 17 keys are present
        if not all(key in data for key in REQUIRED_KEYS):
            missing_keys = [key for key in REQUIRED_KEYS if key not in data]
            return jsonify({"error": f"Missing required data fields: {', '.join(missing_keys)}"}), 400

        # ✅ Create a DataFrame with correct columns
        input_df = pd.DataFrame([[float(data[key]) for key in REQUIRED_KEYS]], columns=REQUIRED_KEYS)
        input_df = input_df[model.feature_names_in_]  # <--- ensure exact order
        prediction = model.predict(input_df)

        # Format output
        if prediction[0] == 2:
            result_label = "Confirmed Exoplanet"
        elif prediction[0] == 1:
            result_label = "Candidate (KOI)"
        else:
            result_label = "False Positive"

        return jsonify({
            "message": "Data processed successfully. Prediction complete.",
            "status": "success",
            "prediction": result_label,
            "raw_prediction": int(prediction[0])
        }), 200

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": f"Internal server error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
