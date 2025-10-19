# test_model.py
import joblib
import pandas as pd

def main():
    # 1. Load the trained model
    try:
        model = joblib.load("model2.pkl")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print("❌ Error loading model:", e)
        return

    # 2. Determine the correct column order
    try:
        X_train = pd.read_csv("X_train.csv")
        column_order = list(X_train.columns)
        print("Columns used in training:", column_order)
    except FileNotFoundError:
        print("⚠️ X_train.csv not found. Using hardcoded column order.")
        column_order = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
                        'koi_model_snr', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff',
                        'koi_slogg', 'koi_srad', 'koi_fpflag_nt', 'koi_fpflag_ss', 
                        'koi_fpflag_co', 'koi_fpflag_ec', 'koi_tce_plnt_num']

    # 3. Create a test input (replace values with realistic ones)
    test_input = pd.DataFrame([{
        "koi_fpflag_nt": 0,
        "koi_fpflag_ss": 1,
        "koi_fpflag_co": 0,
        "koi_fpflag_ec": 0,
        "koi_period": 2.204735417,
        "koi_time0bk": 121.3585417,
        "koi_impact": 0.224,
        "koi_duration": 3.88864,
        "koi_depth": 6.67E+03,
        "koi_prad": 16.1,
        "koi_teq": 2048,
        "koi_insol": 4148.92,
        "koi_model_snr": 5945.9,
        "koi_tce_plnt_num": 1,
        "koi_steff": 6440,
        "koi_slogg": 4.019,
        "koi_srad": 1.952
    }])

    # 4. Reorder test input columns to match training
    test_input = test_input[column_order]
    print("Test input shape:", test_input.shape)

    # 5. Make a prediction
    try:
        prediction = model.predict(test_input)
        print("Raw prediction:", prediction)
        
        # 6. Map numeric label to human-readable
        label_mapping = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
        print("Predicted label:", label_mapping.get(prediction[0], "Unknown"))
    except Exception as e:
        print("❌ Error during prediction:", e)

if __name__ == "__main__":
    main()
