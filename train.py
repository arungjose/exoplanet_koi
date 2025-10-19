import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load Preprocessed Data ---
try:
    X_train = pd.read_csv('X_train.csv')
    y_train = joblib.load('y_train.pkl')
    X_test = pd.read_csv('X_test.csv')
    y_test = joblib.load('y_test.pkl')
    label_mapping = joblib.load('target_mapping.pkl')  # e.g., {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
except FileNotFoundError:
    print("Error: Missing data files. Did you run 'preprocess.py' first?")
    exit()

# --- 1b. Map CANDIDATE (1) → FALSE POSITIVE (0) for binary classification ---
y_train_binary = y_train.replace({1: 0})
y_test_binary = y_test.replace({1: 0})

# Binary label mapping for reference
binary_label_mapping = {0: "FALSE POSITIVE", 2: "CONFIRMED"}
binary_labels = sorted(binary_label_mapping.keys())  # [0, 2]
binary_target_names = [binary_label_mapping[k] for k in binary_labels]

print(f"Training samples: {len(y_train_binary)} (binary)")
print(f"Testing samples: {len(y_test_binary)} (binary)")

# --- 2. Train Model (Random Forest) ---
print("Step 2: Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train_binary)
print("Training complete.")

# --- 3. Evaluate Model ---
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test_binary, y_pred)
print(f"\n🌟 Accuracy on Test Set: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test_binary,
    y_pred,
    labels=binary_labels,
    target_names=binary_target_names,
    zero_division=0
))

# --- 4. Save the trained model ---
joblib.dump(rf_model, 'model2.pkl')
print("✅ Saved model2.pkl (binary classifier, ready for deployment)")
