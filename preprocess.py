import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

# --- 1. Define Constants (The 17 Essential Features) ---

TARGET_COLUMN = 'koi_disposition'
# 17 Feature Columns: Flags (4), Transit Metrics (7), Planetary (3), Stellar (3)
FEATURE_COLUMNS = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 
    'koi_depth', 'koi_model_snr', 'koi_tce_plnt_num', 
    'koi_prad', 'koi_teq', 'koi_insol', 
    'koi_steff', 'koi_slogg', 'koi_srad' 
]

# Identify the subset of features that contain NaNs and require imputation
NUMERICAL_IMPUTE_FEATURES = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 
    'koi_depth', 'koi_model_snr', 
    'koi_prad', 'koi_teq', 'koi_insol', 
    'koi_steff', 'koi_slogg', 'koi_srad'
]


# --- 2. Data Loading and Filtering ---
dataset_path = "KeplerObjectofInterest.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset {dataset_path} not found.")

print("Step 1: Loading and filtering data...")
df = pd.read_csv(dataset_path, comment='#')

# Select only the target and the full 17 feature columns
df_clean = df[[TARGET_COLUMN] + FEATURE_COLUMNS].copy()
df_clean = df_clean.dropna(subset=[TARGET_COLUMN]) # Drop rows where target is unknown

# Separate features (X) and target (y)
X = df_clean[FEATURE_COLUMNS]
y = df_clean[TARGET_COLUMN]


# --- 3. Target Encoding ---
# Map the string labels to integers (required by the model)
label_mapping = {'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2}
y_encoded = y.map(label_mapping)
y_encoded.dropna(inplace=True) 
X = X.loc[y_encoded.index] # Keep X aligned with y after dropping NaNs


# --- 4. Define, Fit, and Save the Preprocessing Pipeline ---

# The pipeline for numerical columns: only imputation (median)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# The ColumnTransformer applies the imputer ONLY to the specified numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, NUMERICAL_IMPUTE_FEATURES)
    ],
    # 'passthrough' keeps all other columns (the four flags and koi_tce_plnt_num) as they are
    remainder='passthrough',
    # Ensure column names are retained after transformation
    verbose_feature_names_out=False
)
preprocessor.set_output(transform="pandas")


# Fit the preprocessor on the full feature set (X)
print("Step 2: Fitting and saving the preprocessing pipeline...")
preprocessor.fit(X)

# Save the fitted preprocessor object (CRITICAL artifact for deployment)
joblib.dump(preprocessor, 'preprocessor.pkl')
print("✅ Saved preprocessor.pkl")


# --- 5. Apply Transform and Split Data ---

# Apply the fitted transformation to X
X_processed = preprocessor.transform(X)
# The column order and names are now finalized (17 total columns)

# Perform the final split: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, 
    y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded 
)

# Save the split data and the target mapping for the next script (train.py and deployment)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(label_mapping, 'target_mapping.pkl')


print("Step 3: Data splitting complete and saved.")
print(f"X_train shape: {X_train.shape}")
print(f"Number of columns (Features): {X_train.shape[1]}")