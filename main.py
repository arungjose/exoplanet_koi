from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import contextlib 

# --- 1. Define Constants and Artifact Paths ---

# Reverse mapping to convert model output (0, 1, 2) back to human-readable labels
REVERSE_LABEL_MAPPING = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}

# File paths for model artifacts
MODEL_PATH = 'model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'

# --- Global Variables (Loaded once at startup) ---
model = None
preprocessor = None


# --- 2. Define the Lifespan Context Manager (Modern Startup/Shutdown Handler) ---

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor
    
    # 🌟 STARTUP LOGIC (Runs BEFORE the server starts accepting requests)
    try:
        # Load the fitted model and preprocessor (imputer)
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✅ Model and Preprocessor loaded successfully via lifespan handler.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Artifact file not found: {e}. Ensure model.pkl and preprocessor.pkl are in the app directory.")
        # In a production environment, you might halt startup or raise a critical error.
    except Exception as e:
        print(f"FATAL ERROR loading artifacts: {e}")
    
    # The yield keyword tells FastAPI to start serving requests
    yield
    
    # 🛑 SHUTDOWN LOGIC (Runs when the server is shutting down - optional)
    print("Application shutdown complete.")

# --- 3. Initialize FastAPI with the Lifespan Handler ---

app = FastAPI(
    title="Exoplanet Classification API", 
    description="Predicts the disposition of a Kepler Object of Interest (KOI) using a Random Forest model trained on 17 features.",
    lifespan=lifespan
)

# --- 4. Define Input Schema (The 17 Essential Features) ---

class KeplerInput(BaseModel):
    """
    Defines the 17 features required by the trained Random Forest model.
    """
    # Flags (0 or 1) - Must be present
    koi_fpflag_nt: float = Field(..., description="Not Transit-Like Flag (0 or 1)")
    koi_fpflag_ss: float = Field(..., description="Stellar Eclipse Flag (0 or 1)")
    koi_fpflag_co: float = Field(..., description="Centroid Offset Flag (0 or 1)")
    koi_fpflag_ec: float = Field(..., description="Ephemeris Match Flag (0 or 1)")

    # Transit and Derived Properties (can be missing in real-world data, but Pydantic requires them here)
    koi_period: float = Field(..., description="Orbital Period [days]")
    koi_time0bk: float = Field(..., description="Transit Epoch [BKJD]")
    koi_impact: float = Field(..., description="Impact Parameter")
    koi_duration: float = Field(..., description="Transit Duration [hrs]")
    koi_depth: float = Field(..., description="Transit Depth [ppm]")
    koi_model_snr: float = Field(..., description="Signal-to-Noise Ratio")
    koi_tce_plnt_num: float = Field(..., description="TCE Planet Number (e.g., 1, 2)")
    koi_prad: float = Field(..., description="Planetary Radius [Earth radii]")
    koi_teq: float = Field(..., description="Equilibrium Temperature [K]")
    koi_insol: float = Field(..., description="Insolation Flux [Earth flux]")
    
    # Stellar Properties
    koi_steff: float = Field(..., description="Stellar Effective Temperature [K]")
    koi_slogg: float = Field(..., description="Stellar Surface Gravity [log10(cm/s**2)]")
    koi_srad: float = Field(..., description="Stellar Radius [Solar radii]")

# --- 5. Prediction Endpoint ---

@app.post("/predict", tags=["Prediction"])
async def predict_exoplanet(input_data: KeplerInput):
    """
    Accepts JSON input containing the 17 Kepler features and returns the predicted disposition.
    """
    global model, preprocessor
    
    # Check if artifacts were loaded (they should be, due to the startup handler)
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model service unavailable. Artifacts failed to load during startup."
        )

    # Convert Pydantic object to a dictionary, then to a Pandas DataFrame (single row)
    # This maintains column names and order, which is essential for the ColumnTransformer
    input_dict = input_data.model_dump()
    data_df = pd.DataFrame([input_dict])
    
    try:
        # 1. Preprocessing (Imputation)
        # The preprocessor cleans the data and aligns the columns
        data_transformed = preprocessor.transform(data_df)
        
        # 2. Prediction
        prediction_encoded = model.predict(data_transformed)[0]
        
        # 3. Output Formatting
        prediction_label = REVERSE_LABEL_MAPPING.get(int(prediction_encoded), 'UNKNOWN')
        
        return {
            "status": "success",
            "prediction_code": int(prediction_encoded),
            "prediction_label": prediction_label,
            "message": "Prediction generated successfully."
        }

    except Exception as e:
        # Catch any errors during transformation or prediction (e.g., bad input types)
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed due to internal error: {e}"
        )
