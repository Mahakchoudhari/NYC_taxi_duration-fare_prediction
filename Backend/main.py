from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

from Backend.preprocess import PreprocessTransformer, OutlierHandling


# =====================================================
# LOAD MODEL
# =====================================================

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "Backend/best_model.pkl"
PIPE_PATH = "Backend/pipe.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        best_model = pickle.load(f)
    print("✅ best_model.pkl loaded successfully")
except Exception as e:
    print("❌ Error loading best_model.pkl:", e)
    best_model = None

try:
    with open(PIPE_PATH, "rb") as f:
        pipe = pickle.load(f)
    print("✅ pipe.pkl loaded successfully")
except Exception as e:
    print("❌ Error loading pipe.pkl:", e)
    pipe = None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="NYC Taxi Prediction API",
    description="API for NYC Taxi Trip Duration and Fare Prediction",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

# app.add_middleware(
#     CORSMiddleware,

#     allow_origins=[
#         "http://localhost:5174",
#         "http://127.0.0.1:5174",

#         "http://localhost:5173",
#         "http://127.0.0.1:5173",
#     ],

#     allow_credentials=True,

#     allow_methods=["*"],

#     allow_headers=["*"],
# )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class TaxiInput(BaseModel):

    vendor_id: int

    pickup_datetime: str

    passenger_count: int

    pickup_longitude: float
    pickup_latitude: float

    dropoff_longitude: float
    dropoff_latitude: float

    store_and_fwd_flag: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "NYC Taxi Prediction API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "model_loaded": best_model is not None,

        "pipeline_loaded": pipe is not None,
    }


# ============================================================
# CREATE DATAFRAME
# ============================================================

def create_dataframe(data: TaxiInput):

    new_data = pd.DataFrame(
        [
            {
                "vendor_id": data.vendor_id,

                "pickup_datetime": data.pickup_datetime,

                "passenger_count": data.passenger_count,

                "pickup_longitude": data.pickup_longitude,

                "pickup_latitude": data.pickup_latitude,

                "dropoff_longitude": data.dropoff_longitude,

                "dropoff_latitude": data.dropoff_latitude,

                "store_and_fwd_flag": data.store_and_fwd_flag,
            }
        ]
    )

    return new_data


# ============================================================
# FARE ESTIMATION
# ============================================================

def fare_estimate(
    pipeline,
    dataset,
    pred_duration,
):

    if pipeline is None:

        raise RuntimeError(
            "Preprocessing pipeline is not loaded."
        )

    # --------------------------------------------
    # PREPROCESS DATA
    # --------------------------------------------

    try:

        preprocess_step = (
            pipeline
            .named_steps["preprocess"]
        )

    except Exception:

        raise RuntimeError(
            "The loaded pipe.pkl does not contain "
            "'preprocess' step."
        )

    processed = preprocess_step.transform(
        dataset.copy()
    )

    # --------------------------------------------
    # DISTANCE
    # --------------------------------------------

    distance = float(
        processed["haversine_km"].iloc[0]
    )

    # --------------------------------------------
    # DURATION
    # --------------------------------------------

    time_hours = float(
        pred_duration / 3600
    )

    # --------------------------------------------
    # RUSH HOUR
    # --------------------------------------------

    rush = int(
        processed["is_rush_hour"].iloc[0]
    )

    # --------------------------------------------
    # PASSENGERS
    # --------------------------------------------

    passenger_count = int(
        processed["passenger_count"].iloc[0]
    )

    # --------------------------------------------
    # HOUR
    # --------------------------------------------

    hour = int(
        processed["pickup_hour"].iloc[0]
    )

    # ========================================================
    # FARE FORMULA
    # ========================================================

    base_fare = 2.5

    fare = (
        base_fare
        + (distance * 1.5)
        + (time_hours * 0.5)
    )

    # Rush hour surcharge

    if rush == 1:

        fare *= 1.2

    # Night surcharge

    if hour >= 20 or hour <= 6:

        fare += 1.0

    # Passenger adjustment

    fare += passenger_count * 0.5

    # Minimum fare

    fare = max(
        fare,
        3.0
    )

    return (
        round(fare, 2),

        round(distance, 2),
    )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: TaxiInput):

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if best_model is None:

        raise HTTPException(
            status_code=500,

            detail=(
                "best_model.pkl could not be loaded."
            ),
        )

    if pipe is None:

        raise HTTPException(
            status_code=500,

            detail=(
                "pipe.pkl could not be loaded."
            ),
        )

    try:

        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        new_data = create_dataframe(data)

        print("\n==============================")
        print("🚕 NEW PREDICTION REQUEST")
        print("==============================")

        print(new_data)

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        pred_duration = best_model.predict(
            new_data
        )

        # Convert numpy value to Python float

        value = float(
            pred_duration[0]
        )

        print(
            f"Predicted duration: {value:.2f} seconds"
        )

        # ====================================================
        # VALIDATE PREDICTION
        # ====================================================

        if value <= 0:

            raise ValueError(
                "Model returned an invalid duration."
            )

        # ====================================================
        # FARE + DISTANCE
        # ====================================================

        fare, distance = fare_estimate(
            pipe,
            new_data,
            value,
        )

        # ====================================================
        # AVERAGE SPEED
        # ====================================================

        if value > 0:

            speed = (
                distance
                / (value / 3600)
            )

        else:

            speed = 0

        speed = round(
            speed,
            2
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        result = {

            "success": True,

            "duration_seconds": round(
                value,
                2,
            ),

            "duration_minutes": round(
                value / 60,
                2,
            ),

            "estimated_fare": fare,

            "distance_km": distance,

            "estimated_speed": speed,
        }

        print("\nPrediction result:")

        print(result)

        print("==============================\n")

        return result

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print("\n❌ PREDICTION ERROR")

        print(str(e))

        print("==============================\n")

        raise HTTPException(
            status_code=500,

            detail=str(e),
        )