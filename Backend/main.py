from pathlib import Path
import logging
import pickle
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from Backend.config import settings
from Backend.schemas import (
    TaxiInput,
    PredictionResponse,
    HealthResponse,
)
from Backend.preprocess import (
    PreprocessTransformer,
    OutlierHandling,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / settings.MODEL_PATH
PIPE_PATH = BASE_DIR / settings.PIPELINE_PATH


# ============================================================
# LOAD MODEL
# ============================================================

best_model = None
pipe = None


def load_models():
    """
    Load trained ML model and preprocessing pipeline.
    """

    global best_model, pipe

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    try:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        with open(MODEL_PATH, "rb") as f:
            best_model = pickle.load(f)

        logger.info(
            "best_model.pkl loaded successfully"
        )

    except Exception as e:

        best_model = None

        logger.exception(
            "Failed to load best_model.pkl"
        )


    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    try:

        if not PIPE_PATH.exists():
            raise FileNotFoundError(
                f"Pipeline file not found: {PIPE_PATH}"
            )

        with open(PIPE_PATH, "rb") as f:
            pipe = pickle.load(f)

        logger.info(
            "pipe.pkl loaded successfully"
        )

    except Exception as e:

        pipe = None

        logger.exception(
            "Failed to load pipe.pkl"
        )


# Load during application startup
load_models()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Production-ready API for NYC Taxi "
        "Trip Duration and Fare Prediction."
    ),
    version=settings.APP_VERSION,
)


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    origin.strip()
    for origin in settings.ALLOWED_ORIGINS.split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    logger.exception(
        "Unhandled exception: %s",
        exc
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": (
                "Something went wrong while "
                "processing the request."
            ),
        },
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "NYC Taxi Prediction API is running",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
        "analytics_endpoint": "/analytics",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    model_loaded = best_model is not None
    pipeline_loaded = pipe is not None

    status = (
        "healthy"
        if model_loaded and pipeline_loaded
        else "unhealthy"
    )

    return {
        "status": status,
        "model_loaded": model_loaded,
        "pipeline_loaded": pipeline_loaded,
    }


# ============================================================
# CREATE DATAFRAME
# ============================================================

def create_dataframe(
    data: TaxiInput
) -> pd.DataFrame:

    return pd.DataFrame(
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


# ============================================================
# FARE ESTIMATION
# ============================================================

def fare_estimate(
    pipeline,
    dataset,
    pred_duration
):

    if pipeline is None:

        raise RuntimeError(
            "Preprocessing pipeline is not loaded."
        )


    # --------------------------------------------------------
    # GET PREPROCESSING STEP
    # --------------------------------------------------------

    try:

        preprocess_step = (
            pipeline.named_steps["preprocess"]
        )

    except Exception as e:

        raise RuntimeError(
            "The loaded pipe.pkl does not contain "
            "'preprocess' step."
        ) from e


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    processed = preprocess_step.transform(
        dataset.copy()
    )


    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    distance = float(
        processed["haversine_km"].iloc[0]
    )


    # --------------------------------------------------------
    # DURATION
    # --------------------------------------------------------

    time_hours = float(
        pred_duration / 3600
    )


    # --------------------------------------------------------
    # RUSH HOUR
    # --------------------------------------------------------

    rush = int(
        processed["is_rush_hour"].iloc[0]
    )


    # --------------------------------------------------------
    # PASSENGERS
    # --------------------------------------------------------

    passenger_count = int(
        processed["passenger_count"].iloc[0]
    )


    # --------------------------------------------------------
    # HOUR
    # --------------------------------------------------------

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
    fare = max(fare, 3.0)


    return (
        round(fare, 2),
        round(distance, 2),
    )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: TaxiInput):

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if best_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction model is currently unavailable."
            ),
        )


    if pipe is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Preprocessing pipeline is currently unavailable."
            ),
        )


    try:

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        new_data = create_dataframe(data)

        logger.info(
            "Prediction request received"
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        pred_duration = best_model.predict(
            new_data
        )


        value = float(
            pred_duration[0]
        )


        # ----------------------------------------------------
        # VALIDATE MODEL OUTPUT
        # ----------------------------------------------------

        if value <= 0:

            raise ValueError(
                "Model returned an invalid duration."
            )


        # ----------------------------------------------------
        # FARE + DISTANCE
        # ----------------------------------------------------

        fare, distance = fare_estimate(
            pipe,
            new_data,
            value
        )


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        speed = (
            distance / (value / 3600)
            if value > 0
            else 0
        )

        speed = round(speed, 2)


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        result = {
            "success": True,

            "duration_seconds": round(
                value,
                2
            ),

            "duration_minutes": round(
                value / 60,
                2
            ),

            "estimated_fare": fare,

            "distance_km": distance,

            "estimated_speed": speed,
        }


        logger.info(
            "Prediction completed successfully"
        )


        return result


    except HTTPException:
        raise


    except Exception as e:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. "
                "Please check the input and server logs."
            ),
        ) from e


# ============================================================
# ANALYTICS ENDPOINT
# ============================================================

# ============================================================
# ANALYTICS ENDPOINT
# ============================================================

@app.get("/analytics")
def get_analytics():

    try:
        logger.info("Analytics request received")

        # ----------------------------------------------------
        # DATASET PATH
        # ----------------------------------------------------

        analytics_path = BASE_DIR / "analytics.json"

        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if not analytics_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Analytics data file not found."
            )

        # ----------------------------------------------------
        # LOAD ANALYTICS DATA
        # ----------------------------------------------------

        with open(
            analytics_path,
            "r",
            encoding="utf-8"
        ) as f:
            analytics_data = json.load(f)

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "success": True,
            **analytics_data
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "Analytics failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load analytics."
        ) from e