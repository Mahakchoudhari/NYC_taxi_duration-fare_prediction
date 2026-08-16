from pathlib import Path
import logging
import pickle

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

@app.get("/analytics")
def get_analytics():

    try:

        logger.info(
            "Analytics request received"
        )


        # ----------------------------------------------------
        # DATASET PATH
        # ----------------------------------------------------

        dataset_path = BASE_DIR / "NYC.csv"


        if not dataset_path.exists():

            raise HTTPException(
                status_code=404,
                detail="Analytics dataset not found."
            )


        # ----------------------------------------------------
        # LOAD DATASET
        # ----------------------------------------------------

        df = pd.read_csv(
            dataset_path
        )


        # ----------------------------------------------------
        # DATETIME
        # ----------------------------------------------------

        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"],
            errors="coerce"
        )


        df = df.dropna(
            subset=["pickup_datetime"]
        )


        # ----------------------------------------------------
        # PICKUP HOUR
        # ----------------------------------------------------

        df["pickup_hour"] = (
            df["pickup_datetime"].dt.hour
        )


        # ----------------------------------------------------
        # TOTAL TRIPS
        # ----------------------------------------------------

        total_trips = int(
            len(df)
        )


        # ----------------------------------------------------
        # AVERAGE PASSENGERS
        # ----------------------------------------------------

        avg_passengers = round(
            float(
                df["passenger_count"].mean()
            ),
            2
        )


        # ----------------------------------------------------
        # AVERAGE DURATION
        # ----------------------------------------------------

        avg_duration_minutes = None

        if "trip_duration" in df.columns:

            avg_duration_minutes = round(
                float(
                    df["trip_duration"].mean()
                ) / 60,
                2
            )


        # ----------------------------------------------------
        # TRIPS BY HOUR
        # ----------------------------------------------------

        trips_by_hour_df = (
            df
            .groupby("pickup_hour")
            .size()
            .reset_index(name="trips")
        )


        trips_by_hour = [

            {
                "hour": int(row["pickup_hour"]),
                "trips": int(row["trips"]),
            }

            for _, row
            in trips_by_hour_df.iterrows()
        ]


        # ----------------------------------------------------
        # DURATION BY HOUR
        # ----------------------------------------------------

        duration_by_hour = []

        if "trip_duration" in df.columns:

            duration_df = (
                df
                .groupby("pickup_hour")[
                    "trip_duration"
                ]
                .mean()
                .reset_index()
            )


            duration_by_hour = [

                {
                    "hour": int(row["pickup_hour"]),

                    "duration_minutes": round(
                        float(
                            row["trip_duration"]
                        ) / 60,
                        2
                    ),
                }

                for _, row
                in duration_df.iterrows()
            ]


        # ----------------------------------------------------
        # RUSH HOUR
        # ----------------------------------------------------

        rush_hours = [
            7,
            8,
            9,
            10,
            16,
            17,
            18,
            19,
            20,
        ]


        df["is_rush_hour"] = (
            df["pickup_hour"]
            .isin(rush_hours)
        )


        rush_trips = int(
            df["is_rush_hour"].sum()
        )


        normal_trips = int(
            (~df["is_rush_hour"]).sum()
        )


        # ----------------------------------------------------
        # RUSH / NORMAL DURATION
        # ----------------------------------------------------

        rush_avg_duration = None
        normal_avg_duration = None


        if "trip_duration" in df.columns:

            rush_data = df[
                df["is_rush_hour"]
            ]

            normal_data = df[
                ~df["is_rush_hour"]
            ]


            if len(rush_data) > 0:

                rush_avg_duration = round(
                    float(
                        rush_data[
                            "trip_duration"
                        ].mean()
                    ) / 60,
                    2
                )


            if len(normal_data) > 0:

                normal_avg_duration = round(
                    float(
                        normal_data[
                            "trip_duration"
                        ].mean()
                    ) / 60,
                    2
                )


        # ----------------------------------------------------
        # WEEKDAY / WEEKEND
        # ----------------------------------------------------

        df["day_of_week"] = (
            df["pickup_datetime"]
            .dt.dayofweek
        )


        df["is_weekend"] = (
            df["day_of_week"] >= 5
        )


        weekday_trips = int(
            (~df["is_weekend"]).sum()
        )


        weekend_trips = int(
            df["is_weekend"].sum()
        )


        # ----------------------------------------------------
        # VENDOR DISTRIBUTION
        # ----------------------------------------------------

        vendor_distribution = []


        if "vendor_id" in df.columns:

            vendor_df = (
                df
                .groupby("vendor_id")
                .size()
                .reset_index(name="trips")
            )


            vendor_distribution = [

                {
                    "vendor_id": int(
                        row["vendor_id"]
                    ),

                    "trips": int(
                        row["trips"]
                    ),
                }

                for _, row
                in vendor_df.iterrows()
            ]


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        result = {

            "success": True,

            "total_trips": total_trips,

            "avg_passengers": avg_passengers,

            "avg_duration_minutes":
                avg_duration_minutes,

            "trips_by_hour":
                trips_by_hour,

            "duration_by_hour":
                duration_by_hour,

            "rush_hour": {

                "rush_trips":
                    rush_trips,

                "normal_trips":
                    normal_trips,

                "rush_avg_duration_minutes":
                    rush_avg_duration,

                "normal_avg_duration_minutes":
                    normal_avg_duration,
            },

            "day_type": {

                "weekday_trips":
                    weekday_trips,

                "weekend_trips":
                    weekend_trips,
            },

            "vendor_distribution":
                vendor_distribution,
        }


        logger.info(
            "Analytics calculated successfully"
        )


        return result


    except HTTPException:
        raise


    except Exception as e:

        logger.exception(
            "Analytics failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to calculate analytics."
            ),
        ) from e