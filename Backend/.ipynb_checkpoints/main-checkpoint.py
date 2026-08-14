from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

from Backend.preprocess import PreprocessTransformer, OutlierHandling

# -----------------------------
# LOAD MODEL
# -----------------------------

with open("Backend/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("Backend/pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

# -----------------------------
# FASTAPI
# -----------------------------

app = FastAPI(
    title="NYC Taxi Prediction API",
    description="API for NYC Taxi Trip Duration and Fare Prediction",
    version="1.0"
)


# -----------------------------
# INPUT SCHEMA
# -----------------------------

class TaxiInput(BaseModel):

    vendor_id: int
    pickup_datetime: str
    passenger_count: int

    pickup_longitude: float
    pickup_latitude: float

    dropoff_longitude: float
    dropoff_latitude: float

    store_and_fwd_flag: str


# -----------------------------
# FARE ESTIMATION
# -----------------------------

def fare_estimate(
    pipeline,
    dataset,
    pred_duration
):

    base_fare = 2.5

    processed = (
        pipeline
        .named_steps['preprocess']
        .transform(dataset)
    )

    distance = processed['haversine_km'].iloc[0]

    time = pred_duration / 60

    rush = processed['is_rush_hour'].iloc[0]

    passenger_count = (
        processed['passenger_count'].iloc[0]
    )

    hour = processed['pickup_hour'].iloc[0]

    fare = (
        base_fare
        + (distance * 1.5)
        + (time * 0.5)
    )

    if rush == 1:
        fare *= 1.2

    if hour >= 20 or hour <= 6:
        fare += 1.0

    fare += passenger_count * 0.5

    return (
        round(max(fare, 3.0), 2),
        round(distance, 2)
    )


# -----------------------------
# PREDICTION API
# -----------------------------

@app.post("/predict")
def predict(data: TaxiInput):

    new_data = pd.DataFrame([{
        'vendor_id': data.vendor_id,
        'pickup_datetime': data.pickup_datetime,
        'passenger_count': data.passenger_count,
        'pickup_longitude': data.pickup_longitude,
        'pickup_latitude': data.pickup_latitude,
        'dropoff_longitude': data.dropoff_longitude,
        'dropoff_latitude': data.dropoff_latitude,
        'store_and_fwd_flag': data.store_and_fwd_flag
    }])

    try:

        pred_duration = best_model.predict(
            new_data
        )

        value = pred_duration.item()

        fare, distance = fare_estimate(
            pipe,
            new_data,
            value
        )

        speed = round(
            distance / (value / 3600),
            2
        )

        return {
            "success": True,
            "duration_seconds": round(value, 2),
            "duration_minutes": round(value / 60, 2),
            "estimated_fare": fare,
            "distance_km": distance,
            "estimated_speed": speed
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }