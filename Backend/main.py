from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin


# -----------------------------
# PREPROCESSING CLASSES
# -----------------------------

class PreprocessTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, d):
        d = d.copy()

        d['pickup_datetime'] = pd.to_datetime(d['pickup_datetime'])

        d['pickup_hour'] = d['pickup_datetime'].dt.hour
        d['day_of_week'] = d['pickup_datetime'].dt.dayofweek
        d['month'] = d['pickup_datetime'].dt.month
        d['day_of_month'] = d['pickup_datetime'].dt.day

        d['is_weekend'] = d['day_of_week'].apply(
            lambda x: 1 if x > 5 else 0
        )

        d['is_rush_hour'] = d['pickup_hour'].isin(
            [7, 8, 9, 10, 16, 17, 18, 19, 20]
        ).astype(int)

        d['is_night'] = d['pickup_hour'].isin(
            [22, 23, 0, 1, 2, 3, 4]
        ).astype(int)

        d['hour_sin'] = np.sin(
            2 * np.pi * d['pickup_hour'] / 24
        )

        d['hour_cos'] = np.cos(
            2 * np.pi * d['pickup_hour'] / 24
        )

        d['dow_sin'] = np.sin(
            2 * np.pi * d['day_of_week'] / 7
        )

        d['dow_cos'] = np.cos(
            2 * np.pi * d['day_of_week'] / 7
        )

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371

            lat1, lon1, lat2, lon2 = map(
                np.radians,
                [lat1, lon1, lat2, lon2]
            )

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1)
                * np.cos(lat2)
                * np.sin(dlon / 2) ** 2
            )

            return R * 2 * np.arcsin(np.sqrt(a))

        def manhattan(lat1, lon1, lat2, lon2):
            return (
                haversine(lat1, lon1, lat2, lon1)
                + haversine(lat2, lon1, lat2, lon2)
            )

        d['haversine_km'] = haversine(
            d['pickup_latitude'],
            d['pickup_longitude'],
            d['dropoff_latitude'],
            d['dropoff_longitude']
        )

        d['manhattan_km'] = manhattan(
            d['pickup_latitude'],
            d['pickup_longitude'],
            d['dropoff_latitude'],
            d['dropoff_longitude']
        )

        d['route_ratio'] = (
            d['manhattan_km']
            / (d['haversine_km'] + 1e-5)
        )

        d['store_and_fwd_flag'] = (
            d['store_and_fwd_flag']
            .astype(str)
            .str.strip()
            .str.lower()
            .map({'y': 1, 'n': 0})
            .fillna(0)
        )

        d.drop(
            [
                'pickup_latitude',
                'pickup_longitude',
                'dropoff_latitude',
                'dropoff_longitude',
                'pickup_datetime'
            ],
            axis=1,
            inplace=True
        )

        if 'dropoff_datetime' in d.columns:
            d.drop('dropoff_datetime', axis=1, inplace=True)

        return d


class OutlierHandling(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        columns,
        lower_quantile=0.01,
        upper_quantile=0.99
    ):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds = {}

    def fit(self, X, y=None):
        X = X.copy()

        for col in self.columns:
            lower = X[col].quantile(self.lower_quantile)
            upper = X[col].quantile(self.upper_quantile)

            self.bounds[col] = (lower, upper)

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.columns:
            lower, upper = self.bounds[col]

            X[col] = X[col].clip(
                lower,
                upper
            )

        return X


# -----------------------------
# LOAD MODEL
# -----------------------------

with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("pipe.pkl", "rb") as f:
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