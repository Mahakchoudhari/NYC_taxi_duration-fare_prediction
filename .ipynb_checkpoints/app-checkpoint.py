import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# ── PEHLE CLASSES DEFINE KARO ──

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
        d['is_weekend'] = d['day_of_week'].apply(lambda x: 1 if x > 5 else 0)
        d['is_rush_hour'] = d['pickup_hour'].isin([7,8,9,10,16,17,18,19,20]).astype(int)
        d['is_night'] = d['pickup_hour'].isin([22,23,0,1,2,3,4]).astype(int)
        d['hour_sin'] = np.sin(2 * np.pi * d['pickup_hour'] / 24)
        d['hour_cos'] = np.cos(2 * np.pi * d['pickup_hour'] / 24)
        d['dow_sin'] = np.sin(2 * np.pi * d['day_of_week'] / 7)
        d['dow_cos'] = np.cos(2 * np.pi * d['day_of_week'] / 7)

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            return R * 2 * np.arcsin(np.sqrt(a))

        def manhattan(lat1, lon1, lat2, lon2):
            return haversine(lat1, lon1, lat2, lon1) + haversine(lat2, lon1, lat2, lon2)

        d['haversine_km'] = haversine(
            d['pickup_latitude'], d['pickup_longitude'],
            d['dropoff_latitude'], d['dropoff_longitude'])
        d['manhattan_km'] = manhattan(
            d['pickup_latitude'], d['pickup_longitude'],
            d['dropoff_latitude'], d['dropoff_longitude'])
        d['route_ratio'] = d['manhattan_km'] / (d['haversine_km'] + 1e-5)
        d['store_and_fwd_flag'] = (
            d['store_and_fwd_flag'].astype(str).str.strip()
            .str.lower().map({'y': 1, 'n': 0}).fillna(0))
        d.drop(['pickup_latitude', 'pickup_longitude',
                'dropoff_latitude', 'dropoff_longitude',
                'pickup_datetime'], axis=1, inplace=True)
        if 'dropoff_datetime' in d.columns:
            d.drop('dropoff_datetime', axis=1, inplace=True)
        return d


class OutlierHandling(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lower_quantile=0.01, upper_quantile=0.99):
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
            X[col] = X[col].clip(lower, upper)
        return X


# ── BAAD MEIN LOAD KARO ──
best_model = pickle.load(open('best_model.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))


# ── FARE ESTIMATE ──
def fare_estimate(pipeline, dataset, pred_duration):
    base_fare = 2.5
    processed = pipeline.named_steps['preprocess'].transform(dataset)
    distance = processed['haversine_km'].iloc[0]
    time = pred_duration / 60
    rush = processed['is_rush_hour'].iloc[0]
    passenger_count = processed['passenger_count'].iloc[0]
    hour = processed['pickup_hour'].iloc[0]
    fare = base_fare + (distance * 1.5) + (time * 0.5)
    if rush == 1:
        fare *= 1.2
    if hour >= 20 or hour <= 6:
        fare += 1.0
    fare += passenger_count * 0.5
    return round(max(fare, 3.0), 2), round(distance, 2)


# ── UI ──
st.title("🚕 NYC Taxi Trip Duration & Fare Estimator")
st.write("Enter trip details to predict duration and estimated fare!")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Pickup Details")
    pickup_datetime = st.text_input("Pickup DateTime", "2016-06-12 08:30:00",
                                    help="Format: YYYY-MM-DD HH:MM:SS")
    pickup_latitude = st.number_input(
    "Pickup Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=40.7489,
    step=0.0001
    )
    
    pickup_longitude = st.number_input(
        "Pickup Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-73.9680,
        step=0.0001
    )

with col2:
    st.subheader("🏁 Dropoff Details")
    dropoff_latitude = st.number_input(
    "Dropoff Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=40.7614,
    step=0.0001
    )
    
    dropoff_longitude = st.number_input(
        "Dropoff Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-73.9776,
        step=0.0001
    )

st.subheader("🚗 Trip Details")
col3, col4, col5 = st.columns(3)
with col3:
    vendor_id = st.selectbox("Vendor ID", [1, 2])
with col4:
    passenger_count = st.slider("Passengers", 1, 6, 1)
with col5:
    store_and_fwd_flag = st.selectbox("Store & Forward Flag", ['N', 'Y'])

if st.button("🔮 Predict"):
    new_data = pd.DataFrame([{
        'vendor_id': vendor_id,
        'pickup_datetime': pickup_datetime,
        'passenger_count': passenger_count,
        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,
        'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,
        'store_and_fwd_flag': store_and_fwd_flag
    }])

    try:
        pred_duration = best_model.predict(new_data)
        value = pred_duration.item()
        fare, distance = fare_estimate(pipe, new_data, value)
        speed = round(distance / (value / 3600), 2)

        st.success("✅ Prediction Done!")

        col6, col7, col8, col9 = st.columns(4)
        col6.metric("⏱️ Duration (sec)", round(value, 2))
        col7.metric("🕐 Duration (min)", round(value / 60, 2))
        col8.metric("💵 Estimated Fare", f"${fare}")
        col9.metric("📍 Distance (km)", distance)

        st.info(f"🚀 Estimated Speed: {speed} km/h")

    except Exception as e:
        st.error(f"Error: {e}")