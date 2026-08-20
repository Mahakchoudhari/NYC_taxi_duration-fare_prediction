import pandas as pd
import json


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading NYC dataset...")

df = pd.read_csv("NYC.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================
# DATETIME PROCESSING
# =========================================================

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"],
    errors="coerce"
)

df["pickup_hour"] = df["pickup_datetime"].dt.hour

df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

df["is_weekend"] = df["day_of_week"].isin([5, 6])

df["is_rush_hour"] = df["pickup_hour"].isin(
    [7, 8, 9, 10, 16, 17, 18, 19, 20]
)


# =========================================================
# TRIPS BY HOUR
# =========================================================

trips_by_hour = (
    df.groupby("pickup_hour")
    .size()
    .reset_index(name="trips")
)

trips_by_hour = [
    {
        "hour": int(row["pickup_hour"]),
        "trips": int(row["trips"])
    }
    for _, row in trips_by_hour.iterrows()
]


# =========================================================
# DURATION BY HOUR
# =========================================================

duration_by_hour = []

if "trip_duration" in df.columns:

    duration_by_hour_df = (
        df.groupby("pickup_hour")["trip_duration"]
        .mean()
        .reset_index()
    )

    duration_by_hour = [
        {
            "hour": int(row["pickup_hour"]),
            "duration_minutes": round(
                float(row["trip_duration"]) / 60,
                2
            )
        }
        for _, row in duration_by_hour_df.iterrows()
    ]


# =========================================================
# VENDOR DISTRIBUTION
# =========================================================

vendor_distribution = []

if "vendor_id" in df.columns:

    vendor_df = (
        df.groupby("vendor_id")
        .size()
        .reset_index(name="trips")
    )

    vendor_distribution = [
        {
            "vendor_id": int(row["vendor_id"]),
            "trips": int(row["trips"])
        }
        for _, row in vendor_df.iterrows()
    ]


# =========================================================
# AVERAGE DURATION
# =========================================================

avg_duration_minutes = None

if "trip_duration" in df.columns:

    avg_duration_minutes = round(
        float(df["trip_duration"].mean()) / 60,
        2
    )


# =========================================================
# AVERAGE PASSENGERS
# =========================================================

avg_passengers = None

if "passenger_count" in df.columns:

    avg_passengers = round(
        float(df["passenger_count"].mean()),
        2
    )


# =========================================================
# RUSH HOUR ANALYSIS
# =========================================================

rush_trips = int(
    df["is_rush_hour"].sum()
)

normal_trips = int(
    (~df["is_rush_hour"]).sum()
)


# =========================================================
# WEEKDAY / WEEKEND
# =========================================================

weekday_trips = int(
    (~df["is_weekend"]).sum()
)

weekend_trips = int(
    df["is_weekend"].sum()
)


# =========================================================
# FINAL ANALYTICS OBJECT
# =========================================================

analytics = {

    "success": True,

    "total_trips": int(len(df)),

    "avg_duration_minutes":
        avg_duration_minutes,

    "avg_passengers":
        avg_passengers,

    "rush_hour": {

        "rush_trips":
            rush_trips,

        "normal_trips":
            normal_trips
    },

    "day_type": {

        "weekday_trips":
            weekday_trips,

        "weekend_trips":
            weekend_trips
    },

    "trips_by_hour":
        trips_by_hour,

    "duration_by_hour":
        duration_by_hour,

    "vendor_distribution":
        vendor_distribution
}


# =========================================================
# SAVE ANALYTICS JSON
# =========================================================

with open(
    "analytics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        analytics,
        file,
        indent=2
    )


# =========================================================
# SUCCESS MESSAGE
# =========================================================

print()
print("==============================================")
print(" analytics.json created successfully!")
print("==============================================")

print(
    "Total trips:",
    analytics["total_trips"]
)

print(
    "Average duration:",
    analytics["avg_duration_minutes"],
    "minutes"
)

print(
    "Average passengers:",
    analytics["avg_passengers"]
)

print(
    "Rush hour trips:",
    analytics["rush_hour"]["rush_trips"]
)

print(
    "Normal hour trips:",
    analytics["rush_hour"]["normal_trips"]
)

print(
    "Weekday trips:",
    analytics["day_type"]["weekday_trips"]
)

print(
    "Weekend trips:",
    analytics["day_type"]["weekend_trips"]
)

print()
print("analytics.json is ready for Render deployment.")