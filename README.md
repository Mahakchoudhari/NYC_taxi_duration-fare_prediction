# 🚖RIDE Insight

<p align="center">
  <b>An End-to-End Machine Learning Application for NYC Taxi Trip Prediction</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/XGBoost-ML-red?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

## 📌 Overview

**NYC Taxi Fare & Trip Duration Prediction** is an end-to-end machine learning project that predicts **taxi trip duration** using trip details, geographical information, passenger data, and temporal features.

The project started as a machine learning and Streamlit prototype and was later developed into a **full-stack ML application** using:

- 🧠 XGBoost for machine learning
- ⚙️ Custom preprocessing and feature engineering
- 🚀 FastAPI for model serving
- ⚛️ React + Vite for the frontend
- 🔌 REST API for frontend-backend communication

The application accepts trip information from the user and returns a real-time prediction along with derived trip insights such as estimated distance, speed, and fare.

---

## 🌐 Live Application

> 🚧 Deployment in progress

| Service | Status |
|---|---|
| Frontend | 🚧 Coming Soon |
| Backend API | 🚧 Coming Soon |
| API Documentation | 🚧 Coming Soon |

Once deployment is completed, the links will be added here.

---

# 🎯 Objectives

The primary objectives of this project are:

- Predict NYC taxi trip duration using machine learning
- Perform meaningful temporal and geographical feature engineering
- Compare multiple regression algorithms
- Optimize the final model using hyperparameter tuning
- Build a reusable preprocessing pipeline
- Serialize the trained ML model
- Create a REST API for real-time inference
- Develop a responsive React frontend
- Integrate frontend and backend into a complete ML application
- Prepare the project for cloud deployment

---

# 🧩 Problem Statement

Taxi trip duration depends on several factors, including:

- Pickup and dropoff locations
- Time of day
- Day of the week
- Passenger count
- Rush-hour conditions
- Distance between locations
- Route characteristics

The goal of this project is to learn these relationships from historical NYC taxi data and predict the expected trip duration for a new taxi ride.

---

# 📊 Dataset

The project uses the **NYC Taxi Trip Duration Dataset**.

### Dataset Information

| Property | Value |
|---|---:|
| Total Rows | 1,458,644 |
| Total Columns | 10 |
| Problem Type | Regression |
| Target Variable | Trip Duration |

### Dataset Features

| Feature | Description |
|---|---|
| `vendor_id` | Taxi vendor identifier |
| `pickup_datetime` | Date and time of pickup |
| `dropoff_datetime` | Date and time of dropoff |
| `passenger_count` | Number of passengers |
| `pickup_longitude` | Pickup longitude |
| `pickup_latitude` | Pickup latitude |
| `dropoff_longitude` | Dropoff longitude |
| `dropoff_latitude` | Dropoff latitude |
| `store_and_fwd_flag` | Indicates whether trip data was stored before forwarding |
| `trip_duration` | Duration of the taxi trip in seconds |

> The original dataset is not included in the GitHub repository because of its large file size.

---

# 🔄 End-to-End Workflow

```text
                NYC Taxi Dataset
                       │
                       ▼
              Data Exploration
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
             Feature Engineering
                       │
                       ▼
              Model Development
                       │
                       ▼
             Model Comparison
                       │
                       ▼
          Hyperparameter Optimization
                       │
                       ▼
               Final XGBoost Model
                       │
                       ▼
           Preprocessing Pipeline
                       │
                       ▼
                Model Serialization
                       │
                       ▼
                 FastAPI Backend
                       │
                       ▼
                 REST API
                       │
                       ▼
                React Frontend
                       │
                       ▼
              Real-Time Prediction
