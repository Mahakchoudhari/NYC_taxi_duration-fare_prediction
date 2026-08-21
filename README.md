# 🚖 RIDE Insight

<p align="center">
  <b>An End-to-End Machine Learning Application for NYC Taxi Trip Duration Prediction</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-Machine%20Learning-EA4335?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/Vite-Build%20Tool-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-7B61FF?style=for-the-badge" alt="Optuna">
  <img src="https://img.shields.io/badge/Deployment-Vercel%20%7C%20Render-000000?style=for-the-badge" alt="Deployment">
  <img src="https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge" alt="License">
</p>

---

## 🌐 Live Demo

| Service | Link |
|---------|------|
| 🚀 **Live Application** | [RIDE Insight](https://rideinsightnyc-taxi-fare-prediction-git-main-one-mind1.vercel.app/) |
| ⚡ **Backend API** | [FastAPI API](https://ride-insight-api.onrender.com/) |
| 📖 **API Documentation** | [Swagger UI](https://ride-insight-api.onrender.com/docs) |
| 💻 **GitHub Repository** | [View Repository](https://github.com/Mahakchoudhari/NYC_Taxi_fare_prediction_project6) |

> The frontend is deployed on **Vercel** and the FastAPI backend is deployed on **Render**.

---

# 📌 Project Overview

**RIDE Insight** is an end-to-end Machine Learning web application that predicts **NYC taxi trip duration** using real-world taxi trip data.

The project combines a trained **XGBoost regression model**, a custom preprocessing pipeline, a **FastAPI REST API**, and a **React + Vite frontend** to provide real-time predictions through a web interface.

The project demonstrates the complete journey from:

**Data → EDA → Feature Engineering → Model Development → Hyperparameter Tuning → Model Serialization → REST API → React Frontend → Cloud Deployment**

---

# 🎯 Problem Statement

Taxi trip duration depends on several factors, including:

- Pickup and drop-off locations
- Distance between locations
- Time of day
- Day of the week
- Passenger count
- Rush-hour patterns
- Trip timing

The goal of this project is to build a machine learning system that can estimate **taxi trip duration in seconds** from these input features.

---

# ✨ Key Features

- 🧠 **XGBoost Regression Model**
- 🔬 Advanced **geospatial and temporal feature engineering**
- 📈 **Optuna** hyperparameter optimization
- ⚙️ Reproducible preprocessing pipeline using **scikit-learn**
- 🚀 Production-style **FastAPI REST API**
- ⚛️ Interactive **React + Vite frontend**
- 📊 Analytics API endpoint
- 🔐 Environment-based configuration
- 🌍 CORS-enabled frontend/backend communication
- ☁️ Cloud deployment using **Vercel + Render**
- 📦 Serialized model and preprocessing pipeline
- ⚡ Real-time prediction through the deployed application

---

# 📊 Dataset

The model was developed using the **NYC Taxi Trip Duration Dataset** from Kaggle.

### Dataset Summary

| Property | Value |
|----------|------:|
| Total Rows | 1,458,644 |
| Total Columns | 10 |
| Problem Type | Regression |
| Target Variable | `trip_duration` |
| Target Unit | Seconds |

### Original Input Features

| Feature | Description |
|---------|-------------|
| `vendor_id` | Taxi vendor identifier |
| `pickup_datetime` | Date and time of pickup |
| `passenger_count` | Number of passengers |
| `pickup_longitude` | Pickup longitude |
| `pickup_latitude` | Pickup latitude |
| `dropoff_longitude` | Drop-off longitude |
| `dropoff_latitude` | Drop-off latitude |
| `store_and_fwd_flag` | Data storage indicator |

> The original dataset is not included in the repository because of its large file size.

---

# 🧹 Data Processing

The raw dataset goes through several preprocessing stages before being passed to the model.

```text
Raw NYC Taxi Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Outlier Detection & Handling
        │
        ▼
Feature Engineering
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Model Training
