# 🚖 RIDE Insight

<p align="center">
  <b>An End-to-End Machine Learning Application for Real-Time NYC Taxi Trip Prediction</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-ML-red?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react" alt="React">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Optuna-Tuning-purple?style=for-the-badge" alt="Optuna">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

## 📌 Project Overview

**NYC Taxi Fare & Trip Duration Prediction** is a production-ready, end-to-end machine learning application that predicts **taxi trip duration and estimated fare** using real NYC taxi trip data.

The project evolved from a machine learning notebook to a **full-stack ML application** featuring:

- 🧠 **XGBoost** with advanced hyperparameter tuning via **Optuna**
- ⚙️ Custom **preprocessing pipeline** with geospatial & temporal feature engineering
- 🚀 **FastAPI** REST backend for real-time model inference
- ⚛️ **React + Vite** responsive frontend
- 📊 **Streamlit** prototype for rapid ML demos

---

## 🌐 Live Application

> 🚧 Deployment in progress

| Service | Status |
|---------|--------|
| Frontend (React) | 🚧 Coming Soon |
| Backend API (FastAPI) | 🚧 Coming Soon |
| API Documentation | 🚧 Coming Soon |

---

## 🎯 Key Highlights

- Trained on **1.45 Million** real NYC taxi trip records
- Advanced **feature engineering** — Haversine distance, Manhattan distance, bearing, rush hour flags, cyclical time encoding
- **Hyperparameter tuning** using Optuna with 15 trials
- Custom **scikit-learn Pipeline** for reproducible preprocessing
- **REST API** for real-time inference
- **Full-stack integration** — React frontend + FastAPI backend

---

## 📊 Dataset

**Source:** NYC Taxi Trip Duration Dataset (Kaggle)

| Property | Value |
|----------|-------|
| Total Rows | 1,458,644 |
| Total Columns | 10 |
| Problem Type | Regression |
| Target Variable | `trip_duration` (seconds) |

### Features Used

| Feature | Description |
|---------|-------------|
| `vendor_id` | Taxi vendor identifier |
| `pickup_datetime` | Date and time of pickup |
| `passenger_count` | Number of passengers |
| `pickup_longitude/latitude` | Pickup GPS coordinates |
| `dropoff_longitude/latitude` | Dropoff GPS coordinates |
| `store_and_fwd_flag` | Trip data storage flag |

> Dataset not included due to large file size. Download from [Kaggle](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data).

---

## 🔧 Feature Engineering

| Feature | Description |
|---------|-------------|
| `haversine_km` | Straight-line distance between pickup & dropoff |
| `manhattan_km` | Road-approximated distance |
| `route_ratio` | Manhattan / Haversine ratio |
| `bearing` | Direction of travel |
| `pickup_hour` | Hour of pickup |
| `day_of_week` | Day of the week |
| `is_rush_hour` | Rush hour flag (7-10 AM, 4-8 PM) |
| `is_weekend` | Weekend flag |
| `is_night` | Night time flag |
| `hour_sin/cos` | Cyclical encoding of hour |
| `dow_sin/cos` | Cyclical encoding of day |

---

## 🤖 Model Development

### Models Compared

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Ridge / Lasso | Regularized regression |
| Random Forest | Ensemble tree model |
| XGBoost | **Final Model** ✅ |

### Hyperparameter Tuning

Used **Optuna** for intelligent hyperparameter search over 15 trials optimizing:
`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`

### Model Performance

| Metric | Score |
|--------|-------|
| R² Score | ~75%+ |
| MAE | Low |
| RMSE | Low |

---

## 🔄 Project Workflow

```
NYC Taxi Dataset (1.45M rows)
         │
         ▼
  Data Exploration & EDA
         │
         ▼
  Outlier Handling (IQR)
         │
         ▼
  Feature Engineering
         │
         ▼
  Model Comparison
         │
         ▼
  Hyperparameter Tuning (Optuna)
         │
         ▼
  Final XGBoost Pipeline
         │
         ▼
  Model Serialization (Pickle)
         │
    ┌────┴────┐
    ▼         ▼
Streamlit   FastAPI Backend
  App        REST API
               │
               ▼
         React Frontend
               │
               ▼
      Real-Time Prediction
```

---

## 🗂️ Project Structure

```
NYC_Taxi_Trip_Duration/
│
├── notebook/
│   └── nyc_taxi.ipynb         ← ML Notebook
│
├── backend/
│   ├── main.py                ← FastAPI app
│   ├── best_model.pkl         ← Trained model
│   └── pipe.pkl               ← Preprocessing pipeline
│
├── frontend/
│   ├── src/
│   └── package.json
│
├── streamlit_app/
│   └── app.py                 ← Streamlit prototype
│
├── requirements.txt
└── README.md
```

---

## 📥 Download Model Files

`.pkl` files not included due to size. Download from Google Drive:

| File | Link |
|------|------|
| best_model.pkl | [Download](https://drive.google.com/your-link-here) |
| pipe.pkl | [Download](https://drive.google.com/your-link-here) |

---

## 🚀 How to Run

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

### FastAPI Backend
```bash
cd backend
uvicorn main:app --reload
```

### React Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 📦 Requirements

```
streamlit
fastapi
uvicorn
xgboost
scikit-learn
pandas
numpy
optuna
matplotlib
seaborn
```

---

## 👩‍💻 Author

**Mahak Choudhari**
B.Tech — Artificial Intelligence & Machine Learning (2nd Year)
[GitHub](https://github.com/Mahakchoudhari) | [LinkedIn](https://linkedin.com/in/mahakchoudhari)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
