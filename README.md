# 🚖 NYC Taxi Fare & Trip Duration Prediction

## 📌 Overview

This project predicts:

- ⏱️ Taxi Trip Duration
- 💰 Estimated Taxi Fare
- 🚗 Estimated Speed

using Machine Learning on the NYC Taxi Dataset.

The project uses:

- Feature Engineering
- XGBoost Regressor
- Hyperparameter Tuning
- Streamlit Web App

---

# 🎯 Objectives

- Predict trip duration accurately
- Estimate taxi fare dynamically
- Improve performance using hyperparameter tuning
- Deploy model using Streamlit

---

# 📂 Dataset

The dataset contains:

- Vendor ID
- Pickup & Dropoff Datetime
- Passenger Count
- Pickup & Dropoff Coordinates
- Store and Forward Flag
- Trip Duration

Dataset Size:

- Rows: 1,458,644
- Columns: 10

---

# ⚙️ Workflow

## 1️⃣ Data Preprocessing

- Removed unnecessary columns
- Checked missing values
- Removed outliers
- Log transformation of target variable

## 2️⃣ Feature Engineering

Created features such as:

- Pickup Hour
- Day of Week
- Weekend Indicator
- Rush Hour Indicator
- Night Indicator
- Haversine Distance
- Manhattan Distance
- Route Ratio
- Bearing Angle

---

# 🤖 Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor 🚀

---

# 🔥 Hyperparameter Tuning

Performed tuning using:

- RandomizedSearchCV
- GridSearchCV
- Optuna

### Best Optuna Parameters

```python
{
 'n_estimators': 792,
 'max_depth': 8,
 'learning_rate': 0.074,
 'subsample': 0.828,
 'colsample_bytree': 0.633,
 'min_child_weight': 8
}
```

---

# 📊 Model Performance

| Metric | Score |
|--------|--------|
| MAE | 203.24 |
| RMSE | 298.30 |
| R² Score | 0.73 |
| Accuracy | 72.75% |

---

# 🚖 Sample Prediction

| Prediction | Value |
|------------|-------|
| Trip Duration | 517.65 sec |
| Distance | 1.81 km |
| Estimated Fare | $11.02 |
| Speed | 12.59 km/h |

---

# 🌐 Streamlit App

The project includes a Streamlit web app for real-time predictions.

## Features

✅ Trip duration prediction  
✅ Fare estimation  
✅ Speed calculation  
✅ Interactive UI  

---

# 📁 Project Structure

```bash
NYC_taxi_duration-fare_prediction/
│
├── notebooks/
│   └── hyperparameter_tuned.ipynb
│
├── models/
│   ├── best_model.pkl
│   └── pipe.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- Streamlit

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/Mahakchoudhari/NYC_taxi_duration-fare_prediction.git
```

## Move to Project Folder

```bash
cd NYC_taxi_duration-fare_prediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 💾 Model Saving

```python
pickle.dump(best_model, open('best_model.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))
```

---

# 🔮 Future Improvements

- Live traffic integration
- Map visualization
- Deep learning models
- Cloud deployment

---

# 👩‍💻 Author

**Mehak Choudhari**

---

⭐ If you like this project, don't forget to star the repository!
