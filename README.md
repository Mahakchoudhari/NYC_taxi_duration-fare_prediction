# 🚖 NYC Taxi Fare & Duration Prediction (Hyperparameter Tuned)

## 📌 Project Overview

This project focuses on predicting:

* 💰 Taxi Fare
* ⏱️ Trip Duration

using Machine Learning models on the **NYC Taxi dataset**.

The key highlight of this project is **Hyperparameter Tuning** to improve model performance and achieve better accuracy compared to baseline models.

---

## 🎯 Objectives

* Build regression models for fare and duration prediction
* Perform data preprocessing & feature engineering
* Apply multiple ML algorithms
* Optimize models using **Hyperparameter Tuning**
* Evaluate and compare model performance

---

## 📂 Dataset

The dataset contains taxi trip details such as:

* Pickup & Dropoff coordinates
* Datetime information
* Passenger count
* Distance (engineered)
* Trip duration (target variables)

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing

* Handling missing values
* Removing outliers
* Feature scaling
* Datetime feature extraction (hour, day, etc.)

### 2️⃣ Feature Engineering

* Distance calculation (Haversine formula)
* Time-based features
* Location-based insights

---

## 🤖 Models Used

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost 

---

## 🔥 Hyperparameter Tuning

Hyperparameter tuning was performed using:

* GridSearchCV / RandomizedSearchCV / Bayesian optimization

### Example Tuned Parameters:


* **XGBoost**

  * learning_rate
  * n_estimators
  * max_depth

---

## 📊 Model Evaluation

Models were evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

### ✅ Outcome

* Tuned models performed **significantly better** than baseline models
* Reduced error and improved generalization

---

## 📈 Results Summary

| Model         | Before Tuning | After Tuning |
| ------------- | ------------- | ------------ |
| Random Forest | Moderate      | Improved ✅   |
| Decision Tree | Overfitting   | Controlled ✅ |
| XGBoost       | Good          | Best 🚀      |

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 📁 Project Structure

```
NYC_taxi_duration-fare_prediction/
│
├── data/
├── notebooks/
│   └── hyperparameter_tuned.ipynb
├── models/
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Mahakchoudhari/NYC_taxi_duration-fare_prediction.git

# Navigate to project
cd NYC_taxi_duration-fare_prediction

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
```

---

## 💡 Key Learnings

* Importance of feature engineering in real-world datasets
* Handling noisy and outlier-heavy data
* Hyperparameter tuning improves performance significantly
* Trade-off between bias and variance

---

## 📌 Future Improvements

* Use Deep Learning models
* Deploy model using Flask / Streamlit
* Real-time prediction system

---

## 👩‍💻 Author

**Mehak Choudhari**

---

⭐ If you like this project, don't forget to star the repo!
