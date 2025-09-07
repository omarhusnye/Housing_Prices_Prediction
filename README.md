# 🏠 House Price Prediction  

## 📌 Overview  
This project predicts **house prices** based on different features (e.g., number of rooms, area, location, etc.).  
It uses **machine learning models** such as:  
- Linear Regression  
- Random Forest Regressor  

The goal is to compare model performance and choose the one that provides the best predictions.  

---

## ⚙️ Installation  
Clone the repository and install the required dependencies:  

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

---

## 📂 Dataset  
- The dataset contains information about houses (features) and their selling prices (target).  
- Example columns:  
  - `area`  
  - `bedrooms`  
  - `bathrooms`  
  - `location`  
  - `price` (target variable)  

---

## 🚀 How to Run  

### 1. Train Models  
Run the training script to fit models and evaluate performance:  

```python
for name, model in models.items():
    impute_model(model)
```

This will print **train & test metrics** for each model.  

### 2. Make Predictions  
Select a trained model and generate predictions:  

```python
rf_model = models['Random_Forest Model']
rf_model.fit(x_train, y_train)
predictions = rf_model.predict(x_test)

print(predictions[:10])  # Show first 10 predictions
```

---

## 📊 Evaluation Metrics  
The following metrics are used to evaluate model performance:  
- **Mean Absolute Error (MAE)**  
- **R² Score**  


## 📈 Results  
- Linear Regression works well when features have linear relationships.  
- Random Forest usually provides better accuracy by capturing non-linear patterns.  

---

## 🛠️ Tech Stack  
- **Python 3.x**  
- **pandas**  
- **scikit-learn**  
- **matplotlib / seaborn** (for visualization, optional)  

---

## 🔮 Future Improvements  
- Add more ML models (XGBoost, Gradient Boosting).  
- Perform hyperparameter tuning.  
- Deploy the model using Flask or FastAPI.  
# gtc_ml_Housing_Prices
