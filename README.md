# 🏡 California House Price Prediction

This project predicts house prices in California using machine learning models.  
We explore the dataset, apply preprocessing, train multiple models, tune hyperparameters, and visualize the results.

---

## 📂 Project Structure
- `housing.csv` → Dataset (California Housing Prices)
- `notebooks/` → Jupyter notebooks with step-by-step workflow
- `best_rf_model.pkl` → Saved best model (Random Forest after tuning)

---

## ⚙️ Workflow

### 1. Import Libraries
- `pandas`, `numpy` for data manipulation  
- `matplotlib`, `seaborn` for visualization  
- `scikit-learn` for ML models and evaluation  
- `joblib` for saving/loading the trained model  

### 2. Load Dataset
- Load the **California Housing dataset** (`housing.csv`)  
- Explore structure, missing values, and feature distributions  

### 3. Exploratory Data Analysis (EDA)
- Check descriptive statistics  
- Plot correlations between numeric features  
- Visualize feature distributions  

### 4. Data Preprocessing
- Handle missing values (`total_bedrooms`)  
- Encode categorical variable (`ocean_proximity`)  
- Feature scaling with `StandardScaler`  
- Train/Test split  

### 5. Model Training
Train and evaluate multiple models:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- K-Nearest Neighbors (KNN)  
- Support Vector Regressor (SVR)  

### 6. Model Evaluation
Metrics used:
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

### 7. Hyperparameter Tuning
- Used `RandomizedSearchCV` for Random Forest  
- Optimized parameters: `n_estimators`, `max_depth`, `min_samples_split`  
- Selected the **best estimator** (`best_rf`)  

### 8. Visualization
- **Feature Importance** → Which features affect house prices the most  
- **Predicted vs Actual** → Check model accuracy  
- **Residuals Plot** → Inspect model errors  

### 9. Save Final Model
Save the best Random Forest model:
```python
import joblib
joblib.dump(best_rf, "best_rf_model.pkl")
```

---

## 📊 Results
| Model              | MAE     | RMSE    | R²   |
|--------------------|---------|---------|------|
| Linear Regression  | ~50,503 | ~69,745 | 0.64 |
| Decision Tree      | ~45,288 | ~71,446 | 0.62 |
| Random Forest      | ~33,280 | ~51,493 | 0.80 |
| KNN                | ~77,426 | ~100,415| 0.26 |
| SVR                | ~89,955 | ~120,093| -0.05 |

✅ Best model: **Random Forest Regressor (after tuning)**

---

## 🚀 Future Work
- Try boosting algorithms (XGBoost, LightGBM, CatBoost)  
- Deploy the model as a Flask/FastAPI web service  
- Build an interactive dashboard with Streamlit  

---

## 👨‍💻 Author
- Developed by **Omar Husnye**  
- 📧 Contact: [your-email@example.com]  
