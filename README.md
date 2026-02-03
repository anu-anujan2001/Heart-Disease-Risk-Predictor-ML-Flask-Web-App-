# 🫀 Heart Disease Risk Predictor (ML + Flask Web App)

A machine learning project that predicts the risk of heart disease using patient medical features.
Includes an end-to-end pipeline (preprocessing + model) and a professional single-page Flask UI.

## ✅ Features
- Data preprocessing (scaling numeric + one-hot encoding categorical)
- Multiple ML models tested (Logistic, KNN, Decision Tree, Random Forest, SVM)
- Best model saved as a **Pipeline** (`heart_disease_model.pkl`)
- Professional **one-page web UI** (Flask + HTML/CSS)
- Outputs prediction + probability (if supported by model)

## 🧾 Dataset
Kaggle: Heart Disease Dataset  
Columns used:
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`  
Target:
`target` (0 = No disease, 1 = Disease)

## 🚀 Run Locally

### 1) Clone repo
```bash
git clone https://github.com/<your-username>/heart-disease-predictor.git
cd heart-disease-predictor
