# 💳 Credit Risk Assessment App

This is a Streamlit-based web application that predicts the credit risk of a customer using a trained machine learning model (XGBoost). It is designed to assist financial institutions in identifying potential defaulters.

## 📌 Project Overview

The Credit Risk Assessment App leverages a supervised machine learning model to classify customers into two categories:
- ✅ Low Risk
- 🚨 High Risk

It uses customer financial and demographic data to make predictions, making it a useful tool for credit scoring and risk management.

---

## 📁 Repository Structure

```
credit-risk-assessment/
│
├── app.py                  # Streamlit web application
├── model_training.ipynb    # Notebook for preprocessing, training, and evaluation
├── xgb_model.pkl           # Trained XGBoost model
├── scaler.pkl              # Scaler used for feature normalization
├── credit_data.csv         # Dataset used for training (Give Me Some Credit)
└── README.md               # Project documentation
```

---

## 📊 Dataset Used

- Source: [Kaggle - Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data)
- Target Column: `SeriousDlqin2yrs` (1 = Default, 0 = No Default)
- Features Used:
  - RevolvingUtilizationOfUnsecuredLines
  - age
  - NumberOfTime30-59DaysPastDueNotWorse
  - DebtRatio
  - MonthlyIncome
  - NumberOfOpenCreditLinesAndLoans
  - NumberOfTimes90DaysLate
  - NumberRealEstateLoansOrLines
  - NumberOfTime60-89DaysPastDueNotWorse
  - NumberOfDependents

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy, Scikit-learn, XGBoost
- Streamlit (Frontend Web App)
- Joblib (Model Serialization)

---

## 🚀 How to Run the App

1. Clone the repository:

```bash
git clonegit clone git@github.com:abaidurerehman/Credit-Risk-Assessment-App.git
cd credit-risk-assessment
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the following files are present:
   - `xgb_model.pkl`
   - `scaler.pkl`

4. Run the Streamlit app:

```bash
streamlit run app.py
```


## 🧠 Model Training Overview

The training process includes:
- Handling missing values
- Scaling features using StandardScaler
- Handling class imbalance using SMOTE
- Training an XGBoost classifier
- Saving the trained model and scaler using Joblib

Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Curve


## ✨ App Features

- Interactive input form with clean UI
- Input value ranges and validations
- Display of both prediction class and probability
- Real-time risk prediction
- Stylish UI with custom CSS

## 🔒 Disclaimer

This app is for educational and demonstration purposes only. It should not be used in production without additional validation, security, and ethical compliance.
