# Churn Sense – Bank Customer Churn Prediction

**Churn Sense** is a machine learning project that predicts whether a bank customer will churn (leave) or stay, based on their demographic and account information. The project uses XGBoost for classification and includes bias analysis and a Streamlit web app for interactive predictions.

---

## Features

- Data preprocessing and train/test split
- Model training using XGBoost with hyperparameter tuning
- Accuracy evaluation on test data
- Bias analysis by gender and geography
- Feature importance analysis
- Interactive Streamlit app for predicting customer churn

---

## Getting Started

### Prerequisites

- Python 3.9+
- Packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `streamlit`

Install the dependencies using:

```bash
pip install -r requirements.txt
````

---

### Folder Structure

```
churn_prediction/
├─ data/
│  └─ processed/          # Cleaned, train, and test datasets
├─ model/                 # Saved models and label encoders
├─ src/
│  ├─ preprocessing.py    # Data preprocessing and train/test split
│  ├─ train.py            # Model training and saving
│  ├─ test.py             # Model evaluation and bias analysis
│  └─ app.py              # Streamlit app for predictions
└─ README.md
```

---

## Usage

### 1. Train the Model

```bash
python src/train.py
```

### 2. Evaluate the Model

```bash
python src/test.py
```

### 3. Run the Web App

```bash
streamlit run src/app.py
```

Enter customer details in the app to see whether they are likely to **churn** or **stay**.

---

## Dataset

* Sourced from MavelAnalytics: [Bank Customer Churn Dataset](https://mavenanalytics.io/data-playground/bank-customer-churn)
---

## Performance

* **Test Accuracy:** ~85.5%
* Bias metrics included for **Gender** and **Geography**
* Feature importance highlights key factors affecting churn

---

## License

This project is licensed under the **MIT License** – see the LICENSE file for details.

``` 

If you want, I can also make a **short, recruiter-friendly version** that emphasizes results and key skills for your portfolio. Do you want me to do that?
```
