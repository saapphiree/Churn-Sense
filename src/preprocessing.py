import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "Bank_Churn.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model")
encoders = {}
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_data(df):
    from sklearn.preprocessing import LabelEncoder
    df = df.drop(columns=['CustomerId', 'Surname'])
    categorical_cols = ["Gender", "Geography"]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        joblib.dump(le, os.path.join(MODEL_DIR, f"{col}_encoder.pkl"))
    return df

df = pd.read_csv(RAW_PATH)
df = clean_data(df)
df.to_csv(os.path.join(PROCESSED_DIR, "Bank_Churn_Cleaned.csv"), index=False)