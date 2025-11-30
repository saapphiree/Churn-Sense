import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLEANED_PATH = os.path.join(PROCESSED_DIR, "Bank_Churn_Cleaned.csv")
df = pd.read_csv(CLEANED_PATH)

def split_xy(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y

def build_model(x_train, y_train):
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=15,
        subsample=0.8,
        colsample_bytree=1,
        random_state=42,
        eval_metric='logloss',
        objective='binary:logistic'
    )
    model.fit(x_train, y_train)
    return model

def main():
    x, y = split_xy(df, target_column="Exited")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    train_path = os.path.join(PROCESSED_DIR, "train_data.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_data.csv")
    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    model = build_model(x_train, y_train)

    model_path = os.path.join(MODEL_DIR, "churn_prediction_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()