import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model")

TEST_PATH = os.path.join(PROCESSED_DIR, "test_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_prediction_model.pkl")
LE_GEOGRAPHY_PATH = os.path.join(MODEL_DIR, "Geography_encoder.pkl")
LE_GENDER_PATH = os.path.join(MODEL_DIR, "Gender_encoder.pkl")

def load_data(path):
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_PATH}. Please run the training script first.")
    return pd.read_csv(TEST_PATH)

def split_xy(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y

def load_model(MODEL_PATH):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run the training script first.")
    return joblib.load(MODEL_PATH)

def calc_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return acc, fpr, fnr

def main():
    model = load_model(MODEL_PATH)
    df = load_data(TEST_PATH)
    x, y = split_xy(df, target_column="Exited")

    predictions = model.predict(x)
    accuracy = np.mean(predictions == y)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    #Gender Bias
    print("=== Gender Bias ===")
    for gender_value, gender_name in zip([0,1], ["Male","Female"]):
        mask = x["Gender"] == gender_value
        acc, fpr, fnr = calc_metrics(y[mask], predictions[mask])
        print(f"{gender_name}: Accuracy={acc*100:.2f}%, FPR={fpr*100:.2f}%, FNR={fnr*100:.2f}%")

    #(Germany vs Others)
    print("\n=== Geography Bias ===")
    germany_value = 2
    ger_mask = x["Geography"] == germany_value
    other_mask = x["Geography"] != germany_value

    for mask, name in zip([ger_mask, other_mask], ["Germany","Other"]):
        acc, fpr, fnr = calc_metrics(y[mask], predictions[mask])
        print(f"{name}: Accuracy={acc*100:.2f}%, FPR={fpr*100:.2f}%, FNR={fnr*100:.2f}%")

if __name__ == "__main__":
    main()