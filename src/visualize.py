import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model")

TEST_PATH = os.path.join(PROCESSED_DIR, "test_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_prediction_model.pkl")

def load_data(path):
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_PATH}. Please run the training script first.")
    else:
        return pd.read_csv(TEST_PATH)

df = load_data(TEST_PATH)

def split_xy(df, target_column):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y

x_test, y_test = split_xy(df, target_column="Exited")

model = joblib.load(MODEL_PATH)

importance = model.feature_importances_
features = pd.read_csv(TEST_PATH).drop(columns=["Exited"]).columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()