# inference.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 🎯 مسیر مدل آموزش‌داده‌شده
MODEL_PATH = "saved_model/fault_lstm_model.h5"

# 🎯 ستون‌های ویژگی
FEATURE_COLUMNS = ['Ia', 'Ig', 'Eg', 'Fg', 'Pg', 'Va', 'Vg']

# 🎯 لیبل‌ها به ترتیب اندیس
CLASS_LABELS = [1.0, 2.1, 2.2, 2.3, 3.1, 3.2, 4.0]

def preprocess_input_data(df):
    """نرمال‌سازی و reshape دیتای جدید برای inference"""
    X = df[FEATURE_COLUMNS].values

    # نرمال‌سازی
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # reshape برای LSTM
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    return X_reshaped

def load_trained_model():
    """بارگذاری مدل ذخیره‌شده"""
    model = load_model(MODEL_PATH)
    return model

def predict_fault_type(model, X_input):
    """پیش‌بینی نوع خطا برای داده ورودی"""
    y_proba = model.predict(X_input)
    y_pred_indices = np.argmax(y_proba, axis=1)
    y_pred_labels = [CLASS_LABELS[i] for i in y_pred_indices]
    return y_pred_labels, y_proba

def run_inference(input_csv_path):
    df_input = pd.read_csv(input_csv_path)
    X_input = preprocess_input_data(df_input)
    model = load_trained_model()
    y_pred, y_proba = predict_fault_type(model, X_input)

    df_input['Predicted Fault'] = y_pred
    return df_input[['Predicted Fault']], y_proba

if __name__ == "__main__":
    result, proba = run_inference("data/clean_fault_data.csv")
    print(result.head())
