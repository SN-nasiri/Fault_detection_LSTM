import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from models.model import build_lstm_model
import os

# ---------- 📦 Load Data ----------
data_path = os.path.join('data', 'clean_fault_data.csv')
df = pd.read_csv(data_path)

# ---------- 🧹 Preprocessing ----------
df = df.dropna()
X = df[['Ia', 'Ig', 'Eg', 'Fg', 'Pg', 'Va', 'Vg']].values
y = df['Fault'].values

# Encode class labels
unique_labels = sorted(np.unique(y))  # ✅ اصلاح‌شده
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label_mapping[val] for val in y])

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels
y_categorical = to_categorical(y_encoded)

# ---------- 🧪 Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ---------- ⚖️ Class Weights ----------
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = {i: float(w) for i, w in enumerate(weights)}

# ---------- 🧠 Build Model ----------
model = build_lstm_model(input_shape=(1, X_train.shape[2]), output_dim=y_train.shape[1])

# ---------- 🚂 Train ----------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)

# ---------- 💾 Save Model ----------
os.makedirs("saved_model", exist_ok=True)
model.save('saved_model/final_lstm_model.h5')
print("✅ Model saved successfully to 'saved_model/fault_lstm_model.h5'")
