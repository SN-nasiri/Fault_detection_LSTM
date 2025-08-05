# Fault Detection with LSTM

## 📁 ساختار پروژه
- `train.py` – آموزش مدل
- `inference.py` – پیش‌بینی با مدل ذخیره‌شده
- `models/model.py` – تعریف مدل LSTM
- `data/clean_fault_data.csv` – دیتای پردازش‌شده

## 🧠 مراحل آموزش مدل
- خواندن داده
- نرمال‌سازی (MinMax)
- one-hot encoding
- class weight
- آموزش LSTM
- ذخیره مدل: `saved_model/fault_lstm_model.h5`

## 🧪 اجرای پیش‌بینی
```bash
python inference.py



---

### 🔄 ۳. **هم‌راستا کردن `train.py` و `inference.py`**

یکی از اصلی‌ترین مشکلات تفاوت در نرمال‌سازی بود:

- `train.py`: `MinMaxScaler`
- `inference.py`: `StandardScaler`

🔧 **باید یکی باشه** — ترجیحاً `MinMaxScaler` چون در مدل train شده همینه.

در `inference.py` این خط رو اصلاح کن:

```python
from sklearn.preprocessing import MinMaxScaler  # ← تغییر به MinMax
scaler = MinMaxScaler()
