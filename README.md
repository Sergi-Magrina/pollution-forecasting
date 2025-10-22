# Air Pollution Time-Series Forecasting Dashboard

Interactive Streamlit app to analyze historical PM2.5 and forecast future values with ARIMA and Prophet.

## 📂 Structure
```
pollution-forecasting/
├─ data/                 # raw + cleaned datasets (add your CSV here)
├─ notebooks/            # EDA & experiments
├─ app/                  # streamlit app code
│  └─ app.py
├─ models/               # (optional) saved models
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## 🚀 Quickstart

```bash
# 1) (optional) create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) run app
streamlit run app/app.py
```

## 📘 Usage
1. Place your dataset CSV in `data/` OR use the uploader in the app.
2. Select date column and PM2.5 column.
3. Choose model (ARIMA or Prophet) and forecast horizon (7/30/60 days).
4. View trend/seasonality, forecasts, and accuracy (MAE, RMSE, MAPE).

## 🗂 Dataset
You can use the Beijing PM2.5 dataset (link to be added). Ensure there is a datetime column and a PM2.5 numeric column.

## ✅ Roadmap (Phases)
- Phase 1: Data prep & EDA (resampling, rolling windows, decomposition)
- Phase 2: Forecasting (ARIMA/Prophet) + metrics
- Phase 3: Streamlit dashboard
- Phase 4: Deploy to Streamlit Cloud & publish GitHub repo

---

_Starter created on 2025-10-22_