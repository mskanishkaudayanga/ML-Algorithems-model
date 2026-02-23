# Sri Lanka Mobile Phone Price Prediction using XGBoost

## Project Overview

This project builds an end-to-end machine learning pipeline to **predict mobile phone prices** in the Sri Lanka market using data scraped from [ikman.lk](https://ikman.lk/en/ads/sri-lanka/mobile-phones).

**Algorithm:** XGBoost Regressor with RandomizedSearchCV hyperparameter tuning.

**Key Features:**
- Web scraping pipeline for data collection (5,000+ records)
- Automated data preprocessing and feature engineering
- XGBoost model training with early stopping
- Comprehensive evaluation (RMSE, MAE, R2)
- SHAP-based model explainability
- Interactive Streamlit web application for price prediction

---

## Folder Structure

```
Sri-Lanka-Mobile-Phone-Price-Prediction/
|
|-- scrape.py                     # Web scraper for ikman.lk
|
|-- src/
|   |-- __init__.py
|   |-- utils.py                  # Shared utility functions
|   |-- preprocess.py             # Data cleaning & feature engineering
|   |-- train.py                  # XGBoost model training
|   |-- evaluate.py               # Model evaluation & plots
|   |-- explain.py                # SHAP explainability analysis
|
|-- models/
|   |-- xgboost_model.joblib      # Trained model
|   |-- label_encoders.joblib     # Fitted label encoders
|   |-- scaler.joblib             # Fitted standard scaler
|   |-- feature_meta.json         # Feature column metadata
|   |-- column_info.json          # Encoder class info for UI
|   |-- best_hyperparameters.json # Best hyperparameters from tuning
|   |-- shap_explainer.joblib     # SHAP TreeExplainer
|
|-- data/
|   |-- mobile_phones.csv         # Raw scraped data
|   |-- processed.csv             # Cleaned & processed data
|   |-- train_val_test_split.npz  # Train/val/test arrays
|
|-- outputs/
|   |-- metrics.json              # Evaluation metrics
|   |-- metrics_table.csv         # Metrics in tabular form
|   |-- plots/
|       |-- predicted_vs_actual.png
|       |-- residual_histogram.png
|       |-- shap_summary.png
|       |-- shap_dependence.png
|       |-- feature_importance.png
|
|-- app/
|   |-- streamlit_app.py          # Interactive web app
|
|-- requirements.txt
|-- README.md
|-- report_outline.md
```

---

## Installation

### 1. Clone or download the project

```bash
cd "Sri-Lanka-Mobile-Phone-Price-Prediction"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Commands (Step-by-Step)

### Step 1: Scrape Data (optional — data already included)

```bash
python scrape.py --max_pages 200 --output data/mobile_phones.csv
```

### Step 2: Preprocess Data

```bash
python src/preprocess.py --input data/mobile_phones.csv --output processed.csv
```

### Step 3: Train the Model

```bash
python src/train.py --data data/processed.csv
```

### Step 4: Evaluate the Model

```bash
python src/evaluate.py
```

### Step 5: Generate SHAP Explanations

```bash
python src/explain.py
```

### Step 6: Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## Dataset

- **Source:** [ikman.lk — Mobile Phones](https://ikman.lk/en/ads/sri-lanka/mobile-phones)
- **Records:** 5,000+
- **Key Columns:**
  | Column | Description |
  |--------|-------------|
  | `title` | Full ad title |
  | `brand` | Phone manufacturer (Apple, Samsung, etc.) |
  | `model` | Phone model name |
  | `price` | Price in LKR (target variable) |
  | `condition` | Used / Brand New |
  | `storage` | Storage capacity |
  | `ram` | RAM capacity |
  | `location` | Seller's location |

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost Regressor |
| Tuning | RandomizedSearchCV (50 iterations, 3-fold CV) |
| Early Stopping | Yes (patience = 30 rounds) |
| Train/Val/Test Split | 70% / 15% / 15% |
| Random State | 42 |

---

## Evaluation Metrics

After training, the following metrics are computed on the held-out test set:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R2 Score** — Coefficient of Determination

Results are saved to `outputs/metrics.json` and `outputs/metrics_table.csv`.

---

## SHAP Explainability

This project uses **SHAP TreeExplainer** to explain model predictions:

1. **Summary Plot** — Which features most influence predictions overall
2. **Dependence Plot** — How the most important feature affects price
3. **Feature Importance** — Ranked bar chart of feature impacts

All plots are saved in `outputs/plots/` and displayed in the Streamlit app.

---

## Technologies Used

- Python 3.10+
- XGBoost
- scikit-learn
- SHAP
- Streamlit
- BeautifulSoup4
- Pandas, NumPy, Matplotlib, Seaborn

---

## License

This project is for educational and research purposes only.
