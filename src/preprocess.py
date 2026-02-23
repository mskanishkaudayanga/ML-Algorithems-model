"""
Data Preprocessing Pipeline
============================
Sri Lanka Mobile Phone Price Prediction

Steps:
    1. Load raw CSV data
    2. Clean currency values (Rs, commas, Lakh, Mn -> numeric)
    3. Handle missing values
    4. Feature engineering (extract phone model, normalize, encode)
    5. Train / Validation / Test split (70 / 15 / 15)
    6. Save processed datasets and encoders

Usage:
    python src/preprocess.py --input data/mobile_phones.csv --output data/processed.csv
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    setup_logger,
    ensure_dirs,
    detect_target_column,
    clean_price,
    extract_storage_gb,
    extract_ram_gb,
    extract_phone_model,
    save_model,
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
)

warnings.filterwarnings("ignore")
logger = setup_logger("preprocess")

# ---------------------------------------------------------------------------
# Step 1: Load Data
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    logger.info(f"Loading data from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Clean Price Column
# ---------------------------------------------------------------------------

def clean_prices(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Clean and normalize the price column.
    Handles: Rs, commas, Lakh (x100,000), Mn (x1,000,000).
    """
    logger.info("Cleaning price column...")

    # If price_raw exists, use it as the source for cleaning
    if "price_raw" in df.columns:
        df[target_col] = df["price_raw"].apply(clean_price)
        # Fallback to existing numeric price if raw cleaning failed
        if "price" in df.columns and target_col != "price":
            mask = df[target_col].isna()
            df.loc[mask, target_col] = pd.to_numeric(df.loc[mask, "price"], errors="coerce")
    else:
        df[target_col] = df[target_col].apply(clean_price)

    # Ensure numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Remove unrealistic prices
    before = len(df)
    df = df[(df[target_col] >= 500) & (df[target_col] <= 2_000_000)].copy()
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} rows with unrealistic prices (< Rs 500 or > Rs 2,000,000)")

    # Remove rows with missing price
    null_prices = df[target_col].isna().sum()
    if null_prices > 0:
        df = df.dropna(subset=[target_col]).copy()
        logger.info(f"Dropped {null_prices} rows with missing prices")

    logger.info(f"Price range: Rs {df[target_col].min():,.0f} - Rs {df[target_col].max():,.0f}")
    logger.info(f"Mean price: Rs {df[target_col].mean():,.0f}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Handle Missing Values
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop missing values based on column type."""
    logger.info("Handling missing values...")
    logger.info(f"Missing values before:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

    # For categorical columns: fill with 'Unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Unknown")

    # For numeric columns: fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Filled '{col}' nulls with median = {median_val:.1f}")

    remaining = df.isnull().sum().sum()
    logger.info(f"Remaining missing values: {remaining}")
    return df


# ---------------------------------------------------------------------------
# Step 4: Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create and transform features for model training.

    Key transformations:
        - Extract clean phone model name from title
        - Convert storage/RAM to numeric GB
        - Create price category bins (for analysis)
        - Map brand frequency for rare-brand handling
    """
    logger.info("Engineering features...")

    # --- Extract phone model from title (CRITICAL) ---
    if "title" in df.columns:
        df["phone_model"] = df["title"].apply(extract_phone_model)
        n_models = df["phone_model"].nunique()
        logger.info(f"Extracted {n_models} unique phone models from titles")
    else:
        df["phone_model"] = "Unknown"
        logger.warning("No 'title' column found; phone_model set to 'Unknown'")

    # --- Storage (GB) ---
    if "storage" in df.columns:
        df["storage_gb"] = df["storage"].apply(extract_storage_gb)
    elif "storage_gb" not in df.columns:
        df["storage_gb"] = np.nan

    # --- RAM (GB) ---
    if "ram" in df.columns:
        df["ram_gb"] = df["ram"].apply(extract_ram_gb)
    elif "ram_gb" not in df.columns:
        df["ram_gb"] = np.nan

    # --- Fill storage/RAM with median ---
    for col in ["storage_gb", "ram_gb"]:
        if df[col].isna().sum() > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.info(f"  '{col}' nulls filled with median = {median:.0f}")

    # --- Brand consolidation: group rare brands into 'Other' ---
    if "brand" in df.columns:
        brand_counts = df["brand"].value_counts()
        rare_brands = brand_counts[brand_counts < 20].index.tolist()
        df["brand_clean"] = df["brand"].apply(lambda x: x if x not in rare_brands else "Other")
        logger.info(f"Consolidated {len(rare_brands)} rare brands into 'Other'")
    else:
        df["brand_clean"] = "Unknown"

    # --- Condition ---
    if "condition" in df.columns:
        df["condition"] = df["condition"].replace({"Unknown": "Used"})

    # --- Location consolidation: group rare locations into 'Other' ---
    if "location" in df.columns:
        loc_counts = df["location"].value_counts()
        rare_locs = loc_counts[loc_counts < 30].index.tolist()
        df["location_clean"] = df["location"].apply(lambda x: x if x not in rare_locs else "Other")
    else:
        df["location_clean"] = "Unknown"

    # --- Phone model consolidation: group rare models into 'Other' ---
    model_counts = df["phone_model"].value_counts()
    rare_models = model_counts[model_counts < 5].index.tolist()
    df["phone_model_clean"] = df["phone_model"].apply(
        lambda x: x if x not in rare_models else "Other"
    )
    logger.info(f"Consolidated {len(rare_models)} rare phone models into 'Other'")

    # --- is_member flag ---
    if "is_member" in df.columns:
        df["is_member_flag"] = df["is_member"].astype(int)
    else:
        df["is_member_flag"] = 0

    return df


# ---------------------------------------------------------------------------
# Step 5: Encode & Normalize
# ---------------------------------------------------------------------------

def encode_and_normalize(df: pd.DataFrame, target_col: str):
    """
    Encode categorical features with LabelEncoder and
    normalize numeric features with StandardScaler.

    Returns:
        df_encoded: DataFrame ready for modelling
        feature_cols: list of feature column names
        encoders: dict of fitted LabelEncoders  (saved for inference)
        scaler: fitted StandardScaler             (saved for inference)
    """
    logger.info("Encoding categorical features and normalizing numerics...")

    # Select modelling columns
    feature_config = {
        "categorical": ["brand_clean", "condition", "location_clean", "phone_model_clean"],
        "numeric": ["storage_gb", "ram_gb", "is_member_flag"],
    }

    # Keep only columns that exist
    cat_cols = [c for c in feature_config["categorical"] if c in df.columns]
    num_cols = [c for c in feature_config["numeric"] if c in df.columns]

    # --- Label Encoding ---
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(f"  Encoded '{col}' -> {len(le.classes_)} classes")

    encoded_cat_cols = [c + "_encoded" for c in cat_cols]

    # --- Standard Scaling ---
    scaler = StandardScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    feature_cols = encoded_cat_cols + num_cols
    logger.info(f"Final feature columns ({len(feature_cols)}): {feature_cols}")

    return df, feature_cols, encoders, scaler


# ---------------------------------------------------------------------------
# Step 6: Train / Validation / Test Split
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Split into 70% train / 15% validation / 15% test.

    Uses random_state=42 for reproducibility.
    """
    logger.info("Splitting data: 70% train / 15% validation / 15% test ...")

    X = df[feature_cols].values
    y = df[target_col].values

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Second split: 50% of temp -> 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Validation: {X_val.shape[0]} samples")
    logger.info(f"  Test: {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(input_csv: str, output_csv: str):
    """Execute the full preprocessing pipeline."""
    ensure_dirs()

    # 1. Load
    df = load_data(input_csv)

    # 2. Detect target
    target_col = detect_target_column(df)

    # 3. Clean prices
    df = clean_prices(df, target_col)

    # 4. Missing values
    df = handle_missing_values(df)

    # 5. Feature engineering
    df = engineer_features(df)

    # 6. Encode & normalize
    df, feature_cols, encoders, scaler = encode_and_normalize(df, target_col)

    # 7. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_cols, target_col
    )

    # --- Save everything ---
    # Processed full dataframe
    output_path = os.path.join(DATA_DIR, output_csv) if not os.path.isabs(output_csv) else output_csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save only the relevant columns
    save_cols = feature_cols + [target_col, "title", "brand", "model", "condition",
                                 "location", "storage", "ram", "phone_model",
                                 "brand_clean", "location_clean", "phone_model_clean"]
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_csv(output_path, index=False)
    logger.info(f"Saved processed data: {output_path} ({len(df)} rows)")

    # Save split arrays
    np.savez(
        os.path.join(DATA_DIR, "train_val_test_split.npz"),
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )
    logger.info(f"Saved train/val/test split arrays to: {DATA_DIR}/train_val_test_split.npz")

    # Save feature column names
    import json
    meta = {"feature_cols": feature_cols, "target_col": target_col}
    meta_path = os.path.join(MODELS_DIR, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved feature metadata: {meta_path}")

    # Save encoders and scaler
    save_model(encoders, "label_encoders.joblib")
    save_model(scaler, "scaler.joblib")

    # Save the column mapping for the Streamlit app
    col_info = {
        "brand_classes": {col: list(le.classes_) for col, le in encoders.items()},
        "numeric_features": ["storage_gb", "ram_gb", "is_member_flag"],
    }
    col_info_path = os.path.join(MODELS_DIR, "column_info.json")
    with open(col_info_path, "w") as f:
        json.dump(col_info, f, indent=2)
    logger.info(f"Saved column info: {col_info_path}")

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"  Rows: {len(df)} | Features: {len(feature_cols)}")
    logger.info(f"  Feature columns: {feature_cols}")
    logger.info("=" * 60)

    return df, feature_cols, target_col


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess mobile phone data")
    parser.add_argument(
        "--input", "-i",
        default="data/mobile_phones.csv",
        help="Path to raw CSV file",
    )
    parser.add_argument(
        "--output", "-o",
        default="processed.csv",
        help="Output filename for processed data (saved in data/)",
    )
    args = parser.parse_args()
    run_preprocessing(args.input, args.output)


if __name__ == "__main__":
    main()
