"""
Utility Functions for Sri Lanka Mobile Phone Price Prediction
=============================================================
Shared helper functions used across the project pipeline.
"""

import os
import re
import json
import logging
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
APP_DIR = os.path.join(PROJECT_ROOT, "app")


def ensure_dirs():
    """Create all required project directories."""
    for d in [MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Price cleaning
# ---------------------------------------------------------------------------

def clean_price(value) -> float:
    """
    Convert a raw price string to a numeric float.

    Handles formats commonly found on ikman.lk:
        - "Rs 119,999"
        - "Rs 1.2 Lakh"  (1 Lakh = 100,000)
        - "Rs 1.5 Mn"    (1 Mn  = 1,000,000)
        - Already-numeric values
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    # Handle Lakh (1 Lakh = 100,000 LKR)
    lakh_match = re.search(r"([\d,.]+)\s*[Ll]akh", text)
    if lakh_match:
        num = float(lakh_match.group(1).replace(",", ""))
        return num * 100_000

    # Handle Mn / Million (1 Mn = 1,000,000 LKR)
    mn_match = re.search(r"([\d,.]+)\s*[Mm]n", text)
    if mn_match:
        num = float(mn_match.group(1).replace(",", ""))
        return num * 1_000_000

    # Strip "Rs", commas, spaces and convert
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
    try:
        return float(cleaned) if cleaned else np.nan
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Target column detection
# ---------------------------------------------------------------------------

def detect_target_column(df: pd.DataFrame, preferred="price") -> str:
    """
    Detect the target (price) column in the dataframe.

    Strategy:
        1. Use the preferred name if it exists.
        2. Look for columns containing 'price' (case-insensitive).
        3. Raise an error if nothing found.
    """
    if preferred in df.columns:
        print(f"[INFO] Target column detected: '{preferred}'")
        return preferred

    for col in df.columns:
        if "price" in col.lower():
            print(f"[INFO] Target column auto-detected: '{col}' (preferred '{preferred}' not found)")
            return col

    raise ValueError(
        f"Could not detect a price/target column. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Storage / RAM extraction
# ---------------------------------------------------------------------------

def extract_storage_gb(value) -> float:
    """Convert storage string like '128GB' to numeric GB."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def extract_ram_gb(value) -> float:
    """Convert RAM string like '8GB' to numeric GB."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


# ---------------------------------------------------------------------------
# Phone model extraction
# ---------------------------------------------------------------------------

def extract_phone_model(title: str) -> str:
    """
    Extract a clean phone model name from the ad title.

    Examples:
        'Apple iPhone 14 Pro 256GB (Used)' -> 'iPhone 14 Pro'
        'Samsung Galaxy S24 Ultra 12GB | 512GB (Brand New)' -> 'Galaxy S24 Ultra'
    """
    if pd.isna(title):
        return "Unknown"

    # Remove condition tags
    clean = re.sub(r"\s*\((Used|Brand New|used|brand new)\)\s*", "", str(title))

    # Remove storage/RAM specs
    clean = re.sub(r"\d+\s*GB\s*/?\s*\d*\s*GB?", "", clean)
    clean = re.sub(r"\d+\s*GB", "", clean)
    clean = re.sub(r"\d+\s*TB", "", clean)

    # Remove extra descriptors
    clean = re.sub(r"\|", " ", clean)
    clean = re.sub(r"Ram\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"Full\s*Set\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"5G\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"4k\b.*", "", clean, flags=re.IGNORECASE)

    # Remove brand prefix to get just the model
    brands = [
        "Apple", "Samsung", "Xiaomi", "Google", "Vivo", "Oppo",
        "Realme", "OnePlus", "Huawei", "Nokia", "Sony", "Infinix",
        "Tecno", "Honor", "Motorola", "ZTE", "Nothing", "Poco",
        "LG", "HTC", "Redmi",
    ]
    for brand in brands:
        clean = re.sub(rf"^{brand}\s+", "", clean, flags=re.IGNORECASE)

    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    # Remove trailing special characters
    clean = re.sub(r"[^\w\s]+$", "", clean).strip()

    return clean if clean else "Unknown"


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_model(model, filename: str):
    """Save a model or encoder to the models/ directory."""
    ensure_dirs()
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"[SAVED] {path}")
    return path


def load_model(filename: str):
    """Load a model or encoder from the models/ directory."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def save_metrics(metrics: dict, json_path: str = None, csv_path: str = None):
    """Save evaluation metrics to JSON and/or CSV."""
    ensure_dirs()
    if json_path is None:
        json_path = os.path.join(OUTPUTS_DIR, "metrics.json")
    if csv_path is None:
        csv_path = os.path.join(OUTPUTS_DIR, "metrics_table.csv")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {json_path}")

    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")
