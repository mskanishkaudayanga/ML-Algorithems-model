"""
Data Validation & Cleaning
============================
Validates the scraped CSV data, removes duplicates, and produces
a clean dataset ready for analysis or ML model training.

Usage:
    python src/2_data_cleaning.py
    python src/2_data_cleaning.py --input data/mobile_phones.csv --output data/mobile_phones_clean.csv
"""

import os
import re
import argparse
import logging
from datetime import datetime

import pandas as pd

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(csv_path):
    """Load the scraped CSV data."""
    logger.info(f"📂 Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"   Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def remove_duplicates(df):
    """Remove duplicate records based on ad_url."""
    before = len(df)
    df = df.drop_duplicates(subset=["ad_url"], keep="first")
    after = len(df)
    removed = before - after
    if removed > 0:
        logger.info(f"🔄 Removed {removed} duplicate records ({before} → {after})")
    else:
        logger.info("✅ No duplicates found")
    return df


def clean_prices(df):
    """Clean and validate price data."""
    logger.info("💰 Cleaning price data...")

    # Convert price to numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Remove unrealistic prices (too low or too high)
    min_price = 500      # Minimum realistic phone price in LKR
    max_price = 2_000_000  # Maximum realistic phone price in LKR

    before = len(df)
    invalid_prices = df[(df["price"] < min_price) | (df["price"] > max_price)]
    if len(invalid_prices) > 0:
        logger.info(f"   ⚠️ Found {len(invalid_prices)} records with unrealistic prices")
        # Don't remove them, just flag them
        df["price_valid"] = df["price"].between(min_price, max_price)
    else:
        df["price_valid"] = True

    null_prices = df["price"].isna().sum()
    logger.info(f"   Null prices: {null_prices} | Valid prices: {df['price_valid'].sum()}")

    return df


def standardize_brands(df):
    """Standardize brand names."""
    logger.info("🏷️ Standardizing brand names...")

    # Brand name corrections
    brand_map = {
        "Redmi": "Xiaomi",
        "Poco": "Xiaomi",
        "Mi": "Xiaomi",
        "iphone": "Apple",
        "Iphone": "Apple",
        "APPLE": "Apple",
        "SAMSUNG": "Samsung",
        "XIAOMI": "Xiaomi",
        "GOOGLE": "Google",
        "VIVO": "Vivo",
        "OPPO": "Oppo",
        "REALME": "Realme",
        "NOKIA": "Nokia",
        "HUAWEI": "Huawei",
        "INFINIX": "Infinix",
        "TECNO": "Tecno",
    }

    df["brand"] = df["brand"].replace(brand_map)

    # Report brand distribution
    brand_counts = df["brand"].value_counts().head(15)
    logger.info(f"   Top brands:\n{brand_counts.to_string()}")

    return df


def standardize_conditions(df):
    """Standardize condition values."""
    logger.info("📋 Standardizing condition values...")

    condition_map = {
        "brand new": "Brand New",
        "Brand New": "Brand New",
        "BRAND NEW": "Brand New",
        "new": "Brand New",
        "New": "Brand New",
        "used": "Used",
        "Used": "Used",
        "USED": "Used",
        "Unknown": "Unknown",
    }

    df["condition"] = df["condition"].map(
        lambda x: condition_map.get(str(x).strip(), str(x).strip())
    )

    condition_counts = df["condition"].value_counts()
    logger.info(f"   Conditions:\n{condition_counts.to_string()}")

    return df


def standardize_locations(df):
    """Standardize location names."""
    logger.info("📍 Standardizing locations...")

    # Fix common location issues
    df["location"] = df["location"].str.strip()
    df["location"] = df["location"].replace("", "Unknown")
    df["location"] = df["location"].fillna("Unknown")

    location_counts = df["location"].value_counts().head(10)
    logger.info(f"   Top locations:\n{location_counts.to_string()}")

    return df


def add_derived_features(df):
    """Add useful derived features."""
    logger.info("🔧 Adding derived features...")

    # Price range categories
    bins = [0, 10000, 25000, 50000, 100000, 200000, 500000, float("inf")]
    labels = ["<10K", "10K-25K", "25K-50K", "50K-100K", "100K-200K", "200K-500K", "500K+"]
    df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels, right=False)

    # Storage as numeric (GB)
    df["storage_gb"] = df["storage"].str.extract(r"(\d+)", expand=False).astype(float)

    # RAM as numeric (GB)
    df["ram_gb"] = df["ram"].str.extract(r"(\d+)", expand=False).astype(float)

    # Title length (can indicate ad quality)
    df["title_length"] = df["title"].str.len()

    logger.info("   Added: price_range, storage_gb, ram_gb, title_length")

    return df


def generate_report(df, output_dir):
    """Generate a data quality report."""
    report_path = os.path.join(output_dir, "data_quality_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("📊 DATA QUALITY REPORT — ikman.lk Mobile Phones\n")
        f.write(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Unique Ad URLs: {df['ad_url'].nunique()}\n\n")

        f.write("─── Column Completeness ───\n")
        for col in df.columns:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            f.write(f"  {col:25s} : {non_null:6d} / {len(df):6d} ({pct:5.1f}%)\n")

        f.write("\n─── Price Statistics ───\n")
        price_stats = df["price"].describe()
        for stat, value in price_stats.items():
            f.write(f"  {stat:10s} : Rs {value:>12,.0f}\n")

        f.write("\n─── Brand Distribution ───\n")
        brand_counts = df["brand"].value_counts().head(20)
        for brand, count in brand_counts.items():
            pct = (count / len(df)) * 100
            f.write(f"  {brand:20s} : {count:5d} ({pct:5.1f}%)\n")

        f.write("\n─── Condition Distribution ───\n")
        cond_counts = df["condition"].value_counts()
        for cond, count in cond_counts.items():
            pct = (count / len(df)) * 100
            f.write(f"  {cond:20s} : {count:5d} ({pct:5.1f}%)\n")

        f.write("\n─── Location Distribution (Top 15) ───\n")
        loc_counts = df["location"].value_counts().head(15)
        for loc, count in loc_counts.items():
            pct = (count / len(df)) * 100
            f.write(f"  {loc:20s} : {count:5d} ({pct:5.1f}%)\n")

        f.write("\n─── Storage Distribution ───\n")
        if "storage" in df.columns:
            storage_counts = df["storage"].value_counts().head(10)
            for storage, count in storage_counts.items():
                f.write(f"  {str(storage):15s} : {count:5d}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Report complete.\n")

    logger.info(f"📄 Data quality report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Clean and validate scraped mobile phone data")
    parser.add_argument(
        "--input", "-i",
        default="data/mobile_phones.csv",
        help="Input CSV file (default: data/mobile_phones.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/mobile_phones_clean.csv",
        help="Output CSV file (default: data/mobile_phones_clean.csv)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"❌ Input file not found: {args.input}")
        return

    # Load
    df = load_data(args.input)

    # Clean
    df = remove_duplicates(df)
    df = clean_prices(df)
    df = standardize_brands(df)
    df = standardize_conditions(df)
    df = standardize_locations(df)
    df = add_derived_features(df)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    logger.info(f"💾 Clean data saved to {args.output} ({len(df)} records)")

    # Report
    generate_report(df, os.path.dirname(args.output))

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ Data Cleaning Complete!")
    logger.info(f"   Input:  {args.input} → {args.output}")
    logger.info(f"   Records: {len(df)}")
    logger.info(f"   Columns: {len(df.columns)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
