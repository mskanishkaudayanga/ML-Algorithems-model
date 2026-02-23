"""Build phone specs mapping from RAW data (before scaling)."""
import pandas as pd
import numpy as np
import json
import re

df = pd.read_csv("data/mobile_phones.csv")

# Extract numeric storage/RAM from raw strings
def extract_gb(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else np.nan

df["storage_num"] = df["storage"].apply(extract_gb)
df["ram_num"] = df["ram"].apply(extract_gb)

# Clean brand
brand_counts = df["brand"].value_counts()
rare_brands = brand_counts[brand_counts < 20].index.tolist()
df["brand_clean"] = df["brand"].apply(lambda x: x if x not in rare_brands else "Other")

# Clean phone model
def extract_phone_model(title):
    if pd.isna(title):
        return "Unknown"
    clean = re.sub(r"\s*\((Used|Brand New|used|brand new)\)\s*", "", str(title))
    clean = re.sub(r"\d+\s*GB\s*/?\s*\d*\s*GB?", "", clean)
    clean = re.sub(r"\d+\s*GB", "", clean)
    clean = re.sub(r"\d+\s*TB", "", clean)
    clean = re.sub(r"\|", " ", clean)
    clean = re.sub(r"Ram\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"Full\s*Set\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"5G\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"4k\b.*", "", clean, flags=re.IGNORECASE)
    brands = ["Apple","Samsung","Xiaomi","Google","Vivo","Oppo","Realme","OnePlus",
              "Huawei","Nokia","Sony","Infinix","Tecno","Honor","Motorola","ZTE",
              "Nothing","Poco","LG","HTC","Redmi"]
    for b in brands:
        clean = re.sub(rf"^{b}\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"[^\w\s]+$", "", clean).strip()
    return clean if clean else "Unknown"

df["phone_model"] = df["title"].apply(extract_phone_model)
model_counts = df["phone_model"].value_counts()
rare_models = model_counts[model_counts < 5].index.tolist()
df["phone_model_clean"] = df["phone_model"].apply(lambda x: x if x not in rare_models else "Other")

# Build specs mapping with RAW GB values
specs = (
    df.groupby(["brand_clean", "phone_model_clean"])
    .agg(
        storage_median=("storage_num", "median"),
        ram_median=("ram_num", "median"),
        count=("price", "count"),
        has_storage_pct=("storage_num", lambda x: x.notna().mean()),
        has_ram_pct=("ram_num", lambda x: x.notna().mean()),
    )
    .reset_index()
)
specs = specs[specs["count"] >= 3]

result = {}
for _, r in specs.iterrows():
    key = f"{r['brand_clean']}||{r['phone_model_clean']}"
    storage = float(r["storage_median"]) if pd.notna(r["storage_median"]) else None
    ram = float(r["ram_median"]) if pd.notna(r["ram_median"]) else None
    has_storage = bool(r["has_storage_pct"] > 0.3)
    has_ram = bool(r["has_ram_pct"] > 0.3)
    result[key] = {
        "storage": storage,
        "ram": ram,
        "has_storage": has_storage,
        "has_ram": has_ram,
    }

with open("models/phone_specs_mapping.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Saved raw specs for {len(result)} phone models")
print()
print("=== Phones WITHOUT storage/RAM ===")
for key, val in sorted(result.items()):
    if not val["has_storage"] or not val["has_ram"]:
        brand, model = key.split("||")
        print(f"  {brand:12s} {model:25s} storage={str(val['storage']):>6s}  ram={str(val['ram']):>6s}  has_storage={val['has_storage']}  has_ram={val['has_ram']}")

print()
print("=== Nokia models ===")
for key, val in sorted(result.items()):
    if key.startswith("Nokia"):
        brand, model = key.split("||")
        print(f"  {model:25s} storage={val['storage']}  ram={val['ram']}  has_storage={val['has_storage']}  has_ram={val['has_ram']}")
