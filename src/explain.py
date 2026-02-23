"""
Model Explainability — SHAP Analysis
=======================================
Sri Lanka Mobile Phone Price Prediction

Uses SHAP (SHapley Additive exPlanations) TreeExplainer for XGBoost.

Generates:
    1. SHAP Summary Plot
       - Shows which features most influence the prediction.
       - Each dot is a single prediction.  The x-axis shows the SHAP value
         (how much that feature pushed the prediction higher or lower).
         Colour indicates the actual feature value (red = high, blue = low).

    2. SHAP Dependence Plot (for the most important feature)
       - Shows how ONE specific feature's value affects the prediction.
       - X-axis = actual feature value; Y-axis = its SHAP value.
         Helps identify non-linear relationships and interactions.

    3. Feature Importance Bar Chart
       - Ranks features by their average absolute SHAP value.
       - Higher bar = the feature has a bigger overall impact on predictions.
         This is more reliable than built-in XGBoost feature importance
         because SHAP accounts for interactions between features.

Usage:
    python src/explain.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    setup_logger,
    ensure_dirs,
    load_model,
    DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
)

warnings.filterwarnings("ignore")
logger = setup_logger("explain")


# ---------------------------------------------------------------------------
# Human-readable feature name mapping
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "brand_clean_encoded": "Phone Brand",
    "condition_encoded": "Condition (New / Used)",
    "location_clean_encoded": "Seller Location",
    "phone_model_clean_encoded": "Phone Model",
    "storage_gb": "Storage Capacity (GB)",
    "ram_gb": "RAM (GB)",
    "is_member_flag": "Verified Seller",
}


def get_readable_name(col: str) -> str:
    """Return a human-readable label for a feature column."""
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_explainability():
    """Generate all SHAP-based explanations and feature importance plots."""
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("SHAP EXPLAINABILITY ANALYSIS")
    logger.info("=" * 60)

    # Load model and data
    model = load_model("xgboost_model.joblib")
    logger.info("Loaded trained XGBoost model")

    split = np.load(os.path.join(DATA_DIR, "train_val_test_split.npz"))
    X_test = split["X_test"]

    # Load feature names
    meta_path = os.path.join(MODELS_DIR, "feature_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    # Create DataFrame with readable column names for plots
    readable_names = [get_readable_name(c) for c in feature_cols]
    X_df = pd.DataFrame(X_test, columns=readable_names)

    logger.info(f"Test samples: {X_test.shape[0]}")
    logger.info(f"Features: {readable_names}")

    # --- SHAP TreeExplainer (optimal for tree-based models like XGBoost) ---
    logger.info("Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    logger.info("SHAP values computed successfully")

    # Save explainer and SHAP values for the Streamlit app
    import joblib
    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.joblib"))
    np.save(os.path.join(MODELS_DIR, "shap_values_test.npy"), shap_values)
    logger.info("Saved SHAP explainer and values")

    # --- 1. SHAP Summary Plot ---
    # What it means:
    #   Each row is a feature.  Each dot is one test sample.
    #   Dots to the RIGHT mean that feature INCREASED the predicted price.
    #   Dots to the LEFT mean that feature DECREASED the predicted price.
    #   Red colour = high feature value, Blue = low feature value.
    #   Features are sorted by overall importance (top = most important).
    generate_summary_plot(shap_values, X_df)

    # --- 2. SHAP Dependence Plot (most important feature) ---
    # What it means:
    #   Shows the relationship between ONE feature and the model's predictions.
    #   X-axis = the feature's actual value.
    #   Y-axis = how much that feature contributed to the prediction (SHAP value).
    #   Colour shows interaction with another correlated feature.
    #   Helps spot non-linear or threshold effects.
    most_important_idx = np.abs(shap_values).mean(axis=0).argmax()
    most_important_feature = readable_names[most_important_idx]
    generate_dependence_plot(shap_values, X_df, most_important_idx, most_important_feature)

    # --- 3. Feature Importance Bar Chart ---
    # What it means:
    #   Bars show the average absolute SHAP value for each feature.
    #   Longer bar = bigger impact on price predictions overall.
    #   This is model-agnostic importance: it reflects how much each
    #   feature actually shifts the prediction, accounting for interactions.
    generate_importance_bar_chart(shap_values, readable_names)

    logger.info("=" * 60)
    logger.info("EXPLAINABILITY ANALYSIS COMPLETE")
    logger.info(f"All plots saved to: {PLOTS_DIR}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Plot 1: SHAP Summary
# ---------------------------------------------------------------------------

def generate_summary_plot(shap_values, X_df):
    """Generate and save the SHAP summary (beeswarm) plot."""
    logger.info("Generating SHAP summary plot...")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_df,
        show=False,
        plot_size=(12, 8),
    )
    plt.title(
        "SHAP Summary — Feature Impact on Mobile Phone Price Prediction",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("SHAP Value (Impact on Predicted Price)", fontsize=12)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: SHAP Dependence
# ---------------------------------------------------------------------------

def generate_dependence_plot(shap_values, X_df, feature_idx, feature_name):
    """Generate SHAP dependence plot for the most important feature."""
    logger.info(f"Generating SHAP dependence plot for: {feature_name}")

    plt.figure(figsize=(10, 7))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_df,
        show=False,
    )
    plt.title(
        f"SHAP Dependence Plot — How {feature_name} Affects Predicted Price",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel(f"{feature_name}", fontsize=12)
    plt.ylabel("SHAP Value (Impact on Price)", fontsize=12)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "shap_dependence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Feature Importance Bar Chart
# ---------------------------------------------------------------------------

def generate_importance_bar_chart(shap_values, feature_names):
    """Generate a bar chart ranking features by mean absolute SHAP value."""
    logger.info("Generating feature importance bar chart...")

    # Calculate mean |SHAP| for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_values = mean_abs_shap[sorted_idx]

    # Professional colour gradient
    n = len(sorted_names)
    colours = plt.cm.Blues(np.linspace(0.9, 0.4, n))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(n), sorted_values[::-1],
        color=colours[::-1],
        edgecolor="#1E3A5F",
        linewidth=0.5,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names[::-1], fontsize=11)
    ax.set_xlabel("Mean Absolute SHAP Value (Average Impact on Price)", fontsize=12)
    ax.set_title(
        "Feature Importance — Which Features Most Affect Price Predictions",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_values[::-1])):
        ax.text(
            bar.get_width() + max(sorted_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center", fontsize=10, color="#1E3A5F",
        )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    # Also save importance as JSON for the Streamlit app
    importance_dict = {name: float(val) for name, val in zip(sorted_names, sorted_values)}
    import json
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w") as f:
        json.dump(importance_dict, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    run_explainability()


if __name__ == "__main__":
    main()
