"""
Model Evaluation
==================
Sri Lanka Mobile Phone Price Prediction

Metrics:
    - RMSE (Root Mean Squared Error)
    - MAE  (Mean Absolute Error)
    - R2   (R-squared Score)

Outputs:
    - outputs/metrics.json
    - outputs/metrics_table.csv
    - outputs/plots/predicted_vs_actual.png
    - outputs/plots/residual_histogram.png

Usage:
    python src/evaluate.py
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
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    setup_logger,
    ensure_dirs,
    load_model,
    save_metrics,
    DATA_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
)

warnings.filterwarnings("ignore")
logger = setup_logger("evaluate")

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "font.family": "sans-serif",
})


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model():
    """Evaluate the trained XGBoost model on the test set."""
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    # Load model
    model = load_model("xgboost_model.joblib")
    logger.info("Loaded trained XGBoost model")

    # Load test data
    split_path = os.path.join(DATA_DIR, "train_val_test_split.npz")
    data = np.load(split_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Predictions
    y_pred = model.predict(X_test)

    # --- Calculate metrics ---
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2_Score": round(r2, 4),
        "Test_Samples": int(X_test.shape[0]),
    }

    logger.info("-" * 40)
    logger.info(f"RMSE      : Rs {rmse:>12,.2f}")
    logger.info(f"MAE       : Rs {mae:>12,.2f}")
    logger.info(f"R2 Score  :    {r2:>12.4f}")
    logger.info("-" * 40)

    # --- Save metrics ---
    save_metrics(metrics)

    # --- Generate plots ---
    plot_predicted_vs_actual(y_test, y_pred, r2)
    plot_residual_histogram(y_test, y_pred)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# Plot: Predicted vs Actual
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(y_true, y_pred, r2):
    """
    Generate a scatter plot of Predicted vs Actual mobile phone prices.

    All labels are HUMAN READABLE — no variable names.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter
    scatter = ax.scatter(
        y_true, y_pred,
        alpha=0.35,
        s=20,
        c="#2563EB",
        edgecolors="none",
        label="Predicted Prices",
    )

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        color="#DC2626",
        linewidth=2,
        linestyle="--",
        label="Perfect Prediction Line",
    )

    ax.set_xlabel("Actual Mobile Phone Price (LKR)", fontsize=14)
    ax.set_ylabel("Predicted Mobile Phone Price (LKR)", fontsize=14)
    ax.set_title(
        "Predicted Mobile Phone Price vs Actual Mobile Phone Price",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Add R2 annotation
    ax.text(
        0.05, 0.92,
        f"R-squared Score = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F9FF", edgecolor="#93C5FD"),
    )

    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Format tick labels with commas
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs {x:,.0f}"))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs {x:,.0f}"))

    path = os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot: Residual Histogram
# ---------------------------------------------------------------------------

def plot_residual_histogram(y_true, y_pred):
    """
    Generate a histogram of prediction residuals (errors).

    Residual = Actual Price - Predicted Price
    A good model should have residuals centered around zero.
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.hist(
        residuals,
        bins=60,
        color="#7C3AED",
        edgecolor="#4C1D95",
        alpha=0.75,
        label="Prediction Residuals",
    )

    # Add vertical line at zero
    ax.axvline(
        x=0, color="#DC2626", linewidth=2, linestyle="--",
        label="Zero Error Line",
    )

    # Add mean residual line
    mean_residual = residuals.mean()
    ax.axvline(
        x=mean_residual, color="#059669", linewidth=2, linestyle="-.",
        label=f"Mean Residual (Rs {mean_residual:,.0f})",
    )

    ax.set_xlabel("Prediction Error (Actual Price - Predicted Price) in LKR", fontsize=13)
    ax.set_ylabel("Number of Predictions", fontsize=13)
    ax.set_title(
        "Distribution of Prediction Errors (Residuals)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # Format x-axis ticks
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs {x:,.0f}"))

    # Add statistics text box
    stats_text = (
        f"Mean Error: Rs {mean_residual:,.0f}\n"
        f"Std Dev: Rs {residuals.std():,.0f}\n"
        f"Median Error: Rs {np.median(residuals):,.0f}"
    )
    ax.text(
        0.02, 0.92, stats_text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F3FF", edgecolor="#C4B5FD"),
    )

    path = os.path.join(PLOTS_DIR, "residual_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    evaluate_model()


if __name__ == "__main__":
    main()
