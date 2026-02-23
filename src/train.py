"""
Model Training — XGBoost Regressor
====================================
Sri Lanka Mobile Phone Price Prediction

Algorithm: XGBoost Regressor (MANDATORY)
Tuning:   RandomizedSearchCV

Why RandomizedSearchCV instead of GridSearchCV?
    - GridSearchCV exhaustively searches ALL combinations in the hyperparameter
      grid, which is computationally expensive.  With 6+ hyperparameters each
      having 4-5 possible values, the grid has thousands of combinations.
    - RandomizedSearchCV samples a fixed number of random combinations from the
      grid, which is significantly faster while still finding near-optimal
      parameters.  Research shows it reaches comparable performance using only
      a fraction of the evaluations (Bergstra & Bengio, 2012).
    - For this project with a moderate dataset (~5,000 rows), RandomizedSearchCV
      provides a good balance between search quality and training time.

Usage:
    python src/train.py --data data/processed.csv
"""

import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    setup_logger,
    ensure_dirs,
    save_model,
    DATA_DIR,
    MODELS_DIR,
)

warnings.filterwarnings("ignore")
logger = setup_logger("train")

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 1000),       # Number of boosting rounds
    "max_depth": randint(3, 12),              # Tree depth
    "learning_rate": uniform(0.01, 0.29),     # Step size shrinkage (eta)
    "subsample": uniform(0.6, 0.4),           # Row sampling per tree
    "colsample_bytree": uniform(0.5, 0.5),    # Column sampling per tree
    "min_child_weight": randint(1, 10),       # Min sum of instance weight in child
    "gamma": uniform(0, 0.5),                 # Min loss reduction for split
    "reg_alpha": uniform(0, 1.0),             # L1 regularization
    "reg_lambda": uniform(0.5, 1.5),          # L2 regularization
}

N_ITER = 50   # Number of random parameter combinations to try
CV_FOLDS = 3  # Cross-validation folds during search


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_split_data():
    """Load the train/val/test arrays saved by preprocess.py."""
    split_path = os.path.join(DATA_DIR, "train_val_test_split.npz")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split data not found at {split_path}. Run preprocess.py first."
        )
    data = np.load(split_path)
    return (
        data["X_train"], data["X_val"], data["X_test"],
        data["y_train"], data["y_val"], data["y_test"],
    )


def load_feature_meta():
    """Load the feature metadata saved by preprocess.py."""
    meta_path = os.path.join(MODELS_DIR, "feature_meta.json")
    with open(meta_path, "r") as f:
        return json.load(f)


def train_model():
    """
    Train an XGBoost Regressor with RandomizedSearchCV tuning
    and early stopping on the validation set.
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING — XGBoost Regressor")
    logger.info("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_split_data()
    meta = load_feature_meta()
    feature_cols = meta["feature_cols"]

    logger.info(f"Training set:   {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set:       {X_test.shape}")
    logger.info(f"Features:       {feature_cols}")

    # --- Phase 1: Hyperparameter search with RandomizedSearchCV ---
    logger.info("-" * 60)
    logger.info("Phase 1: Hyperparameter tuning (RandomizedSearchCV)")
    logger.info(f"  Iterations: {N_ITER} | CV folds: {CV_FOLDS}")

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",       # Fast histogram-based method
        random_state=42,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=CV_FOLDS,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    search_time = time.time() - start_time

    best_params = search.best_params_
    logger.info(f"Search completed in {search_time:.1f}s")
    logger.info(f"Best CV RMSE: {-search.best_score_:,.0f}")
    logger.info("Best hyperparameters:")
    for k, v in sorted(best_params.items()):
        logger.info(f"  {k}: {v}")

    # --- Phase 2: Retrain with best params + early stopping ---
    logger.info("-" * 60)
    logger.info("Phase 2: Final training with early stopping")

    final_model = xgb.XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        verbosity=1,
        early_stopping_rounds=30,
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    # Print final training info
    best_iteration = final_model.best_iteration
    best_score = final_model.best_score
    logger.info(f"Best iteration: {best_iteration}")
    logger.info(f"Best validation RMSE: {best_score:,.2f}")

    # --- Save model ---
    model_path = save_model(final_model, "xgboost_model.joblib")

    # Save best hyperparameters
    params_to_save = {k: float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v for k, v in best_params.items()}
    params_to_save["best_iteration"] = int(best_iteration)
    params_to_save["best_val_score"] = float(best_score)
    params_to_save["search_time_seconds"] = round(search_time, 1)

    params_path = os.path.join(MODELS_DIR, "best_hyperparameters.json")
    with open(params_path, "w") as f:
        json.dump(params_to_save, f, indent=2)
    logger.info(f"Saved hyperparameters: {params_path}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Model saved to: {model_path}")
    logger.info(f"  Best params saved to: {params_path}")
    logger.info("=" * 60)

    return final_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--data", default="data/processed.csv", help="Processed CSV (unused, reads .npz)")
    args = parser.parse_args()
    train_model()


if __name__ == "__main__":
    main()
