# models/catboost_model.py

from catboost import CatBoostClassifier
import numpy as np
import joblib
import os

import config
from utils import set_seed, display_result

def build_catboost_model():
    """
    Builds and returns a CatBoost Classifier model.

    Returns:
        CatBoostClassifier: The CatBoost model.
    """
    set_seed(config.RANDOM_SEED)
    model_catboost = CatBoostClassifier(
        n_estimators=500,
        random_seed=config.RANDOM_SEED,
        learning_rate=0.25,
        max_depth=12,
        loss_function='MultiClass', # Specify loss function for multi-class
        verbose=0 # Set verbose to 0 to suppress training output by default
    )
    return model_catboost

def train_and_evaluate_catboost(X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw):
    """
    Trains, evaluates, and saves the CatBoost model.

    Args:
        X_train (np.ndarray): Training features.
        y_train_raw (np.ndarray): Raw integer training labels.
        X_val (np.ndarray): Validation features.
        y_val_raw (np.ndarray): Raw integer validation labels.
        X_test (np.ndarray): Test features.
        y_test_raw (np.ndarray): Raw integer test labels.
    """
    print("\n--- Training CatBoost Model ---")
    model_catboost = build_catboost_model()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.CATBOOST_MODEL_PATH), exist_ok=True)

    model_catboost.fit(
        X_train, y_train_raw,
        eval_set=(X_val, y_val_raw),
        verbose=1, # Set verbose to 1 for this function call
        early_stopping_rounds=10 # Example early stopping rounds
    )

    print("\n--- CatBoost Model Evaluation ---")
    print("---------------------Test Set--------------------------")
    y_pred_catboost = model_catboost.predict(X_test)
    # CatBoost predict returns 2D array for multiclass, need to flatten to 1D
    y_pred_catboost_labels = y_pred_catboost.flatten()
    display_result(y_test_raw, y_pred_catboost_labels)

    # Save the trained model
    joblib.dump(model_catboost, config.CATBOOST_MODEL_PATH)
    print(f"CatBoost model saved to {config.CATBOOST_MODEL_PATH}")

    # Load and test the saved model
    loaded_catboost_model = joblib.load(config.CATBOOST_MODEL_PATH)
    print("\n--- Loaded CatBoost Model Evaluation ---")
    y_pred_loaded_catboost = loaded_catboost_model.predict(X_test)
    y_pred_loaded_catboost_labels = y_pred_loaded_catboost.flatten()
    display_result(y_test_raw, y_pred_loaded_catboost_labels)