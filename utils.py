# utils.py

import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import os # Import os for os.environ

import config

def set_seed(seed=config.RANDOM_SEED):
    """
    Sets random seeds for reproducibility across TensorFlow, NumPy, and Python's random module.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For TensorFlow 2.x, set environment variable for deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def display_result(y_true, y_pred_labels):
    """
    Displays accuracy, precision, recall, and F1-score.
    Ensures y_true and y_pred_labels are 1D integer arrays for scikit-learn metrics.

    Args:
        y_true (array-like): True labels (can be 1D integer or 2D one-hot encoded).
        y_pred_labels (array-like): Predicted labels (expected to be 1D integer).
    """
    # Ensure y_true is a 1D numpy array of integer labels
    # If y_true is one-hot encoded (e.g., from Y_test_csv), convert it to integer labels
    if y_true.ndim > 1 and y_true.shape[1] > 1: # Check if it looks like one-hot encoded
        y_true_processed = np.argmax(y_true, axis=1)
    else:
        y_true_processed = np.asarray(y_true).ravel() # Ensure it's a 1D numpy array

    # Ensure y_pred_labels is a 1D numpy array of integer labels
    y_pred_labels_processed = np.asarray(y_pred_labels).ravel()

    # --- Debugging prints (optional, remove after fixing) ---
    # print(f"DEBUG in display_result: y_true_processed type: {type(y_true_processed)}, shape: {y_true_processed.shape}, unique: {np.unique(y_true_processed)}")
    # print(f"DEBUG in display_result: y_pred_labels_processed type: {type(y_pred_labels_processed)}, shape: {y_pred_labels_processed.shape}, unique: {np.unique(y_pred_labels_processed)}")
    # print(f"DEBUG in display_result: y_true_processed values sample: {y_true_processed[:5]}")
    # print(f"DEBUG in display_result: y_pred_labels_processed values sample: {y_pred_labels_processed[:5]}")
    # --- End Debugging prints ---

    print('Accuracy score : ', accuracy_score(y_true_processed, y_pred_labels_processed))
    print('Precision score : ', precision_score(y_true_processed, y_pred_labels_processed, average='weighted', zero_division=0))
    print('Recall score : ', recall_score(y_true_processed, y_pred_labels_processed, average='weighted', zero_division=0))
    print('F1 score : ', f1_score(y_true_processed, y_pred_labels_processed, average='weighted', zero_division=0))
    print('Balanced Accuracy score : ', balanced_accuracy_score(y_true_processed, y_pred_labels_processed))
    print('Confusion Matrix:\n', confusion_matrix(y_true_processed, y_pred_labels_processed))


def plot_training_history(history, model_name="Model"):
    """
    Plots training history (accuracy, loss, precision, recall, f1_score).

    Args:
        history (keras.callbacks.History): History object returned by model.fit().
        model_name (str): Name of the model for plot titles.
    """
    plt.figure(figsize=(15, 10))

    # Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history.get('categorical_accuracy'), label='train_accuracy')
    plt.plot(history.history.get('val_categorical_accuracy'), label='valid_accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history.get('loss'), label='train_loss')
    plt.plot(history.history.get('val_loss'), label='valid_loss')
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Precision
    plt.subplot(2, 2, 3)
    plt.plot(history.history.get('precision'), label='train_precision')
    plt.plot(history.history.get('val_precision'), label='valid_precision')
    plt.title(f'{model_name} Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history.get('recall'), label='train_recall')
    plt.plot(history.history.get('val_recall'), label='valid_recall')
    plt.title(f'{model_name} Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    # F1 Score (if available)
    # Use .get() to safely access keys, as F1Score metric might not be present in older TF/Keras versions or if not compiled with it
    if 'f1_score' in history.history and 'val_f1_score' in history.history:
        plt.figure(figsize=(7, 5))
        plt.plot(history.history['f1_score'], label='train_f1_score')
        plt.plot(history.history['val_f1_score'], label='valid_f1_score')
        plt.title(f'{model_name} F1 Score')
        plt.ylabel('F1 Score')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()