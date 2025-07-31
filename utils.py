# utils.py

import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import os
import config

def set_seed(seed=config.RANDOM_SEED):
    """
    Sets random seeds for reproducibility across TensorFlow, NumPy, and Python's random module.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # Enable deterministic ops for TensorFlow 2.4+

def display_result(y_true, y_pred_labels):
    """
    Displays accuracy, precision, recall, and F1-score.

    Args:
        y_true (array-like): True labels.
        y_pred_labels (array-like): Predicted labels.
    """
    print('Accuracy score : ', accuracy_score(y_true, y_pred_labels))
    print('Precision score : ', precision_score(y_true, y_pred_labels, average='weighted', zero_division=0))
    print('Recall score : ', recall_score(y_true, y_pred_labels, average='weighted', zero_division=0))
    print('F1 score : ', f1_score(y_true, y_pred_labels, average='weighted', zero_division=0))
    print('Balanced Accuracy score : ', balanced_accuracy_score(y_true, y_pred_labels))
    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred_labels))


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
    plt.plot(history.history['categorical_accuracy'], label='train_accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='valid_accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='valid_loss')
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Precision
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'], label='train_precision')
    plt.plot(history.history['val_precision'], label='valid_precision')
    plt.title(f'{model_name} Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'], label='train_recall')
    plt.plot(history.history['val_recall'], label='valid_recall')
    plt.title(f'{model_name} Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    # F1 Score (if available)
    if 'f1_score' in history.history and 'val_f1_score' in history.history:
        plt.figure(figsize=(7, 5))
        plt.plot(history.history['f1_score'], label='train_f1_score')
        plt.plot(history.history['val_f1_score'], label='valid_f1_score')
        plt.title(f'{model_name} F1 Score')
        plt.ylabel('F1 Score')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()