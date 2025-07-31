# models/mlp_model.py

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import os

import config
from utils import set_seed, display_result, plot_training_history

def build_mlp_model(input_shape):
    """
    Builds and compiles a Multi-Layer Perceptron (MLP) model for CSV data.

    Args:
        input_shape (int): The number of input features for the MLP.

    Returns:
        tf.keras.Model: The compiled MLP model.
    """
    set_seed(config.RANDOM_SEED)
    
    model = Sequential([
        Dense(2000, activation=tf.nn.relu, input_shape=(input_shape,)),
        BatchNormalization(),
        Dense(600, activation=tf.nn.relu),
        BatchNormalization(),
        Dropout(0.2),
        Dense(config.NUM_CLASSES, activation='softmax'),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, name='Adam'),
        loss='categorical_crossentropy',
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(name='f1_score', average='weighted'),
        ]
    )
    return model

def train_and_evaluate_mlp(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_test_raw):
    """
    Trains, evaluates, and saves the MLP model.

    Args:
        X_train (np.ndarray): Training features.
        Y_train (np.ndarray): One-hot encoded training labels.
        X_val (np.ndarray): Validation features.
        Y_val (np.ndarray): One-hot encoded validation labels.
        X_test (np.ndarray): Test features.
        Y_test (np.ndarray): One-hot encoded test labels.
        y_test_raw (np.ndarray): Raw integer test labels (for sklearn metrics).
    """
    print("\n--- Training Multilayer Perceptron (MLP) Model ---")
    model_mlp = build_mlp_model(X_train.shape[1])
    model_mlp.summary()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.MLP_MODEL_PATH), exist_ok=True)

    f1_callback_mlp = ModelCheckpoint(
        config.MLP_MODEL_PATH,
        monitor='val_f1_score',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    history_mlp = model_mlp.fit(
        X_train, Y_train,
        epochs=150,
        batch_size=2**10,
        validation_data=(X_val, Y_val),
        callbacks=[f1_callback_mlp]
    )

    print("\n--- MLP Model Evaluation ---")
    print("best model: ")
    # Load the best model weights saved by ModelCheckpoint
    model_mlp.load_weights(config.MLP_MODEL_PATH)

    print('Validation Set')
    val_results = model_mlp.evaluate(X_val, Y_val, verbose=0)
    print(f"Loss: {val_results[0]:.4f}, Accuracy: {val_results[1]:.4f}, "
          f"Precision: {val_results[2]:.4f}, Recall: {val_results[3]:.4f}, F1-Score: {val_results[4]:.4f}")

    print('Test Set')
    test_results = model_mlp.evaluate(X_test, Y_test, verbose=0)
    print(f"Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}, "
          f"Precision: {test_results[2]:.4f}, Recall: {test_results[3]:.4f}, F1-Score: {test_results[4]:.4f}")

    # For sklearn metrics, predict raw labels
    y_pred_mlp_prob = model_mlp.predict(X_test)
    y_pred_mlp_labels = tf.argmax(y_pred_mlp_prob, axis=1).numpy()
    display_result(y_test_raw, y_pred_mlp_labels)

    plot_training_history(history_mlp, "MLP (CSV Data)")