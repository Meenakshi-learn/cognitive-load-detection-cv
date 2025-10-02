import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import get_preprocessed_data_for_cnn_lstm

# Ensure the saved models directory exists
MODEL_DIR = 'saved_models'
if not os.path.exists(MODEL_DIR):
    print("Error: The 'saved_models' directory was not found. Please run the training script first.")
    exit()

def evaluate_and_visualize(model_name, model, X_val, y_val, le):
    """
    Evaluates a single model and generates its confusion matrix.
    """
    print(f"\n--- Evaluating {model_name} ---")
    
    # Get model accuracy
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy for {model_name}: {accuracy:.4f}")
    
    # Generate confusion matrix
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    print(f"Saved confusion matrix to confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.close()

def live_testing_setup():
    """
    Loads all models and evaluates them to generate all requested outputs.
    """
    # Load preprocessed validation data
    print("Loading preprocessed validation data...")
    X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, le = get_preprocessed_data_for_cnn_lstm(verbose=False)
    
    if X_val_sequences is None:
        print("Failed to load validation data. Exiting.")
        return

    # Base models to load and evaluate
    model_names = [
        "CNN_LSTM"
    ]

    # Evaluate each base model
    for name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{name}.h5")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            evaluate_and_visualize(name, model, X_val_sequences, y_val_sequences, le)
        else:
            print(f"Warning: Model file {model_path} not found. Skipping.")

if __name__ == "__main__":
    live_testing_setup()