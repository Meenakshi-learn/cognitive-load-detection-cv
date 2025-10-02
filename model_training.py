import time
import os
import numpy as np
import json
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization, Dropout, TimeDistributed, LSTM
from tensorflow.keras.optimizers import Adam
from data_preprocessing import get_preprocessed_data_for_cnn_lstm
from advanced_models import build_vgg16_model, build_cnn_lstm_model 

# --- Model Building Functions ---

def build_simple_cnn(input_shape, num_classes):
    """Builds a simple-CNN model."""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def build_simpler_cnn(input_shape, num_classes):
    """Builds a simpler-CNN model."""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def build_tiny_xception(input_shape, num_classes):
    """Builds a tiny-XCEPTION model."""
    inputs = Input(shape=input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = SeparableConv2D(8, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def build_mini_xception(input_shape, num_classes):
    """Builds a mini-XCEPTION model."""
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def build_big_xception(input_shape, num_classes):
    """Builds a big-XCEPTION model."""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def create_stacking_ensemble(models, input_shape, num_classes):
    """Creates the stacking ensemble model."""
    model_inputs = Input(shape=input_shape)
    base_model_outputs = [model(model_inputs) for model in models]
    
    concatenated_output = tf.keras.layers.Concatenate()(base_model_outputs)
    
    meta_model = Dense(128, activation='relu')(concatenated_output)
    meta_model = Dense(num_classes, activation='softmax')(meta_model)
    
    return Model(inputs=model_inputs, outputs=meta_model)

def train_and_evaluate_models(X_train, y_train, X_val, y_val, le):
    """Trains and evaluates the CNN+LSTM model using sequence data."""
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    # Define the model builders for this CNN+LSTM training
    model_builders = {
        'CNN_LSTM': build_cnn_lstm_model
    }
    
    trained_models = []
    
    # Create the directory to save models and results
    os.makedirs('saved_models', exist_ok=True)
    results = {}
    
    print("--- Training CNN+LSTM Model ---")
    for name, builder in model_builders.items():
        print(f"Training {name}...")
        model = builder(input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train directly on the 5D sequence arrays
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=1)
        
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy for {name}: {accuracy:.4f}\n")
        
        model_path = f'saved_models/{name.replace(" ", "_")}.h5'
        model.save(model_path)
        print(f"Saved model to {model_path}")
        
        results[name] = {
            'accuracy': accuracy,
            'history': {k: v for k, v in history.history.items()}
        }
        
        trained_models.append(model)
        
        # Generate and save a confusion matrix
        y_pred_probs = model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
        print(f"Saved confusion matrix for {name} to confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close() # Close the plot figure to free up memory

    with open('training_results_cnn_lstm.json', 'w') as f:
        json.dump(results, f, indent=4)
        print("Saved training results to training_results_cnn_lstm.json")


if __name__ == "__main__":
    print("Starting data preparation...")
    X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, le = get_preprocessed_data_for_cnn_lstm(verbose=True) 
    if X_train_sequences is not None:
        print("\n--- Starting Model Training ---")
        train_and_evaluate_models(X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, le)
    else:
        print("Model training could not proceed due to data loading errors.")
    time.sleep(2)