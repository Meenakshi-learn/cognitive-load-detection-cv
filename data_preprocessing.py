import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image
import time

# Global variables for dataset paths
DATA_PATHS = {
    'FER2013_Root': 'C:\\Users\\1_HOME\\2_Meenakshi\\1_Now_M.Tech_CSE_DSU\\FYP_M.Tech\\Online Datasets\\FER2013 Facial Expression Recognition 2013_Kaggle',
    'JAFFE': 'C:\\Users\\1_HOME\\2_Meenakshi\\1_Now_M.Tech_CSE_DSU\\FYP_M.Tech\\Online Datasets\\JAFFE-Japanese Female Facial Expression Database\\jaffe'
}

def load_images_from_folders(data_path, image_size=(48, 48), verbose=True):
    """Loads and preprocesses images from a folder-based dataset."""
    if verbose:
        print(f"Loading images from folder structure at {data_path}...")
    
    if not os.path.isdir(data_path):
        if verbose:
            print(f"Error: Directory not found at {data_path}")
        return None, None
    
    images = []
    labels = []
    
    emotion_folders = sorted(os.listdir(data_path))
    
    for emotion_folder in emotion_folders:
        emotion_path = os.path.join(data_path, emotion_folder)
        if os.path.isdir(emotion_path):
            if verbose:
                print(f"  Processing folder: {emotion_folder}")
            for filename in os.listdir(emotion_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(emotion_path, filename)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        resized_image = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
                        normalized_image = resized_image / 255.0
                        images.append(normalized_image)
                        labels.append(emotion_folder)
                    else:
                        if verbose:
                            print(f"    Warning: Could not read image {image_path}. Skipping.")
    
    images = np.array(images)
    labels = np.array(labels)
    
    if verbose:
        print(f"Dataset loaded. Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels

def load_jaffe(data_path, image_size=(48, 48), verbose=True):
    """Loads and preprocesses the JAFFE dataset from a single folder of TIFF files."""
    if verbose:
        print(f"Loading JAFFE dataset from {data_path}...")
    
    if not os.path.isdir(data_path):
        if verbose:
            print(f"Error: JAFFE directory not found at {data_path}")
        return None, None
    
    images = []
    labels = []
    
    label_map = {
        'AN': 'anger', 'DI': 'disgust', 'FE': 'fear', 
        'HA': 'happy', 'SA': 'sadness', 'SU': 'surprise', 'NE': 'neutral'
    }
    
    for filename in os.listdir(data_path):
        if filename.endswith(".tiff"):
            try:
                emotion_code = filename.split('.')[1][:2].upper()
                if emotion_code in label_map:
                    label = label_map[emotion_code]
                    image_path = os.path.join(data_path, filename)
                    img = Image.open(image_path).convert('L')
                    img_array = np.array(img, dtype='float32')
                    resized_image = cv2.resize(img_array, image_size, interpolation=cv2.INTER_AREA)
                    normalized_image = resized_image / 255.0
                    images.append(normalized_image)
                    labels.append(label)
            except (IOError, IndexError) as e:
                if verbose:
                    print(f"Skipping file {filename} due to an error: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    if verbose:
        print(f"JAFFE loaded. Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels

def preprocess_data_with_encoder(images, labels, encoder, verbose=True):
    """Applies final preprocessing steps and prepares images for 3-channel input."""
    if verbose:
        print("Applying final data preprocessing...")
    integer_labels = encoder.transform(labels)
    one_hot_labels = to_categorical(integer_labels, num_classes=len(encoder.classes_))
    
    # Check if images are grayscale (4D tensor with 1 channel)
    if images.ndim == 4 and images.shape[-1] == 1:
        # Convert grayscale to 3 channels by stacking
        images = np.concatenate([images, images, images], axis=-1)
    elif images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
        images = np.concatenate([images, images, images], axis=-1)

    if verbose:
        print(f"Preprocessing complete. Images shape: {images.shape}, Labels shape: {one_hot_labels.shape}")
    return images, one_hot_labels

def create_sequences(images, labels, sequence_length=5):
    """
    Creates sequences of images and labels for spatiotemporal modeling.
    This function simulates video sequences from individual images.
    """
    sequences = []
    sequence_labels = []

    # We will slide a window of size sequence_length over the images.
    # The label for a sequence is the label of the final image in that sequence.
    for i in range(len(images) - sequence_length):
        sequences.append(images[i:i + sequence_length])
        sequence_labels.append(labels[i + sequence_length])

    return np.array(sequences), np.array(sequence_labels)

def get_preprocessed_data_for_cnn_lstm(verbose=True):
    """Orchestrates the data pipeline for CNN + LSTM and returns sequences."""
    fer_train_path = os.path.join(DATA_PATHS['FER2013_Root'], 'train')
    train_images_raw, train_labels_raw = load_images_from_folders(fer_train_path, verbose=verbose)

    fer_test_path = os.path.join(DATA_PATHS['FER2013_Root'], 'test')
    val_images_raw, val_labels_raw = load_images_from_folders(fer_test_path, verbose=verbose)
    
    jaffe_images_raw, jaffe_labels_raw = load_jaffe(DATA_PATHS['JAFFE'], verbose=verbose)
    
    if train_images_raw is None or val_images_raw is None or jaffe_images_raw is None:
        if verbose:
            print("Dataset loading failed.")
        return None, None, None, None, None
        
    all_raw_labels = np.concatenate((train_labels_raw, val_labels_raw, jaffe_labels_raw), axis=0)
    le = LabelEncoder()
    le.fit(all_raw_labels)

    train_images_proc, train_labels_proc = preprocess_data_with_encoder(train_images_raw, train_labels_raw, le, verbose=verbose)
    val_images_proc, val_labels_proc = preprocess_data_with_encoder(val_images_raw, val_labels_raw, le, verbose=verbose)
    jaffe_images_proc, jaffe_labels_proc = preprocess_data_with_encoder(jaffe_images_raw, jaffe_labels_raw, le, verbose=verbose)

    X_train_combined = np.concatenate((train_images_proc, jaffe_images_proc), axis=0)
    y_train_combined = np.concatenate((train_labels_proc, jaffe_labels_proc), axis=0)
    
    # Create sequences from the processed data
    X_train_sequences, y_train_sequences = create_sequences(X_train_combined, y_train_combined, sequence_length=5)
    X_val_sequences, y_val_sequences = create_sequences(val_images_proc, val_labels_proc, sequence_length=5)

    return X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, le

if __name__ == "__main__":
    print("Running data preprocessing script...")
    X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, le = get_preprocessed_data_for_cnn_lstm(verbose=True)
    if X_train_sequences is not None:
        print("\n--- Final Data Shapes ---")
        print(f"Combined Training set shape: {X_train_sequences.shape}, {y_train_sequences.shape}")
        print(f"Validation set shape: {X_val_sequences.shape}, {y_val_sequences.shape}")
        print(f"Final Emotion labels: {list(le.classes_)}")
    else:
        print("Data preparation failed.")