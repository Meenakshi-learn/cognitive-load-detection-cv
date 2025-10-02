import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Global variables for dataset paths
DATA_PATHS = {
    # Updated path for FER2013, pointing to the main folder with 'train' and 'test' subdirectories
    'FER2013_Root': 'C:\\Users\\1_HOME\\2_Meenakshi\\1_Now_M.Tech_CSE_DSU\\FYP_M.Tech\\Online Datasets\\FER2013 Facial Expression Recognition 2013_Kaggle',
    # Updated path for JAFFE, pointing to the folder with all the .tiff files
    'JAFFE': 'C:\\Users\\1_HOME\\2_Meenakshi\\1_Now_M.Tech_CSE_DSU\\FYP_M.Tech\\Online Datasets\\JAFFE-Japanese Female Facial Expression Database\\jaffe'
}

def load_images_from_folders(data_path, image_size=(48, 48)):
    """
    Loads and preprocesses images from a folder-based dataset.
    The dataset is expected to have emotion subfolders.
    """
    print(f"Loading images from folder structure at {data_path}...")
    if not os.path.isdir(data_path):
        print(f"Error: Directory not found at {data_path}")
        return None, None
        
    images = []
    labels = []
    
    emotion_folders = sorted(os.listdir(data_path))
    
    for emotion_folder in emotion_folders:
        emotion_path = os.path.join(data_path, emotion_folder)
        if os.path.isdir(emotion_path):
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
                        print(f"    Warning: Could not read image {image_path}. Skipping.")
        
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Dataset loaded. Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels

def load_jaffe(data_path, image_size=(48, 48)):
    """
    Loads and preprocesses the JAFFE dataset from a single folder of TIFF files.
    """
    print(f"Loading JAFFE dataset from {data_path}...")
    if not os.path.isdir(data_path):
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
                print(f"Skipping file {filename} due to an error: {e}")
                
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"JAFFE loaded. Images shape: {images.shape}, Labels shape: {labels.shape}")
    return images, labels

def preprocess_data_with_encoder(images, labels, encoder):
    """
    Applies final preprocessing steps using a pre-fitted LabelEncoder.
    """
    print("Applying final data preprocessing...")
    
    integer_labels = encoder.transform(labels)
    one_hot_labels = to_categorical(integer_labels, num_classes=len(encoder.classes_))
    
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    
    print(f"Preprocessing complete. Images shape: {images.shape}, Labels shape: {one_hot_labels.shape}")
    return images, one_hot_labels

def main():
    """
    Main function to orchestrate the entire data pipeline.
    """
    # 1. Load all raw data from specified directories
    fer_train_path = os.path.join(DATA_PATHS['FER2013_Root'], 'train')
    train_images, train_labels = load_images_from_folders(fer_train_path)

    fer_test_path = os.path.join(DATA_PATHS['FER2013_Root'], 'test')
    val_images, val_labels = load_images_from_folders(fer_test_path)
    
    jaffe_images, jaffe_labels = load_jaffe(DATA_PATHS['JAFFE'])
    
    if train_images is None or val_images is None or jaffe_images is None:
        print("Dataset loading failed. Exiting.")
        return
        
    # 2. Combine all raw labels to ensure consistent one-hot encoding
    all_raw_labels = np.concatenate((train_labels, val_labels, jaffe_labels), axis=0)
    le = LabelEncoder()
    le.fit(all_raw_labels)

    # 3. Process each dataset using the fitted encoder
    train_images_proc, train_labels_proc = preprocess_data_with_encoder(train_images, train_labels, le)
    val_images_proc, val_labels_proc = preprocess_data_with_encoder(val_images, val_labels, le)
    jaffe_images_proc, jaffe_labels_proc = preprocess_data_with_encoder(jaffe_images, jaffe_labels, le)

    # 4. Create final combined training and validation sets
    X_train = np.concatenate((train_images_proc, jaffe_images_proc), axis=0)
    y_train = np.concatenate((train_labels_proc, jaffe_labels_proc), axis=0)
    
    X_val = val_images_proc
    y_val = val_labels_proc
    
    print("\n--- Final Data Shapes ---")
    print(f"Combined Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    print(f"Final Emotion labels: {list(le.classes_)}")

if __name__ == "__main__":
    main()