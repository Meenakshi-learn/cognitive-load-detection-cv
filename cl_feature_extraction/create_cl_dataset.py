# --- File: create_cl_dataset.py ---
import os
import cv2
import numpy as np
import pandas as pd
from cl_feature_logic import extract_cl_features # Import the new logic
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing import DATA_PATHS # This now correctly finds the file in the parent folder
import tensorflow as tf

# Load the saved CNN+LSTM Teacher Model (for the Emotion feature)
TEACHER_MODEL_PATH = 'saved_models/CNN_LSTM.h5'
try:
    teacher_model = tf.keras.models.load_model(TEACHER_MODEL_PATH)
except:
    print("Warning: Teacher model not loaded. Emotion feature will be skipped.")
    teacher_model = None

def generate_cl_dataset():
    """
    Iterates through FER2013 training and testing images to create a unified 
    numerical feature dataset for Cognitive Load training.
    """
    
    # Get list of all image paths and their true labels from the training directory
    image_data = []
    
    for data_dir in ['train', 'test']:
        full_dir = os.path.join(DATA_PATHS['FER2013_Root'], data_dir)
        for emotion in os.listdir(full_dir):
            emotion_path = os.path.join(full_dir, emotion)
            if os.path.isdir(emotion_path):
                for filename in os.listdir(emotion_path):
                    if filename.endswith(('.jpg', '.png', '.jpeg')):
                        image_data.append({
                            'path': os.path.join(emotion_path, filename),
                            'true_label': emotion
                        })

    feature_rows = []
    print(f"Processing {len(image_data)} images for feature extraction...")
    
    for i, item in enumerate(image_data):
        img = cv2.imread(item['path'])
        if img is None: continue

        # 1. Extract geometric features (Head Pose & EAR)
        cl_features = extract_cl_features(img)
        
        if cl_features is not None:
            
            # 2. Get Emotion Score from Teacher Model
            emotion_score = [0] * 9 # Placeholder if model is not loaded
            if teacher_model:
                # Prepare image for CNN input (48x48x3)
                img_resized = cv2.resize(img, (48, 48))
                img_normalized = img_resized / 255.0
                cnn_input = np.expand_dims(img_normalized, axis=0) # (1, 48, 48, 3)
                
                # Predict (Soft Targets for Distillation)
                emotion_prob = teacher_model.predict(cnn_input, verbose=0)[0]
                emotion_score = emotion_prob.tolist() # Get the 9 probability scores

            # 3. Assemble final row
            row = {
                'file_path': item['path'],
                'true_label': item['true_label'],
                'Head_Yaw': cl_features['nose_yaw_proxy'],
                #'Head_Pitch': cl_features['pitch'],
                'EAR': cl_features['ear'],
                # Add the 9 emotion scores as a list or separate columns
            }
            # Add emotion probabilities as separate columns
            for j, prob in enumerate(emotion_score):
                row[f'Emotion_Prob_{j}'] = prob
            
            feature_rows.append(row)
            
        if (i + 1) % 1000 == 0:
            print(f"--- Processed {i+1} images ---")

    # Save the final dataset
    df = pd.DataFrame(feature_rows)
    output_path = 'final_cl_features/cl_multimodal_dataset.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSUCCESS: Multimodal CL dataset saved to {output_path}")

if __name__ == "__main__":
    generate_cl_dataset()