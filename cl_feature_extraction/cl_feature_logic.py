# --- File: cl_feature_logic.py ---
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Indices for the EAR calculation (simplified example)
# EAR uses points around the eye to measure openness/closure.
LEFT_EYE_INDICES = [33, 160, 158, 133, 144, 153] 

def calculate_ear(face_landmarks, image_w, image_h):
    """Calculates the Eye Aspect Ratio (EAR) for alertness."""
    
    # Get 2D coordinates for the left eye points
    points = np.array([[face_landmarks.landmark[i].x * image_w, face_landmarks.landmark[i].y * image_h] 
                       for i in LEFT_EYE_INDICES])
    
    A = dist.euclidean(points[1], points[5]) 
    B = dist.euclidean(points[2], points[4]) 
    C = dist.euclidean(points[0], points[3]) 
    ear = (A + B) / (2.0 * C)
    return ear

def extract_cl_features(image):
    """
    Extracts essential geometric features (EAR and a normalized Nose Position for Head Pose proxy).
    
    Returns: A dictionary of features or None.
    """
    h, w, c = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None 

    face_landmarks = results.multi_face_landmarks[0]
    
    # 1. Eye Aspect Ratio (Alertness)
    ear_value = calculate_ear(face_landmarks, w, h)
    
    # 2. Head Position Proxy (Normalized Nose X-coordinate for Yaw)
    nose_x_norm = face_landmarks.landmark[33].x # 33 is nose tip index
    
    return {
        'ear': ear_value,
        'nose_yaw_proxy': nose_x_norm
    }