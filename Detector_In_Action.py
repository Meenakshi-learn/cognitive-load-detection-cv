import numpy as np
import argparse
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
from collections import deque

# Load the model and emotion dictionary
model = tf.keras.models.load_model('saved_models/CNN_LSTM.h5')
emotion_dict = {
    0: "anger", 
    1: "angry", 2: "disgust", 3: "fear", 4: "happy",
    5: "neutral", 6: "sad", 7: "sadness", 8: "surprise"
}

# Define a queue to store recent predictions (e.g., last 10 frames) for temporal smoothing
sequence_history = deque(maxlen=5) # Queue to store frames for the CNN+LSTM sequence
prediction_history = deque(maxlen=10) # Queue for temporal smoothing

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        
        # Original code used grayscale image, which has a shape of (48, 48, 1)
        # Your model expects a 3-channel image.
        cropped_img_gray = cv2.resize(roi_gray, (48, 48))
        
        # Convert the grayscale image to a 3-channel image by stacking
        cropped_img_rgb = cv2.cvtColor(cropped_img_gray, cv2.COLOR_GRAY2RGB)

        # Expand dimensions to create the (1, 48, 48, 3) shape that the model expects
        final_cropped_img = np.expand_dims(cropped_img_rgb, 0)

        # Add the current frame to the sequence history
        sequence_history.append(final_cropped_img)
        
        # Only make a prediction if the sequence history is full
        if len(sequence_history) == sequence_history.maxlen:
            # Stack the frames to create a sequence of shape (1, 5, 48, 48, 3)
            input_sequence = np.concatenate(list(sequence_history), axis=0)
            input_sequence = np.expand_dims(input_sequence, 0)
            
            # Get predictions for the current frame
            prediction_probabilities = model.predict(input_sequence, verbose=0)
            
            # Add the prediction to our history queue for temporal smoothing
            prediction_history.append(prediction_probabilities[0])

            # If we have enough history, average the predictions
            if len(prediction_history) == prediction_history.maxlen:
                # Average the probabilities across all stored frames
                averaged_prediction = np.mean(prediction_history, axis=0)
                maxindex = int(np.argmax(averaged_prediction))
            else:
                # Otherwise, use the current prediction
                maxindex = int(np.argmax(prediction_probabilities))
        else:
            maxindex = None
        
        if maxindex is not None:
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()