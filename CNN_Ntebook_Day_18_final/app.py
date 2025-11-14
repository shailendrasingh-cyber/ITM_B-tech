import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# Load model and class names
model = load_model("manish_face_cnn_model.h5")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_webcam():
    """Real-time prediction using webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Error: Could not open webcam")
        return
    
    print("🎥 Webcam started. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess and predict
        processed_frame = preprocess_image(frame)
        predictions = model.predict(processed_frame, verbose=0)
        confidence = np.max(predictions[0])
        predicted_class = class_names[np.argmax(predictions[0])]
        
        # Display prediction
        label = f"{predicted_class}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def predict_single_image(image_path):
    """Predict a single image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f" Error: Could not load image {image_path}")
        return
    
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    confidence = np.max(predictions[0])
    predicted_class = class_names[np.argmax(predictions[0])]
    
    print(f"📷 Prediction for {image_path}:")
    print(f"   Class: {predicted_class}")
    print(f"   Confidence: {confidence:.2f}")
    
    # Display image with prediction
    display_image = image.copy()
    cv2.putText(display_image, f"{predicted_class}: {confidence:.2f}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Prediction Result', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose prediction mode
    print("Choose prediction mode:")
    print("1. Webcam Live Prediction")
    print("2. Predict Single Image")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        predict_webcam()
    elif choice == "2":
        image_path = input("Enter image path: ")
        predict_single_image(image_path)
    else:
        print(" Invalid choice")