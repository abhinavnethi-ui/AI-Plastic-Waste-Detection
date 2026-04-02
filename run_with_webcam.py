import cv2
import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------------
# Configuration
# ------------------------
# Using local webcam instead of phone
VIDEO_URL = 0  # Use local webcam (device 0)

MODEL_PATH = "train_model/plastic_model.h5"  # your trained CNN model
PLASTIC_TYPES = ["PET", "LDPE", "HDPE", "PVC", "PP", "PS", "OTHER_TYPES"]  # your plastic classes
TARGET_SIZE = (128, 128)  # Match the model's expected input size

# ------------------------
# Load trained CNN model
# ------------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Model not found at {MODEL_PATH}. Please train the model first.")
    exit()

# ------------------------
# Helper function to predict plastic type
# ------------------------
def predict_plastic(frame):
    img = cv2.resize(frame, TARGET_SIZE)
    arr = img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)
    return PLASTIC_TYPES[np.argmax(pred)]

# ------------------------
# Start camera
# ------------------------
print("🎥 Starting webcam...")
cap = cv2.VideoCapture(VIDEO_URL)

if not cap.isOpened():
    print("❌ Could not open webcam. Make sure no other app is using it.")
    exit()

print("✅ Webcam opened successfully!")
print("📱 Press 'q' to quit")
print("🔍 Point plastic objects at the camera to classify them!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Predict plastic type
    plastic_type = predict_plastic(frame)

    # Display predicted type on frame
    cv2.putText(frame, f"Predicted: {plastic_type}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Live Plastic Prediction", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Goodbye!")
