import cv2
import os
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------------
# Configuration
# ------------------------
PHONE_IP = "192.168.33.247:8080"  # Your phone's IP
VIDEO_URL = f"http://{PHONE_IP}/video"

MODEL_PATH = "train_model/plastic_model.h5"  # your trained CNN model
LABELS_PATH = "train_model/labels.json"
TARGET_SIZE = (128, 128)  # Match the model's expected input size

# ------------------------
# Load trained CNN model
# ------------------------
if not os.path.exists(MODEL_PATH):
	print(f"Model not found at {MODEL_PATH}. Please train the model first.")
	exit()
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

if not os.path.exists(LABELS_PATH):
	print(f"Labels file not found at {LABELS_PATH}. Please retrain to generate it.")
	exit()
with open(LABELS_PATH, "r", encoding="utf-8") as f:
	PLASTIC_TYPES = json.load(f)["labels"]

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
cap = cv2.VideoCapture(VIDEO_URL)
print("Press 'q' to quit")

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
