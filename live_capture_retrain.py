import cv2
import os
import json

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from datetime import datetime
import time
import csv

# -----------------------------
# Paths
# -----------------------------
project_root = os.path.abspath(os.path.dirname(__file__))
images_folder = os.path.join(project_root, '..', 'images')
model_path = os.path.join(project_root, '..', 'train_model', 'plastic_model.h5')
log_file = os.path.join(project_root, '..', 'capture_log.csv')

labels_path = os.path.join(project_root, '..', 'train_model', 'labels.json')
plastic_types = None
if os.path.isfile(labels_path):
	with open(labels_path, 'r', encoding='utf-8') as f:
		plastic_types = json.load(f)["labels"]
else:
	# Fallback to directory names if labels.json not found
	plastic_types = sorted([d for d in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, d))])
TARGET_SIZE = (128, 128)  # Match the model's expected input size
BATCH_SIZE = 16
EPOCHS = 2  # small epochs for on-the-fly retraining
RETRAIN_INTERVAL = 10  # retrain after every 10 new images

# -----------------------------
# Load or initialize model
# -----------------------------
if os.path.isfile(model_path):
    model = load_model(model_path)
    print("Existing model loaded!")
else:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(len(plastic_types), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    print("New model initialized!")

# -----------------------------
# Helper: Predict plastic type
# -----------------------------
def predict_plastic(img):
    img_resized = cv2.resize(img, TARGET_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return plastic_types[class_index]

# -----------------------------
# Helper: Check unique frame
# -----------------------------
def is_different(frame1, frame2, threshold=1000):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    score = np.sum(diff)
    return score > threshold

# -----------------------------
# Prepare folders and CSV
# -----------------------------
for p_type in plastic_types:
    os.makedirs(os.path.join(images_folder, p_type), exist_ok=True)
if not os.path.isfile(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp','Filename','PlasticType'])

# -----------------------------
# Live camera loop
# -----------------------------
cap = cv2.VideoCapture("http://192.168.33.247:8080/video")
capture_interval = 5
last_capture_time = time.time()
last_frame = None
new_image_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    plastic_type = predict_plastic(frame)

    cv2.putText(frame, f"Predicted: {plastic_type}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.putText(frame, "Press 'c' to manual capture, 'q' to quit", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    cv2.imshow("Live Plastic Capture & Retrain", frame)

    current_time = time.time()
    # Automatic capture
    if current_time - last_capture_time >= capture_interval:
        if last_frame is None or is_different(frame, last_frame):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{plastic_type}_{timestamp}.jpg"
            save_path = os.path.join(images_folder, plastic_type, filename)
            frame_resized = cv2.resize(frame, TARGET_SIZE)
            cv2.imwrite(save_path, frame_resized)
            print(f"Auto-saved: {filename} in {plastic_type}")

            # Log
            with open(log_file,'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, filename, plastic_type])

            last_frame = frame.copy()
            new_image_count += 1
        last_capture_time = current_time

    # Manual capture
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_manual")
        filename = f"{plastic_type}_{timestamp}.jpg"
        save_path = os.path.join(images_folder, plastic_type, filename)
        frame_resized = cv2.resize(frame, TARGET_SIZE)
        cv2.imwrite(save_path, frame_resized)
        print(f"Manual saved: {filename} in {plastic_type}")
        with open(log_file,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, filename, plastic_type])
        new_image_count += 1
    elif key == ord('q'):
        break

    # Retrain if enough new images
    if new_image_count >= RETRAIN_INTERVAL:
        print("Retraining model with new images...")
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(images_folder, target_size=TARGET_SIZE,
                                                batch_size=BATCH_SIZE, subset='training', class_mode='categorical')
        val_gen = datagen.flow_from_directory(images_folder, target_size=TARGET_SIZE,
                                              batch_size=BATCH_SIZE, subset='validation', class_mode='categorical')
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
        save_model(model, model_path)
        print("Model retrained and saved!")
        new_image_count = 0

cap.release()
cv2.destroyAllWindows()
