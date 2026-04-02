import cv2
import os
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import time
import csv

# ------------------------
# Configuration
# ------------------------
MODEL_PATH = "train_model/plastic_model.h5"
LABELS_PATH = "train_model/labels.json"
IMAGE_FOLDER = "images"
TARGET_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5
RETRAIN_INTERVAL = 5  # retrain after every 5 new images
CONFIDENCE_THRESHOLD = 0.7  # minimum confidence for saving

# Construction optimization data
CONSTRUCTION_DATA = {
    "HDPE": {"recyclability": 95, "durability": 90, "cost": 60, "construction_use": "Pipes, containers, outdoor furniture"},
    "LDPE": {"recyclability": 85, "durability": 70, "cost": 50, "construction_use": "Insulation, vapor barriers, protective films"},
    "PET": {"recyclability": 100, "durability": 80, "cost": 70, "construction_use": "Fiber insulation, composite materials"},
    "PP": {"recyclability": 90, "durability": 95, "cost": 55, "construction_use": "Pipes, fittings, geotextiles"},
    "PS": {"recyclability": 60, "durability": 60, "cost": 40, "construction_use": "Insulation boards, decorative panels"},
    "PVC": {"recyclability": 50, "durability": 85, "cost": 45, "construction_use": "Pipes, window frames, flooring"}
}

# ------------------------
# Load model and labels
# ------------------------
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}. Please train the model first.")
    exit()

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

if not os.path.exists(LABELS_PATH):
    print(f"❌ Labels file not found at {LABELS_PATH}. Please retrain to generate it.")
    exit()

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    PLASTIC_TYPES = json.load(f)["labels"]

print(f"📋 Available plastic types: {', '.join(PLASTIC_TYPES)}")

# ------------------------
# Helper functions
# ------------------------
def predict_plastic(frame):
    img = cv2.resize(frame, TARGET_SIZE)
    arr = img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)
    confidence = np.max(pred)
    predicted_class = PLASTIC_TYPES[np.argmax(pred)]
    return predicted_class, confidence, pred[0]

def save_detection(frame, plastic_type, confidence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plastic_type}_{confidence:.2f}_{timestamp}.jpg"
    save_folder = os.path.join(IMAGE_FOLDER, plastic_type)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    cv2.imwrite(save_path, cv2.resize(frame, TARGET_SIZE))
    return filename

def log_detection(timestamp, filename, plastic_type, confidence):
    log_file = "detection_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Filename', 'PlasticType', 'Confidence', 'Recyclability%', 'Durability%', 'Cost%'])
        
        data = CONSTRUCTION_DATA.get(plastic_type, {})
        writer.writerow([
            timestamp, filename, plastic_type, f"{confidence:.2f}",
            data.get('recyclability', 0), data.get('durability', 0), data.get('cost', 0)
        ])

def retrain_model():
    print("🔄 Retraining model with new data...")
    try:
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(
            IMAGE_FOLDER, target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE, subset='training', class_mode='categorical'
        )
        val_gen = datagen.flow_from_directory(
            IMAGE_FOLDER, target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE, subset='validation', class_mode='categorical'
        )
        
        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train
        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=1)
        
        # Save updated model
        save_model(model, MODEL_PATH)
        
        # Get final accuracy
        final_accuracy = history.history['accuracy'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        
        print(f"✅ Model retrained! Training accuracy: {final_accuracy:.2f}, Validation accuracy: {val_accuracy:.2f}")
        return final_accuracy, val_accuracy
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        return 0, 0

def get_construction_analysis(plastic_type):
    data = CONSTRUCTION_DATA.get(plastic_type, {})
    return {
        'recyclability': data.get('recyclability', 0),
        'durability': data.get('durability', 0),
        'cost': data.get('cost', 0),
        'construction_use': data.get('construction_use', 'Unknown')
    }

def calculate_optimization_score(plastic_type, confidence):
    data = CONSTRUCTION_DATA.get(plastic_type, {})
    recyclability = data.get('recyclability', 0)
    durability = data.get('durability', 0)
    cost = data.get('cost', 0)
    
    # Weighted score: 40% recyclability, 30% durability, 20% cost, 10% confidence
    score = (recyclability * 0.4 + durability * 0.3 + (100-cost) * 0.2 + confidence * 100 * 0.1)
    return min(100, max(0, score))

# ------------------------
# Initialize counters and logs
# ------------------------
detection_count = 0
total_detections = {ptype: 0 for ptype in PLASTIC_TYPES}
accuracy_history = []

# ------------------------
# Start camera
# ------------------------
print("🎥 Starting webcam...")
cap = cv2.VideoCapture(0)  # Use computer webcam

if not cap.isOpened():
    print("❌ Could not open webcam. Make sure no other app is using it.")
    exit()

print("✅ Webcam opened successfully!")
print("📱 Press 'q' to quit, 'r' to retrain manually, 's' to show statistics")
print("🔍 Point plastic objects at the camera to classify them!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Predict plastic type
    plastic_type, confidence, all_predictions = predict_plastic(frame)
    
    # Display prediction with confidence
    color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 255, 255)
    cv2.putText(frame, f"Predicted: {plastic_type}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show construction analysis
    analysis = get_construction_analysis(plastic_type)
    optimization_score = calculate_optimization_score(plastic_type, confidence)
    
    cv2.putText(frame, f"Recyclability: {analysis['recyclability']}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Durability: {analysis['durability']}%", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Optimization: {optimization_score:.1f}%", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Auto-save high confidence detections
    if confidence > CONFIDENCE_THRESHOLD:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_detection(frame, plastic_type, confidence)
        log_detection(timestamp, filename, plastic_type, confidence)
        
        total_detections[plastic_type] += 1
        detection_count += 1
        
        print(f"💾 Saved: {filename} (Confidence: {confidence:.2f})")
        
        # Retrain if enough new detections
        if detection_count % RETRAIN_INTERVAL == 0:
            train_acc, val_acc = retrain_model()
            accuracy_history.append((train_acc, val_acc))
    
    # Show all predictions
    y_offset = 160
    for i, (ptype, pred_val) in enumerate(zip(PLASTIC_TYPES, all_predictions)):
        color = (0, 255, 0) if ptype == plastic_type else (128, 128, 128)
        cv2.putText(frame, f"{ptype}: {pred_val:.2f}", (10, y_offset + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.putText(frame, f"Total detections: {detection_count}", (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit, 'r' to retrain, 's' for stats", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Smart Plastic Detector", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        train_acc, val_acc = retrain_model()
        accuracy_history.append((train_acc, val_acc))
    elif key == ord('s'):
        print("\n📊 DETECTION STATISTICS:")
        print("=" * 50)
        for ptype, count in total_detections.items():
            if count > 0:
                analysis = get_construction_analysis(ptype)
                print(f"{ptype}: {count} detections")
                print(f"  - Recyclability: {analysis['recyclability']}%")
                print(f"  - Durability: {analysis['durability']}%")
                print(f"  - Cost: {analysis['cost']}%")
                print(f"  - Construction use: {analysis['construction_use']}")
                print()
        
        if accuracy_history:
            print("📈 ACCURACY HISTORY:")
            for i, (train_acc, val_acc) in enumerate(accuracy_history):
                print(f"  Retrain {i+1}: Training={train_acc:.2f}, Validation={val_acc:.2f}")
        print("=" * 50)

cap.release()
cv2.destroyAllWindows()

# Final summary
print("\n🎯 FINAL SUMMARY:")
print(f"Total detections: {detection_count}")
print("Detection breakdown:")
for ptype, count in total_detections.items():
    if count > 0:
        percentage = (count / detection_count) * 100
        print(f"  {ptype}: {count} ({percentage:.1f}%)")

print("\n💡 CONSTRUCTION OPTIMIZATION RECOMMENDATIONS:")
# Sort by optimization score
sorted_plastics = sorted(PLASTIC_TYPES, key=lambda x: calculate_optimization_score(x, 1.0), reverse=True)
for i, ptype in enumerate(sorted_plastics):
    if total_detections[ptype] > 0:
        score = calculate_optimization_score(ptype, 1.0)
        analysis = get_construction_analysis(ptype)
        print(f"{i+1}. {ptype} (Score: {score:.1f}) - {analysis['construction_use']}")

print("👋 Goodbye!")
