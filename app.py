from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import io
from PIL import Image
import threading
import time

app = Flask(__name__)

# Configuration
MODEL_PATH = "train_model/plastic_model.h5"
LABELS_PATH = "train_model/labels.json"
TARGET_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.7

# Construction optimization data
CONSTRUCTION_DATA = {
    "HDPE": {
        "recyclability": 95, 
        "durability": 90, 
        "cost": 60, 
        "construction_use": "Pipes, containers, outdoor furniture",
        "description": "High-Density Polyethylene - Excellent for outdoor construction due to weather resistance",
        "applications": ["Water pipes", "Gas pipes", "Outdoor furniture", "Storage containers", "Geotextiles"],
        "construction_purposes": ["Road construction", "Drainage systems", "Underground utilities", "Bridge components", "Retaining walls", "Landscaping"],
        "specific_uses": ["Road base materials", "Pipe systems", "Cable protection", "Outdoor decking", "Fencing materials"],
        "environmental_impact": "Highly recyclable, low environmental impact",
        "strength_rating": "High",
        "temperature_range": "-40°C to 60°C"
    },
    "LDPE": {
        "recyclability": 85, 
        "durability": 70, 
        "cost": 50, 
        "construction_use": "Insulation, vapor barriers, protective films",
        "description": "Low-Density Polyethylene - Flexible and lightweight, ideal for protective applications",
        "applications": ["Insulation materials", "Vapor barriers", "Protective films", "Cable insulation", "Packaging"],
        "construction_purposes": ["Building insulation", "Moisture barriers", "Concrete curing", "Foundation protection", "Roofing underlayment"],
        "specific_uses": ["Insulation boards", "Vapor barriers", "Concrete curing blankets", "Roofing membranes", "Protective wraps"],
        "environmental_impact": "Good recyclability, moderate environmental impact",
        "strength_rating": "Medium",
        "temperature_range": "-40°C to 80°C"
    },
    "PET": {
        "recyclability": 100, 
        "durability": 80, 
        "cost": 70, 
        "construction_use": "Fiber insulation, composite materials",
        "description": "Polyethylene Terephthalate - Excellent for fiber-based construction materials",
        "applications": ["Fiber insulation", "Composite materials", "Reinforcement fibers", "Textile applications", "Bottles"],
        "construction_purposes": ["Fiber-reinforced concrete", "Insulation materials", "Composite panels", "Textile reinforcements", "Sustainable building materials"],
        "specific_uses": ["Concrete reinforcement fibers", "Insulation batts", "Composite decking", "Fiber cement boards", "Geotextiles"],
        "environmental_impact": "Fully recyclable, sustainable option",
        "strength_rating": "High",
        "temperature_range": "-40°C to 70°C"
    },
    "PP": {
        "recyclability": 90, 
        "durability": 95, 
        "cost": 55, 
        "construction_use": "Pipes, fittings, geotextiles",
        "description": "Polypropylene - Superior chemical resistance and durability for construction",
        "applications": ["Water pipes", "Pipe fittings", "Geotextiles", "Carpet backing", "Automotive parts"],
        "construction_purposes": ["Plumbing systems", "Geotechnical applications", "Chemical-resistant pipes", "Carpet underlayment", "Industrial flooring"],
        "specific_uses": ["Water supply pipes", "Drainage systems", "Geotextile fabrics", "Carpet backing", "Chemical storage tanks"],
        "environmental_impact": "Highly recyclable, good environmental profile",
        "strength_rating": "Very High",
        "temperature_range": "-20°C to 100°C"
    },
    "PS": {
        "recyclability": 60, 
        "durability": 60, 
        "cost": 40, 
        "construction_use": "Insulation boards, decorative panels",
        "description": "Polystyrene - Lightweight and cost-effective for insulation applications",
        "applications": ["Insulation boards", "Decorative panels", "Packaging", "Disposable items", "Foam products"],
        "construction_purposes": ["Building insulation", "Decorative elements", "Lightweight concrete", "Packaging materials", "Temporary structures"],
        "specific_uses": ["Insulation panels", "Decorative moldings", "Foam concrete", "Protective packaging", "Signage materials"],
        "environmental_impact": "Limited recyclability, higher environmental impact",
        "strength_rating": "Low-Medium",
        "temperature_range": "-40°C to 70°C"
    },
    "PVC": {
        "recyclability": 50, 
        "durability": 85, 
        "cost": 45, 
        "construction_use": "Pipes, window frames, flooring",
        "description": "Polyvinyl Chloride - Versatile and widely used in construction industry",
        "applications": ["Water pipes", "Window frames", "Flooring", "Cable insulation", "Siding"],
        "construction_purposes": ["Plumbing systems", "Window and door frames", "Flooring materials", "Electrical conduits", "Exterior siding"],
        "specific_uses": ["Water supply pipes", "Window frames", "Vinyl flooring", "Electrical conduits", "Exterior cladding"],
        "environmental_impact": "Limited recyclability, contains chlorine",
        "strength_rating": "High",
        "temperature_range": "-10°C to 60°C"
    }
}

# Global variables
model = None
class_labels = []
camera = None
detection_history = []

def load_model_and_labels():
    global model, class_labels
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            print("🔄 Loading model... This may take 10-20 seconds on first run...")
            # Disable TensorFlow logging to speed up loading
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            model = load_model(MODEL_PATH)
            with open(LABELS_PATH, 'r') as f:
                class_labels = json.load(f)['labels']
            print(f"✅ Model loaded with {len(class_labels)} classes: {class_labels}")
            print("🚀 Model ready! Server starting...")
            return True
        else:
            print("❌ Model or labels file not found")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_plastic(image_array):
    if model is None:
        return None, 0, []
    
    try:
        # Resize and preprocess image
        img = cv2.resize(image_array, TARGET_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        # Make prediction
        pred = model.predict(arr, verbose=0)
        confidence = np.max(pred)
        predicted_class = class_labels[np.argmax(pred)]
        
        return predicted_class, confidence, pred[0].tolist()
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0, []

def calculate_optimization_score(plastic_type, confidence):
    data = CONSTRUCTION_DATA.get(plastic_type, {})
    recyclability = data.get('recyclability', 0)
    durability = data.get('durability', 0)
    cost = data.get('cost', 0)
    
    # Weighted score: 40% recyclability, 30% durability, 20% cost, 10% confidence
    score = (recyclability * 0.4 + durability * 0.3 + (100-cost) * 0.2 + confidence * 100 * 0.1)
    return min(100, max(0, score))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_plastic():
    try:
        # Get image from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Predict plastic type
        plastic_type, confidence, all_predictions = predict_plastic(image_array)
        
        if plastic_type is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Get construction data
        construction_data = CONSTRUCTION_DATA.get(plastic_type, {})
        optimization_score = calculate_optimization_score(plastic_type, confidence)
        
        # Create response
        response = {
            'plastic_type': plastic_type,
            'confidence': float(confidence),
            'all_predictions': dict(zip(class_labels, all_predictions)),
            'construction_data': construction_data,
            'optimization_score': float(optimization_score),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to detection history
        detection_history.append(response)
        if len(detection_history) > 50:  # Keep only last 50 detections
            detection_history.pop(0)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    return jsonify(detection_history)

@app.route('/api/stats')
def get_stats():
    if not detection_history:
        return jsonify({'total_detections': 0, 'by_type': {}})
    
    # Calculate statistics
    total_detections = len(detection_history)
    by_type = {}
    
    for detection in detection_history:
        plastic_type = detection['plastic_type']
        if plastic_type not in by_type:
            by_type[plastic_type] = 0
        by_type[plastic_type] += 1
    
    # Calculate percentages
    for plastic_type in by_type:
        by_type[plastic_type] = {
            'count': by_type[plastic_type],
            'percentage': round((by_type[plastic_type] / total_detections) * 100, 1)
        }
    
    return jsonify({
        'total_detections': total_detections,
        'by_type': by_type
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model_and_labels():
        print("🚀 Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please train the model first.")
