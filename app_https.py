from flask import Flask, render_template, request, jsonify
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
import ssl
import sqlite3

app = Flask(__name__)

# Configuration
MODEL_PATH = "train_model/plastic_model.h5"
LABELS_PATH = "train_model/labels.json"
TARGET_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.1
DATABASE_PATH = "plastic_detections.db"

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
    },
    "Unknown": {
        "recyclability": 0, 
        "durability": 0, 
        "cost": 0, 
        "construction_use": "Unknown plastic type",
        "description": "Unknown plastic type - Unable to identify with sufficient confidence",
        "applications": ["Unknown applications"],
        "construction_purposes": ["Unknown construction purposes"],
        "specific_uses": ["Unknown specific uses"],
        "environmental_impact": "Unknown environmental impact",
        "strength_rating": "Unknown",
        "temperature_range": "Unknown"
    }
}

# Global variables
model = None
class_labels = []

def init_database():
    """Initialize the SQLite database and create tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plastic_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            optimization_score REAL NOT NULL,
            recyclability INTEGER NOT NULL,
            durability INTEGER NOT NULL,
            cost INTEGER NOT NULL,
            construction_use TEXT NOT NULL,
            description TEXT NOT NULL,
            applications TEXT NOT NULL,
            construction_purposes TEXT NOT NULL,
            specific_uses TEXT NOT NULL,
            environmental_impact TEXT NOT NULL,
            strength_rating TEXT NOT NULL,
            temperature_range TEXT NOT NULL,
            all_predictions TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def save_detection(data):
    """Save detection data to database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO detections (
            plastic_type, confidence, optimization_score, recyclability, durability, cost,
            construction_use, description, applications, construction_purposes, specific_uses,
            environmental_impact, strength_rating, temperature_range, all_predictions, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['plastic_type'],
        data['confidence'],
        data['optimization_score'],
        data['construction_data']['recyclability'],
        data['construction_data']['durability'],
        data['construction_data']['cost'],
        data['construction_data']['construction_use'],
        data['construction_data']['description'],
        json.dumps(data['construction_data']['applications']),
        json.dumps(data['construction_data']['construction_purposes']),
        json.dumps(data['construction_data']['specific_uses']),
        data['construction_data']['environmental_impact'],
        data['construction_data']['strength_rating'],
        data['construction_data']['temperature_range'],
        json.dumps(data['all_predictions']),
        data['timestamp']
    ))
    
    conn.commit()
    conn.close()
    print(f"Detection saved: {data['plastic_type']} ({data['confidence']:.3f})")

def get_all_detections():
    """Get all detection data from database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM detections ORDER BY timestamp DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    detections = []
    for row in rows:
        detection = {
            'id': row[0],
            'plastic_type': row[1],
            'confidence': row[2],
            'optimization_score': row[3],
            'recyclability': row[4],
            'durability': row[5],
            'cost': row[6],
            'construction_use': row[7],
            'description': row[8],
            'applications': json.loads(row[9]),
            'construction_purposes': json.loads(row[10]),
            'specific_uses': json.loads(row[11]),
            'environmental_impact': row[12],
            'strength_rating': row[13],
            'temperature_range': row[14],
            'all_predictions': json.loads(row[15]),
            'timestamp': row[16]
        }
        detections.append(detection)
    
    return detections

def load_model_and_labels():
    global model, class_labels
    try:
        # Initialize database first
        init_database()
        
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            model = load_model(MODEL_PATH)
            with open(LABELS_PATH, 'r') as f:
                class_labels = json.load(f)['labels']
            print(f"Model loaded with {len(class_labels)} classes: {class_labels}")
            return True
        else:
            print("Model or labels file not found")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_plastic(image_array):
    if model is None:
        return None, 0, []
    
    try:
        # Enhanced preprocessing
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Already RGB
            img = image_array
        else:
            img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Resize with better interpolation
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Normalize
        arr = img_to_array(img) / 255.0
        
        # Add slight augmentation for better accuracy
        arr = np.clip(arr, 0, 1)
        
        arr = np.expand_dims(arr, axis=0)
        
        # Make prediction
        pred = model.predict(arr, verbose=0)
        confidence = np.max(pred)
        predicted_class = class_labels[np.argmax(pred)]
        
        # Debug output
        print(f"Prediction: {predicted_class}, Confidence: {confidence:.3f}")
        print(f"All predictions: {dict(zip(class_labels, pred[0]))}")
        
        # Only return prediction if confidence is above threshold
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Low confidence ({confidence:.3f}), returning Unknown")
            return "Unknown", confidence, pred[0].tolist()
        
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

@app.route('/data')
def view_data():
    return render_template('data.html')

@app.route('/api/data')
def get_data():
    try:
        detections = get_all_detections()
        return jsonify({
            'success': True,
            'detections': detections,
            'total_count': len(detections)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
        
        # Ensure all required arrays exist
        if 'applications' not in construction_data:
            construction_data['applications'] = []
        if 'construction_purposes' not in construction_data:
            construction_data['construction_purposes'] = []
        if 'specific_uses' not in construction_data:
            construction_data['specific_uses'] = []
        
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
        
        # Save detection to database
        save_detection(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model_and_labels():
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to load model. Please train the model first.")
