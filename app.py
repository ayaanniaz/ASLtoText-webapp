from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import base64
from PIL import Image
import io
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Disable GPU for TensorFlow and MediaPipe
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Initialize MediaPipe with CPU-only configuration
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load the trained model with error handling
model = None
try:
    # Try to load with custom objects if needed
    model = load_model('action.h5', compile=False)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Try alternative loading methods
    try:
        model = tf.keras.models.load_model('action.h5', compile=False)
        logger.info("Model loaded with alternative method!")
    except Exception as e2:
        logger.error(f"Alternative loading also failed: {e2}")
        model = None

# Actions that the model can predict
actions = np.array(['hello', 'thanks', 'iloveyou', 'night', 'please', 'help', 'life', 'no', 'yes'])

# Global variables for prediction
sequence = []
predictions = []
threshold = 0.5

def mediapipe_detection(image, model):
    """Process image with MediaPipe"""
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    except Exception as e:
        logger.error(f"Error in MediaPipe detection: {e}")
        return image, None

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    except Exception as e:
        logger.error(f"Error extracting keypoints: {e}")
        return np.zeros(33*4 + 468*3 + 21*3 + 21*3)  # Return zeros if extraction fails

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global sequence, predictions
    
    try:
        # Get the base64 image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'prediction': '',
                'confidence': 0,
                'status': 'error',
                'error': 'No image data received'
            })
        
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize MediaPipe with CPU-only settings
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,  # Use lighter model
            static_image_mode=False,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False
        )
        
        predicted_word = ""
        confidence = 0
        
        try:
            # Make detections
            image, results = mediapipe_detection(image, holistic)
            
            if model is not None and results:
                # Extract keypoints
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep only last 30 frames
                
                if len(sequence) == 30:
                    # Make prediction
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    predictions.append(np.argmax(res))
                    
                    # Only show if confidence is high enough
                    if res[np.argmax(res)] > threshold:
                        predicted_word = actions[np.argmax(res)]
                        confidence = float(res[np.argmax(res)])
        
        except Exception as prediction_error:
            logger.error(f"Error in prediction: {prediction_error}")
        
        finally:
            # Always close holistic to prevent memory leaks
            holistic.close()
        
        return jsonify({
            'prediction': predicted_word,
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return jsonify({
            'prediction': '',
            'confidence': 0,
            'status': 'error',
            'error': str(e)
        })

@app.route('/reset_sequence', methods=['POST'])
def reset_sequence():
    global sequence, predictions
    try:
        sequence = []
        predictions = []
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error resetting sequence: {e}")
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
