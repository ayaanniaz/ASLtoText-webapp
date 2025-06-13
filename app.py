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

app = Flask(__name__)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load the trained model
try:
    model = load_model('action.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Actions that the model can predict
actions = np.array(['hello', 'thanks', 'iloveyou', 'night', 'please', 'help', 'life', 'no', 'yes'])

# Global variables for prediction
sequence = []
predictions = []
threshold = 0.5

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global sequence, predictions
    
    try:
        # Get the base64 image data from the request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize MediaPipe
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Make detections
        image, results = mediapipe_detection(image, holistic)
        
        predicted_word = ""
        confidence = 0
        
        if model is not None and results:
            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep only last 30 frames
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                # Only show if confidence is high enough
                if res[np.argmax(res)] > threshold:
                    predicted_word = actions[np.argmax(res)]
                    confidence = float(res[np.argmax(res)])
        
        holistic.close()
        
        return jsonify({
            'prediction': predicted_word,
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({
            'prediction': '',
            'confidence': 0,
            'status': 'error',
            'error': str(e)
        })

@app.route('/reset_sequence', methods=['POST'])
def reset_sequence():
    global sequence, predictions
    sequence = []
    predictions = []
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)