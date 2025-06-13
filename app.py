from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import base64
import os
from PIL import Image
import io

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load the trained model
model = None
try:
    if os.path.exists('action.h5'):
        model = load_model('action.h5')
        print("Model loaded successfully!")
    else:
        print("Model file 'action.h5' not found. Please upload your trained model.")
except Exception as e:
    print(f"Error loading model: {e}")

# Actions that the model can predict
actions = np.array(['hello', 'thanks', 'iloveyou', 'night', 'please', 'help', 'life', 'no', 'yes'])

# Global variables for prediction
sequence = []
predictions = []
threshold = 0.5

def mediapipe_detection(image, model):
    """Process image with MediaPipe"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """Draw landmarks on image"""
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    # Draw right hand connections  
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    """Visualize prediction probabilities"""
    output_frame = input_frame.copy()
    for num, (prob, action) in enumerate(zip(res, actions)):
        prob = float(prob)
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, action, (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def process_frame(image_data):
    """Process a single frame for ASL recognition"""
    global sequence, predictions
    
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, "Error: Could not decode image"
        
        # Initialize MediaPipe
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Make detections
            image, results = mediapipe_detection(image, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            predicted_word = ""
            colors = [(245,117,16), (117,245,16), (16,117,245)]
            
            if model is not None:
                # Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep only last 30 frames
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    
                    # Only show if confidence is high enough
                    if res[np.argmax(res)] > threshold:
                        predicted_word = actions[np.argmax(res)]
                    
                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
            
            # Display predicted word
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, predicted_word, (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Encode processed image back to base64
        _, buffer = cv2.imencode('.jpg', image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return processed_image_base64, predicted_word
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None, f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    """Process a single frame sent from client"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        processed_image, prediction = process_frame(image_data)
        
        if processed_image is None:
            return jsonify({'error': prediction}), 500
        
        return jsonify({
            'processed_image': f"data:image/jpeg;base64,{processed_image}",
            'prediction': prediction,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_prediction')
def get_prediction():
    """Get the latest prediction"""
    global predictions
    if predictions:
        latest_prediction = actions[predictions[-1]] if predictions[-1] < len(actions) else ""
        return jsonify({'prediction': latest_prediction})
    return jsonify({'prediction': ''})

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
