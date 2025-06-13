import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image
import sys
import traceback

# Configure page - must be first Streamlit command
st.set_page_config(
    page_title="ASL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress TensorFlow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'mp_initialized' not in st.session_state:
    st.session_state.mp_initialized = False

# Actions that the model can predict
ACTIONS = np.array(['hello', 'thanks', 'iloveyou', 'night', 'please', 'help', 'life', 'no', 'yes'])
CONFIDENCE_THRESHOLD = 0.5
SEQUENCE_LENGTH = 30

# Initialize MediaPipe - cached to avoid reinitialization
@st.cache_resource
def initialize_mediapipe():
    """Initialize MediaPipe components"""
    try:
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        return mp_holistic, mp_drawing, mp_face_mesh, True
    except Exception as e:
        st.error(f"Failed to initialize MediaPipe: {e}")
        return None, None, None, False

# Load MediaPipe
mp_holistic, mp_drawing, mp_face_mesh, mp_success = initialize_mediapipe()

@st.cache_resource
def load_asl_model(model_path=None):
    """Load the trained ASL model"""
    try:
        # Try different possible model locations
        possible_paths = [
            'action.h5',
            './action.h5',
            'models/action.h5',
            model_path
        ]
        
        model_path_used = None
        for path in possible_paths:
            if path and os.path.exists(path):
                model_path_used = path
                break
        
        if model_path_used:
            # Configure TensorFlow for Streamlit Cloud
            tf.get_logger().setLevel('ERROR')
            model = load_model(model_path_used, compile=False)
            return model, True, f"Model loaded successfully from {model_path_used}"
        else:
            return None, False, "Model file not found. Please upload your trained model."
            
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        st.error(error_msg)
        # Print full traceback for debugging
        traceback.print_exc()
        return None, False, error_msg

def mediapipe_detection(image, holistic_model):
    """Process image with MediaPipe"""
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Make prediction
        results = holistic_model.process(image_rgb)
        
        # Convert back to BGR
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr, results
    except Exception as e:
        st.error(f"MediaPipe detection error: {e}")
        return image, None

def draw_styled_landmarks(image, results):
    """Draw landmarks on image with error handling"""
    try:
        if not mp_drawing or not mp_face_mesh or not mp_holistic:
            return image
            
        # Draw face landmarks
        if results and results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )
        
        # Draw pose landmarks
        if results and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
        
        # Draw hand landmarks
        if results and results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
        
        if results and results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        return image
    except Exception as e:
        st.warning(f"Could not draw landmarks: {e}")
        return image

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    try:
        # Pose keypoints (33 landmarks * 4 values each)
        if results and results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility] 
                           for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33*4)
        
        # Face keypoints (468 landmarks * 3 values each)
        if results and results.face_landmarks:
            face = np.array([[res.x, res.y, res.z] 
                           for res in results.face_landmarks.landmark]).flatten()
        else:
            face = np.zeros(468*3)
        
        # Left hand keypoints (21 landmarks * 3 values each)
        if results and results.left_hand_landmarks:
            left_hand = np.array([[res.x, res.y, res.z] 
                                for res in results.left_hand_landmarks.landmark]).flatten()
        else:
            left_hand = np.zeros(21*3)
        
        # Right hand keypoints (21 landmarks * 3 values each)
        if results and results.right_hand_landmarks:
            right_hand = np.array([[res.x, res.y, res.z] 
                                 for res in results.right_hand_landmarks.landmark]).flatten()
        else:
            right_hand = np.zeros(21*3)
        
        return np.concatenate([pose, face, left_hand, right_hand])
    
    except Exception as e:
        st.error(f"Error extracting keypoints: {e}")
        # Return zeros if extraction fails
        return np.zeros(33*4 + 468*3 + 21*3 + 21*3)

def process_image_for_asl(image, model):
    """Process image for ASL recognition with comprehensive error handling"""
    try:
        if not mp_success:
            return None, "MediaPipe not initialized", None, None
            
        if model is None:
            return None, "Model not loaded", None, None
        
        # Convert PIL Image to OpenCV format
        if isinstance(image, Image.Image):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
        
        # Initialize MediaPipe Holistic
        with mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1  # Use lighter model for cloud deployment
        ) as holistic:
            
            # Make detections
            image_processed, results = mediapipe_detection(image_cv, holistic)
            
            if results is None:
                return image_processed, "No landmarks detected", None, None
            
            # Draw landmarks
            image_processed = draw_styled_landmarks(image_processed, results)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            
            # Update sequence
            st.session_state.sequence.append(keypoints)
            st.session_state.sequence = st.session_state.sequence[-SEQUENCE_LENGTH:]
            
            predicted_word = ""
            probabilities = None
            
            # Make prediction if we have enough frames
            if len(st.session_state.sequence) == SEQUENCE_LENGTH:
                try:
                    # Prepare input for model
                    sequence_array = np.expand_dims(st.session_state.sequence, axis=0)
                    
                    # Make prediction
                    probabilities = model.predict(sequence_array, verbose=0)[0]
                    
                    # Get prediction
                    predicted_idx = np.argmax(probabilities)
                    confidence = probabilities[predicted_idx]
                    
                    # Only show prediction if confidence is above threshold
                    if confidence > CONFIDENCE_THRESHOLD:
                        predicted_word = ACTIONS[predicted_idx]
                    
                    st.session_state.predictions.append(predicted_idx)
                    
                except Exception as pred_error:
                    st.error(f"Prediction error: {pred_error}")
                    probabilities = np.zeros(len(ACTIONS))
            
            # Add prediction text to image
            try:
                cv2.rectangle(image_processed, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image_processed, predicted_word, (3, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass  # Continue even if text overlay fails
            
            return image_processed, predicted_word, probabilities, results
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        st.error(error_msg)
        traceback.print_exc()
        return None, error_msg, None, None

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤟 ASL Recognition App")
    st.markdown("**Upload an image or use your webcam to recognize American Sign Language gestures!**")
    
    # Sidebar for model management and info
    with st.sidebar:
        st.header("Model Status")
        
        # Try to load model automatically
        if not st.session_state.model_loaded:
            with st.spinner("Loading model..."):
                model, model_loaded, model_message = load_asl_model()
                st.session_state.model = model
                st.session_state.model_loaded = model_loaded
        
        # Display model status
        if st.session_state.model_loaded:
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Model not loaded")
            
            # Model upload option
            st.subheader("Upload Model")
            uploaded_model = st.file_uploader(
                "Upload your action.h5 model file", 
                type=['h5'],
                help="Upload your trained ASL recognition model"
            )
            
            if uploaded_model is not None:
                with st.spinner("Loading uploaded model..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                        tmp_file.write(uploaded_model.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Clear cache and load new model
                        st.cache_resource.clear()
                        model, model_loaded, model_message = load_asl_model(tmp_file_path)
                        
                        if model_loaded:
                            st.session_state.model = model
                            st.session_state.model_loaded = True
                            st.success("Model uploaded and loaded successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to load model: {model_message}")
                            
                    except Exception as e:
                        st.error(f"Error loading uploaded model: {str(e)}")
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
        
        # Display recognizable signs
        st.subheader("Recognizable Signs")
        for i, action in enumerate(ACTIONS):
            st.write(f"• **{action.upper()}**")
        
        # Settings
        st.subheader("Settings")
        if st.button("Clear Session Data"):
            st.session_state.sequence = []
            st.session_state.predictions = []
            st.success("Session data cleared!")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🖼️ Upload Image", "📊 About"])
    
    with tab1:
        st.subheader("Webcam Recognition")
        
        if not st.session_state.model_loaded:
            st.warning("Please load a model first using the sidebar.")
            return
        
        # Camera input
        picture = st.camera_input("Take a picture for ASL recognition")
        
        if picture is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(picture, caption="Original Image", use_column_width=True)
            
            with col2:
                with st.spinner("Processing image..."):
                    # Load and process image
                    image = Image.open(picture)
                    
                    processed_image, prediction, probabilities, results = process_image_for_asl(
                        image, st.session_state.model
                    )
                    
                    if processed_image is not None:
                        # Convert back to RGB for display
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)
                        
                        # Show prediction
                        if prediction and prediction != "No landmarks detected":
                            st.success(f"🎯 **Predicted Sign: {prediction.upper()}**")
                        else:
                            st.info("No confident prediction")
                        
                        # Show probabilities chart
                        if probabilities is not None:
                            st.subheader("Prediction Confidence")
                            prob_data = {action: float(prob) for action, prob in zip(ACTIONS, probabilities)}
                            st.bar_chart(prob_data)
                    else:
                        st.error("Failed to process image")
    
    with tab2:
        st.subheader("Upload Image Recognition")
        
        if not st.session_state.model_loaded:
            st.warning("Please load a model first using the sidebar.")
            return
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing ASL gestures"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                with st.spinner("Processing image..."):
                    processed_image, prediction, probabilities, results = process_image_for_asl(
                        image, st.session_state.model
                    )
                    
                    if processed_image is not None:
                        # Convert back to RGB for display
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)
                        
                        # Show prediction
                        if prediction and prediction != "No landmarks detected":
                            st.success(f"🎯 **Predicted Sign: {prediction.upper()}**")
                        else:
                            st.info("No confident prediction")
                        
                        # Show probabilities chart
                        if probabilities is not None:
                            st.subheader("Prediction Confidence")
                            prob_data = {action: float(prob) for action, prob in zip(ACTIONS, probabilities)}
                            st.bar_chart(prob_data)
                    else:
                        st.error("Failed to process image")
    
    with tab3:
        st.subheader("About This App")
        
        st.markdown("""
        This American Sign Language (ASL) recognition app uses computer vision and machine learning to identify sign language gestures.
        
        **How it works:**
        1. **MediaPipe**: Detects hand, pose, and face landmarks
        2. **Deep Learning**: Uses a trained neural network to classify gestures
        3. **Sequence Analysis**: Analyzes multiple frames for accurate recognition
        
        **Supported Signs:**
        """)
        
        # Display signs in a grid
        cols = st.columns(3)
        for i, action in enumerate(ACTIONS):
            with cols[i % 3]:
                st.info(f"**{action.upper()}**")
        
        st.markdown(f"""
        **Technical Details:**
        - Model Type: Sequential Neural Network
        - Input Features: {33*4 + 468*3 + 21*3 + 21*3} keypoints per frame
        - Sequence Length: {SEQUENCE_LENGTH} frames
        - Confidence Threshold: {CONFIDENCE_THRESHOLD}
        - Supported Formats: JPG, JPEG, PNG
        """)
        
        # System info
        with st.expander("System Information"):
            st.write(f"Python Version: {sys.version}")
            st.write(f"TensorFlow Version: {tf.__version__}")
            st.write(f"OpenCV Available: {cv2.__version__}")
            st.write(f"MediaPipe Available: {mp_success}")
            st.write(f"Model Loaded: {st.session_state.model_loaded}")

if __name__ == "__main__":
    main()
