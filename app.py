import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image

# Configure page
st.set_page_config(
    page_title="ASL Recognition",
    page_icon="🤟",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Actions that the model can predict
actions = np.array(['hello', 'thanks', 'iloveyou', 'night', 'please', 'help', 'life', 'no', 'yes'])
threshold = 0.5

# Initialize MediaPipe
@st.cache_resource
def initialize_mediapipe():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    return mp_holistic, mp_drawing, mp_face_mesh

mp_holistic, mp_drawing, mp_face_mesh = initialize_mediapipe()

@st.cache_resource
def load_asl_model():
    """Load the trained ASL model"""
    try:
        if os.path.exists('action.h5'):
            model = load_model('action.h5')
            return model, True, "Model loaded successfully!"
        else:
            return None, False, "Model file 'action.h5' not found. Please upload your trained model."
    except Exception as e:
        return None, False, f"Error loading model: {str(e)}"

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

def process_image(image, model):
    """Process image for ASL recognition"""
    try:
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize MediaPipe
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Make detections
            image_processed, results = mediapipe_detection(image_cv, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image_processed, results)
            
            predicted_word = ""
            colors = [(245,117,16), (117,245,16), (16,117,245)]
            
            if model is not None:
                # Prediction logic
                keypoints = extract_keypoints(results)
                st.session_state.sequence.append(keypoints)
                st.session_state.sequence = st.session_state.sequence[-30:]  # Keep only last 30 frames
                
                if len(st.session_state.sequence) == 30:
                    res = model.predict(np.expand_dims(st.session_state.sequence, axis=0))[0]
                    st.session_state.predictions.append(np.argmax(res))
                    
                    # Only show if confidence is high enough
                    if res[np.argmax(res)] > threshold:
                        predicted_word = actions[np.argmax(res)]
                    
                    # Viz probabilities
                    image_processed = prob_viz(res, actions, image_processed, colors)
                    
                    return image_processed, predicted_word, res
            
            # Display predicted word
            cv2.rectangle(image_processed, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image_processed, predicted_word, (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return image_processed, predicted_word, None
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, "", None

# Main App
def main():
    st.title("🤟 ASL Recognition App")
    st.write("Upload an image or use your webcam to recognize American Sign Language gestures!")
    
    # Load model
    model, model_loaded, model_message = load_asl_model()
    st.session_state.model = model
    
    # Display model status
    if model_loaded:
        st.success(model_message)
    else:
        st.error(model_message)
        
        # Model upload option
        st.subheader("Upload Model")
        uploaded_model = st.file_uploader("Upload your action.h5 model file", type=['h5'])
        if uploaded_model is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                model = load_model(tmp_file_path)
                st.session_state.model = model
                st.success("Model uploaded and loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading uploaded model: {str(e)}")
            finally:
                os.unlink(tmp_file_path)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🖼️ Upload Image", "📊 Live Demo"])
    
    with tab1:
        st.subheader("Webcam Input")
        
        # Webcam input
        picture = st.camera_input("Take a picture for ASL recognition")
        
        if picture is not None and st.session_state.model is not None:
            # Process the image
            image = Image.open(picture)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                with st.spinner("Processing..."):
                    processed_image, prediction, probabilities = process_image(image, st.session_state.model)
                    
                    if processed_image is not None:
                        # Convert back to RGB for display
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)
                        
                        if prediction:
                            st.success(f"Predicted Sign: **{prediction.upper()}**")
                        else:
                            st.info("No confident prediction")
                            
                        # Show probabilities
                        if probabilities is not None:
                            st.subheader("Prediction Probabilities")
                            prob_data = {action: float(prob) for action, prob in zip(actions, probabilities)}
                            st.bar_chart(prob_data)
    
    with tab2:
        st.subheader("Upload Image")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None and st.session_state.model is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                with st.spinner("Processing..."):
                    processed_image, prediction, probabilities = process_image(image, st.session_state.model)
                    
                    if processed_image is not None:
                        # Convert back to RGB for display
                        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)
                        
                        if prediction:
                            st.success(f"Predicted Sign: **{prediction.upper()}**")
                        else:
                            st.info("No confident prediction")
                            
                        # Show probabilities
                        if probabilities is not None:
                            st.subheader("Prediction Probabilities")
                            prob_data = {action: float(prob) for action, prob in zip(actions, probabilities)}
                            st.bar_chart(prob_data)
    
    with tab3:
        st.subheader("About the Model")
        st.write(f"This ASL recognition model can predict the following signs:")
        
        # Display actions in a nice grid
        cols = st.columns(3)
        for i, action in enumerate(actions):
            with cols[i % 3]:
                st.info(f"**{action.upper()}**")
        
        st.write(f"**Model Details:**")
        st.write(f"- Total Actions: {len(actions)}")
        st.write(f"- Confidence Threshold: {threshold}")
        st.write(f"- Sequence Length: 30 frames")
        st.write(f"- Model Status: {'✅ Loaded' if st.session_state.model else '❌ Not Loaded'}")
        
        # Clear session button
        if st.button("Clear Session Data"):
            st.session_state.sequence = []
            st.session_state.predictions = []
            st.success("Session data cleared!")

if __name__ == "__main__":
    main()
