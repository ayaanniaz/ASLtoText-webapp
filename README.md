# ASL to Text Recognition Web App

A real-time American Sign Language (ASL) recognition system that converts sign language gestures into text using deep learning and computer vision.

## Features

- **Real-time Recognition**: Live webcam feed with instant ASL sign detection
- **9 ASL Signs Supported**: hello, thanks, iloveyou, night, please, help, life, no, yes
- **MediaPipe Integration**: Advanced hand, pose, and face landmark detection
- **LSTM Neural Network**: Deep learning model for sequence recognition
- **Confidence Threshold**: Only displays predictions above 50% confidence
- **Learning Resources**: YouTube links for each supported ASL sign

## Model Architecture

- **Input**: 1662-dimensional feature vector (pose + face + hand landmarks)
- **Sequence Length**: 30 frames for temporal modeling
- **Network**: LSTM layers (128 → 64 units) with Dense layers
- **Output**: 9 classes with softmax activation
- **Framework**: TensorFlow/Keras

## Project Structure

```
asl-recognition/
├── app.py                 # Flask application
├── action.h5             # Trained LSTM model
├── requirements.txt      # Python dependencies
├── render.yaml          # Render deployment config
├── Dockerfile           # Docker configuration
├── templates/
│   └── index.html       # Web interface
└── README.md           # Project documentation
```

## Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayaanniaz/ASLtoText-webapp.git
   cd asl-recognition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your trained model**
   - Place your `action.h5` model file in the root directory

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the app**
   - Open http://localhost:5000 in your browser
   - Allow camera permissions when prompted

### Deployment on Render

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Render**
   - Connect your GitHub repository to Render
   - Use the provided `render.yaml` configuration
   - Render will automatically deploy your app

3. **Upload Model File**
   - Due to GitHub's file size limits, upload `action.h5` directly to Render
   - Or use Git LFS for large files

## Usage Instructions

1. **Camera Setup**: Ensure good lighting and position yourself clearly in the camera frame
2. **Sign Performance**: Perform ASL signs slowly and clearly
3. **Wait for Recognition**: The model needs 30 frames (about 1 second) to make a prediction
4. **Confidence Display**: Only predictions above 50% confidence are shown
5. **Learning**: Use the provided YouTube links to learn proper ASL signs

## Model Training Details

- **Dataset**: Custom collected sequences (30 samples per sign)
- **Preprocessing**: MediaPipe keypoint extraction
- **Training**: 500 epochs with early stopping
- **Validation**: 5% test split
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score


**Note**: Webcam access requires HTTPS in production environments.

## Performance Optimization

- Uses `opencv-python-headless` for production deployment
- Optimized model inference with batch processing
- Efficient keypoint extraction and visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- MediaPipe team for pose estimation
- TensorFlow team for deep learning framework
- ASL community for sign language resources

## Future Enhancements

- [ ] Add more ASL signs
- [ ] Improve model accuracy
- [ ] Add sentence construction
