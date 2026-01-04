# Sign Language Conversation App

A real-time sign language recognition and conversation application that enables communication between sign language users and text-based communicators. The application uses computer vision and machine learning to detect and classify hand gestures into meaningful words and phrases.

## 🌟 Features

- **Real-time Hand Gesture Recognition**: Detects and classifies hand signs using MediaPipe and TensorFlow
- **Dual Interface Support**:
  - **OpenCV Interface** (`app.py`): Standalone application with camera feed
  - **Streamlit Web Interface** (`streamlit_app.py`): Interactive web-based application with split-screen conversation layout
- **Two-way Communication**: Facilitates conversation between sign language users and text-based users
- **Multiple Gesture Support**: Recognizes hand signs including "Hi", "How", "you", "are", and "Peace"
- **Finger Movement Tracking**: Tracks and classifies finger movements (Stop, Clockwise, Counter Clockwise, Move)
- **Real-time FPS Display**: Shows performance metrics
- **Data Collection Mode**: Allows users to contribute training data for new gestures

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.x**: Primary programming language
- **OpenCV (cv2)**: Computer vision and image processing
- **MediaPipe**: Hand landmark detection and tracking
- **TensorFlow/Keras**: Deep learning framework for gesture classification
- **Streamlit**: Web application framework

### Machine Learning

- **TensorFlow Lite**: Optimized inference for hand gesture classification
- **Custom Neural Networks**: 
  - Keypoint Classifier: Classifies static hand poses
  - Point History Classifier: Classifies dynamic finger movements

### Data Processing

- **NumPy**: Numerical computations and array operations
- **Pandas**: Dataset handling and manipulation
- **scikit-learn**: Machine learning utilities and preprocessing
- **Matplotlib**: Visualization for model training and debugging

### Key Libraries

```
mediapipe==0.10.9          # Hand landmark detection
opencv-python>=4.5.3       # Computer vision
tensorflow>=2.11.0         # Deep learning
streamlit>=1.28.0          # Web interface
numpy>=1.23.0             # Numerical operations
scikit-learn>=1.2.0       # ML utilities
pandas>=1.5.0             # Data handling
matplotlib>=3.7.0         # Visualization
protobuf==3.20.3          # Protocol buffers
```

## 📐 Architecture

### System Flow

1. **Camera Input**: Captures video frames from webcam
2. **Hand Detection**: MediaPipe detects hand landmarks (21 keypoints per hand)
3. **Preprocessing**: Converts landmarks to normalized relative coordinates
4. **Classification**: 
   - Static poses → Keypoint Classifier → Word/Sign
   - Finger movements → Point History Classifier → Gesture
5. **Output**: Displays recognized gestures with bounding boxes and labels

### Model Architecture

#### Keypoint Classifier
- **Input**: 42 normalized coordinates (21 landmarks × 2D)
- **Output**: Classification of hand pose (5 classes)
- **Model Format**: TensorFlow Lite (.tflite)
- **Labels**: Hi, How, you, are, Peace

#### Point History Classifier
- **Input**: Time-series of fingertip positions (16 frames)
- **Output**: Classification of finger movement (4 classes)
- **Model Format**: TensorFlow Lite (.tflite)
- **Labels**: Stop, Clockwise, Counter Clockwise, Move

### Data Flow

```
Camera → MediaPipe → Landmarks → Preprocessing → Classifiers → Display
                                      ↓
                                 Data Logging (optional)
```

## 📂 Project Structure

```
Sign_Language_Conversation_app/
│
├── app.py                          # OpenCV-based standalone application
├── streamlit_app.py                # Streamlit web-based application
├── requirements.txt                # Python dependencies
│
├── model/                          # Machine learning models
│   ├── __init__.py
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier.py           # Keypoint classifier implementation
│   │   ├── keypoint_classifier.tflite       # TFLite model for hand poses
│   │   ├── keypoint_classifier.keras        # Keras model (alternative)
│   │   ├── keypoint_classifier.hdf5         # HDF5 model (alternative)
│   │   ├── keypoint_classifier_label.csv    # Class labels
│   │   └── keypoint.csv                     # Training data
│   │
│   └── point_history_classifier/
│       ├── point_history_classifier.py      # Movement classifier implementation
│       ├── point_history_classifier.tflite  # TFLite model for movements
│       ├── point_history_classifier.hdf5    # HDF5 model (alternative)
│       ├── point_history_classifier_label.csv # Class labels
│       └── point_history.csv                # Training data
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   └── cvfpscalc.py               # FPS calculation utility
│
├── keypoint_classification.ipynb   # Model training notebook
├── keypoint_classification_EN.ipynb # English version notebook
├── point_history_classification.ipynb # Movement model training
│
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- pip package manager

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Hari-Padmesh/Sign_Language_Conversation_app.git
   cd Sign_Language_Conversation_app
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Option 1: OpenCV Application (app.py)

Run the standalone application with OpenCV interface:

```bash
python app.py
```

**Command-line Arguments:**
- `--device`: Camera device number (default: 0)
- `--width`: Camera width (default: 960)
- `--height`: Camera height (default: 540)
- `--use_static_image_mode`: Use static image mode
- `--min_detection_confidence`: Minimum detection confidence (default: 0.7)
- `--min_tracking_confidence`: Minimum tracking confidence (default: 0.5)

**Example:**
```bash
python app.py --device 0 --width 1280 --height 720
```

**Keyboard Controls:**
- `ESC`: Exit application
- `k`: Enter keypoint logging mode
- `h`: Enter point history logging mode
- `n`: Normal mode
- `0-9`: Enter number for data logging

### Option 2: Streamlit Web Application (streamlit_app.py)

Run the interactive web-based application:

```bash
streamlit run streamlit_app.py
```

This will open a web browser with:
- **Person 1 (Sign Language)**: Camera feed with gesture recognition
- **Person 2 (Text Input)**: Text area for typing messages
- Real-time sentence building from detected signs
- Clear buttons for resetting conversation

**Features:**
- Split-screen interface for two-way communication
- Automatic sentence building with cooldown (2 seconds between gestures)
- Clear sentence and text options
- Start/Stop camera control

## 🎓 Training Custom Models

The repository includes Jupyter notebooks for training custom gesture recognition models:

1. **Keypoint Classification**:
   - `keypoint_classification.ipynb`: Train models for static hand poses
   - `keypoint_classification_EN.ipynb`: English documentation version

2. **Point History Classification**:
   - `point_history_classification.ipynb`: Train models for dynamic movements

### Training Workflow:

1. **Collect Data**:
   - Run the application in logging mode (`k` for keypoints, `h` for history)
   - Press numbers 0-9 to assign labels while performing gestures
   - Data is saved to CSV files in the model directories

2. **Train Model**:
   - Open the relevant Jupyter notebook
   - Run cells to preprocess data and train the neural network
   - Export model to TensorFlow Lite format

3. **Update Labels**:
   - Edit the `*_label.csv` files to match your custom gestures

## 🧠 Model Details

### Keypoint Classifier

- **Purpose**: Recognize static hand shapes/poses
- **Input Processing**:
  1. Extract 21 hand landmarks from MediaPipe
  2. Convert to relative coordinates (base = wrist)
  3. Flatten to 1D array (42 values)
  4. Normalize by maximum absolute value
- **Current Classes**: Hi, How, you, are, Peace

### Point History Classifier

- **Purpose**: Recognize dynamic finger movements
- **Input Processing**:
  1. Track index fingertip position over 16 frames
  2. Convert to relative coordinates
  3. Normalize by image dimensions
  4. Flatten to 1D array (32 values)
- **Current Classes**: Stop, Clockwise, Counter Clockwise, Move

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Add New Gestures**:
   - Use logging mode to collect training data
   - Train models using provided notebooks
   - Submit pull request with updated models

2. **Improve Accuracy**:
   - Contribute more training samples
   - Optimize model architectures
   - Enhance preprocessing techniques

3. **Add Features**:
   - Text-to-speech for recognized signs
   - Sign language to text translation
   - Multi-language support
   - Gesture recording and playback

4. **Bug Fixes**:
   - Report issues on GitHub
   - Submit pull requests with fixes

## 📝 License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute as needed.

## 🙏 Acknowledgments

- **MediaPipe**: For providing excellent hand tracking capabilities
- **TensorFlow**: For the machine learning framework
- **OpenCV**: For computer vision utilities
- **Streamlit**: For the web application framework

## 📧 Contact

For questions, suggestions, or collaboration:
- GitHub: [Hari-Padmesh](https://github.com/Hari-Padmesh)
- Repository: [Sign_Language_Conversation_app](https://github.com/Hari-Padmesh/Sign_Language_Conversation_app)

## 🔮 Future Enhancements

- [ ] Add more sign language gestures and words
- [ ] Implement sentence grammar correction
- [ ] Add support for multiple sign languages (ASL, BSL, ISL)
- [ ] Integrate text-to-speech for accessibility
- [ ] Mobile application development
- [ ] Real-time translation to multiple languages
- [ ] Video recording and playback features
- [ ] Gesture sequence learning (phrases/sentences)

---

**Made with ❤️ for accessible communication**
