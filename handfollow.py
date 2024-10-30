import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import warnings
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# ================== Configuration Parameters ==================
# Set image brightness threshold (adjust as needed)
BRIGHTNESS_THRESHOLD = 30  # Range: 0-255

# Mediapipe Hands module configuration
STATIC_IMAGE_MODE = False  # Set to False for real-time mode
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3
# =============================================================

# Ignore specific UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

# Configure TensorFlow log level to reduce unnecessary output (if using TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths to the model and preprocessing objects
model_path = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\random_forest_model.pkl'
label_encoder_path = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\label_encoder.pkl'
scaler_path = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\scaler.pkl'
selector_path = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\selector.pkl'
pca_path = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\pca.pkl'

# Check if all necessary files exist
for path in [model_path, label_encoder_path, scaler_path, selector_path, pca_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load the model and preprocessing objects
model = joblib.load(model_path)
le = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)
selector = joblib.load(selector_path)
pca = joblib.load(pca_path)

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def process_landmarks(landmarks):
    """
    Process hand landmarks by normalizing their positions relative to the wrist (landmark 0)
    and converting them into a one-dimensional array.

    :param landmarks: List of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark
    :return: List of normalized coordinates
    """
    # Landmark 0 is the Wrist
    base_x = landmarks[0].x
    base_y = landmarks[0].y

    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.extend([lm.x - base_x, lm.y - base_y])

    return normalized_landmarks

# Set detection frequency to 100Hz (process a frame every 0.01 seconds)
DETECTION_INTERVAL = 0.01  # Seconds
last_detection_time = time.time()

with mp_hands.Hands(static_image_mode=STATIC_IMAGE_MODE,
                    max_num_hands=MAX_NUM_HANDS, 
                    min_detection_confidence=MIN_DETECTION_CONFIDENCE, 
                    min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot read from webcam")
            break

        # Flip the image horizontally for a mirror effect and convert BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hand landmarks
        result = hands.process(rgb_frame)

        current_time = time.time()
        if current_time - last_detection_time >= DETECTION_INTERVAL:
            last_detection_time = current_time

            # Check if any hands are detected
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw hand landmarks and connections on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Process the landmarks
                    landmarks = process_landmarks(hand_landmarks.landmark)
                    if len(landmarks) != 42:
                        print("Mismatch in number of landmarks, cannot predict")
                        continue

                    # Convert to NumPy array
                    input_data = np.array(landmarks).reshape(1, -1)

                    # Apply preprocessing steps
                    input_scaled = scaler.transform(input_data)
                    input_selected = selector.transform(input_scaled)
                    input_pca = pca.transform(input_selected)

                    # Predict the gesture
                    try:
                        prediction_encoded = model.predict(input_pca)
                        prediction = le.inverse_transform(prediction_encoded)[0]
                        print(f"Predicted Gesture: {prediction}")

                        # Display the prediction on the frame
                        cv2.putText(frame, f'Gesture: {prediction}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Prediction error: {e}")
            else:
                print("No hand landmarks detected")
                # Display 'nothing' on the frame
                cv2.putText(frame, 'Gesture: nothing', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the processed frame
        cv2.imshow('Hand Tracking', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
