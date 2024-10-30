import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import warnings
import logging  # Added import
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib  # For saving models and encoders

# ================== Configuration Parameters ==================
# Set image brightness threshold (adjust as needed)
BRIGHTNESS_THRESHOLD = 30  # Range 0-255

# Mediapipe Hands module configuration
STATIC_IMAGE_MODE = True
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# ===============================================================

# Configure logging
save_dir = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major'
log_path = os.path.join(save_dir, 'processing.log')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logging.basicConfig(
    filename=log_path,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Ignore specific UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

# Configure TensorFlow log level to reduce unnecessary output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variable for Mediapipe Hands instance in each subprocess
hands_instance = None

def initialize_hands():
    """
    Initialize Mediapipe Hands instance in each subprocess.
    
    This function is called when each subprocess starts to ensure that each process has its own hand detection instance.
    """
    global hands_instance
    hands_instance = mp_hands.Hands(
        static_image_mode=STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    # Log subprocess initialization
    logging.info(f"Subprocess {os.getpid()} initialized.")
    print(f"Subprocess {os.getpid()} initialized.")

def is_valid_landmarks(landmarks):
    """
    Check if all landmarks have reasonable relative coordinates.
    
    :param landmarks: List of relative coordinates of hand landmarks.
    :return: True if all landmarks are within the range, False otherwise.
    """
    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i], landmarks[i+1]
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0):
            logging.warning(f"Landmark coordinates out of range: x={x}, y={y}")
            print(f"Landmark coordinates out of range: x={x}, y={y}")
            return False
    return True

def calculate_distances(landmarks):
    """
    Calculate distances between landmarks.
    
    :param landmarks: List of relative coordinates of hand landmarks.
    :return: List of distances between landmarks.
    """
    distances = []
    for i in range(0, len(landmarks), 2):
        for j in range(i+2, len(landmarks), 2):
            point1 = np.array([landmarks[i], landmarks[i+1]])
            point2 = np.array([landmarks[j], landmarks[j+1]])
            distance = np.linalg.norm(point1 - point2)
            distances.append(distance)
    return distances

def main():
    """
    Main function to process training and testing data, train the model, and evaluate it.
    
    Steps:
    1. Read and filter training data.
    2. Preprocess data (label encoding, scaling, feature selection, dimensionality reduction).
    3. Train Random Forest model.
    4. Process test data (extract landmarks, visualize).
    5. Preprocess test data.
    6. Predict using the model.
    7. Evaluate model performance.
    8. Save model and preprocessing objects.
    """
    # -------------------- 1. Read Training Data --------------------
    train_csv_path = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\AMME4710asl_train_landmarks.csv'
    
    if not os.path.exists(train_csv_path):
        logging.error(f"Training CSV file does not exist: {train_csv_path}")
        print(f"Training CSV file does not exist: {train_csv_path}")
        return
    
    # Read training data
    df_train = pd.read_csv(train_csv_path)
    logging.info(f"Training data shape: {df_train.shape}")
    print(f"Training data shape: {df_train.shape}")
    
    # Filter training data to include only labels 'A' to 'Z'
    valid_labels = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'
    df_train = df_train[df_train['label'].isin(valid_labels)]
    logging.info(f"Training data shape after filtering: {df_train.shape}")
    print(f"Training data shape after filtering: {df_train.shape}")
    
    # Check CSV columns
    expected_feature_columns = [str(i) for i in range(42)]  # Assuming feature columns are '0', '1', ..., '41'
    if 'label' not in df_train.columns or len(df_train.columns) != 43:
        logging.error(f"Incorrect number of columns or missing 'label' column: {df_train.columns}")
        print(f"Incorrect number of columns or missing 'label' column: {df_train.columns}")
        return
    
    # Separate features and labels
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    
    # -------------------- 2. Data Preprocessing --------------------
    # Label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    logging.info(f"Label classes: {le.classes_}")
    print(f"Label classes: {le.classes_}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=30)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_encoded)
    
    # Dimensionality reduction (PCA)
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train_selected)
    
    # -------------------- 3. Train Random Forest Model --------------------
    logging.info("Starting training Random Forest model...")
    print("Starting training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_pca, y_train_encoded)
    logging.info("Random Forest model training completed.")
    print("Random Forest model training completed.")
    
    # Optional: Hyperparameter tuning (commented out)
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    # grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, 
    #                            cv=5, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train_pca, y_train_encoded)
    # logging.info(f"Best parameters: {grid_search.best_params_}")
    # print(f"Best parameters: {grid_search.best_params_}")
    # best_model = grid_search.best_estimator_
    
    # -------------------- 4. Process Test Data --------------------
    test_dir = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\archive\\asl_alphabet_test\\asl_alphabet_test'
    
    if not os.path.exists(test_dir):
        logging.error(f"Test directory does not exist: {test_dir}")
        print(f"Test directory does not exist: {test_dir}")
        return
    
    # Initialize Mediapipe Hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, 
                        min_detection_confidence=0.3, 
                        min_tracking_confidence=0.3) as hands:
        # Get all test image files
        test_image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logging.info(f"Number of test images: {len(test_image_files)}")
        print(f"Number of test images: {len(test_image_files)}")
        
        X_test = []
        y_test = []
        
        # Create visualization output directory
        output_dir = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\visualized_test_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for img_file in tqdm(test_image_files, desc="Processing test images"):
            img_path = os.path.join(test_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Unable to read image: {img_path}")
                print(f"Unable to read image: {img_path}")
                # Assign default landmarks for unreadable images
                X_test.append([0.0] * 42)
                label = img_file.split('_')[0].upper()
                if label in le.classes_:
                    y_test.append(label)
                else:
                    logging.warning(f"Unknown label: {label}, skipping image: {img_path}")
                    print(f"Unknown label: {label}, skipping image: {img_path}")
                continue
            
            # -------------------- Remove image preprocessing steps --------------------
            # Use the original image without any preprocessing
            
            # Optional: Check image brightness
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
            except Exception as e:
                logging.error(f"Failed to convert image to grayscale: {img_path} - {e}")
                print(f"Failed to convert image to grayscale: {img_path} - {e}")
                X_test.append([0.0] * 42)
                label = img_file.split('_')[0].upper()
                if label in le.classes_:
                    y_test.append(label)
                else:
                    logging.warning(f"Unknown label: {label}, skipping image: {img_path}")
                    print(f"Unknown label: {label}, skipping image: {img_path}")
                continue

            if avg_brightness < 25:  # Lower brightness threshold (optional)
                logging.info(f"Image too dark, assigning default landmarks: {img_path} (Brightness={avg_brightness:.2f})")
                print(f"Image too dark, assigning default landmarks: {img_path} (Brightness={avg_brightness:.2f})")
                X_test.append([0.0] * 42)  # Assign default landmarks
                label = img_file.split('_')[0].upper()
                if label in le.classes_:
                    y_test.append(label)
                else:
                    logging.warning(f"Unknown label: {label}, skipping image: {img_path}")
                    print(f"Unknown label: {label}, skipping image: {img_path}")
                continue
            
            try:
                # Convert image from BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logging.error(f"Failed to convert image to RGB: {img_path} - {e}")
                print(f"Failed to convert image to RGB: {img_path} - {e}")
                X_test.append([0.0] * 42)
                label = img_file.split('_')[0].upper()
                if label in le.classes_:
                    y_test.append(label)
                else:
                    logging.warning(f"Unknown label: {label}, skipping image: {img_path}")
                    print(f"Unknown label: {label}, skipping image: {img_path}")
                continue
            
            # Use Mediapipe to process the image and detect hand landmarks
            result = hands.process(image_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    wrist = hand_landmarks.landmark[0]
                    wrist_x = wrist.x
                    wrist_y = wrist.y

                    # Extract relative coordinates (relative to wrist)
                    for lm in hand_landmarks.landmark:
                        x_rel = lm.x - wrist_x
                        y_rel = lm.y - wrist_y
                        landmarks.extend([x_rel, y_rel])

                    if len(landmarks) == 42:  # 21 landmarks * 2 coordinates
                        if is_valid_landmarks(landmarks):
                            # Visualize landmarks
                            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                            for i in range(0, len(landmarks), 2):
                                x_rel, y_rel = landmarks[i], landmarks[i+1]
                                x = int((wrist_x + x_rel) * image_bgr.shape[1])
                                y = int((wrist_y + y_rel) * image_bgr.shape[0])
                                cv2.circle(image_bgr, (x, y), 2, (0, 255, 0), -1)
                                cv2.putText(image_bgr, str(i//2), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                            # Save visualization image for verification
                            output_path = os.path.join(output_dir, 'visualized_test_' + os.path.basename(img_path))
                            cv2.imwrite(output_path, image_bgr)
                            logging.info(f"Visualization image saved: {output_path}")
                            print(f"Visualization image saved: {output_path}")
                            
                            # Add to test set
                            X_test.append(landmarks)
                            # Extract label
                            label = img_file.split('_')[0].upper()
                            if label in le.classes_:
                                y_test.append(label)
                            else:
                                logging.warning(f"Unknown label: {label}, skipping image: {img_path}")
                                print(f"Unknown label: {label}, skipping image: {img_path}")
                        else:
                            logging.warning(f"Invalid landmark coordinates, assigning default landmarks: {img_path}")
                            print(f"Invalid landmark coordinates, assigning default landmarks: {img_path}")
                            X_test.append([0.0] * 42)  # Assign default landmarks
                            label = img_file.split('_')[0].upper()
                            if label in le.classes_:
                                y_test.append(label)
                            else:
                                logging.warning(f"Unknown label: {label}, skipping image: {img_path}")
                                print(f"Unknown label: {label}, skipping image: {img_path}")
                    else:
                        logging.warning(f"Mismatch in number of landmarks ({len(landmarks) // 2} landmarks), assigning default landmarks: {img_path}")
                        print(f"Mismatch in number of landmarks ({len(landmarks) // 2} landmarks), assigning default landmarks: {img_path}")
                        X_test.append([0.0] * 42)  # Assign default landmarks
                        label = img_file.split('_')[0].upper()
                        if label in le.classes_:
                            y_test.append(label)
                        else:
                            logging.warning(f"Unknown label: {label}, skipping image: {img_path}")
                            print(f"Unknown label: {label}, skipping image: {img_path}")
            else:
                logging.info(f"No hand landmarks detected, assigning default landmarks: {img_path}")
                print(f"No hand landmarks detected, assigning default landmarks: {img_path}")
                X_test.append([0.0] * 42)  # Assign default landmarks
                label = img_file.split('_')[0].upper()
                if label in le.classes_:
                    y_test.append(label)
                else:
                    logging.warning(f"Unknown label: {label}, skipping image: {img_path}")
                    print(f"Unknown label: {label}, skipping image: {img_path}")
    
    # -------------------- 5. Check Test Data --------------------
        if not X_test:
            logging.warning("No test landmark data extracted.")
            print("No test landmark data extracted.")
            return
        
        # Convert test data to DataFrame, ensuring column names match training data
        df_test = pd.DataFrame(X_test, columns=X_train.columns)
        
        # -------------------- 6. Test Data Preprocessing --------------------
        X_test_scaled = scaler.transform(df_test)
        
        # Feature selection
        X_test_selected = selector.transform(X_test_scaled)
        
        # Dimensionality reduction (PCA)
        X_test_pca = pca.transform(X_test_selected)
        
        # -------------------- 7. Model Prediction --------------------
        logging.info("Starting prediction on test data...")
        print("Starting prediction on test data...")
        y_pred_encoded = model.predict(X_test_pca)
        y_pred = le.inverse_transform(y_pred_encoded)
        
        # -------------------- 8. Model Evaluation --------------------
        # Encode true labels
        try:
            y_test_encoded = le.transform(y_test)
        except ValueError as e:
            logging.error(f"Label encoding error: {e}")
            print(f"Label encoding error: {e}")
            return
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        logging.info(f"Test set accuracy: {accuracy * 100:.2f}%")
        print(f"\nTest set accuracy: {accuracy * 100:.2f}%\n")
        
        # Get unique labels present in y_test and y_pred
        unique_labels = np.unique(np.concatenate((y_test_encoded, y_pred_encoded)))
        class_names_present = le.inverse_transform(unique_labels)
        
        # Print classification report
        logging.info("Classification Report:")
        logging.info(classification_report(y_test_encoded, y_pred_encoded, labels=unique_labels, target_names=class_names_present))
        print("Classification Report:")
        print(classification_report(y_test_encoded, y_pred_encoded, labels=unique_labels, target_names=class_names_present))
        
        # Print confusion matrix
        logging.info("Confusion Matrix:")
        logging.info(confusion_matrix(y_test_encoded, y_pred_encoded, labels=unique_labels))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test_encoded, y_pred_encoded, labels=unique_labels))
        
        # -------------------- 9. Save Model and Preprocessing Objects --------------------
        model_path_save = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\random_forest_model.pkl'
        label_encoder_path_save = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\label_encoder.pkl'
        scaler_path_save = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\scaler.pkl'
        selector_path_save = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\selector.pkl'
        pca_path_save = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\pca.pkl'
        
        try:
            joblib.dump(model, model_path_save)
            joblib.dump(le, label_encoder_path_save)
            joblib.dump(scaler, scaler_path_save)
            joblib.dump(selector, selector_path_save)
            joblib.dump(pca, pca_path_save)
            logging.info(f"Model saved to {model_path_save}")
            logging.info(f"Label encoder saved to {label_encoder_path_save}")
            logging.info(f"Scaler saved to {scaler_path_save}")
            logging.info(f"Selector saved to {selector_path_save}")
            logging.info(f"PCA saved to {pca_path_save}")
            print(f"Model saved to {model_path_save}")
            print(f"Label encoder saved to {label_encoder_path_save}")
            print(f"Scaler saved to {scaler_path_save}")
            print(f"Selector saved to {selector_path_save}")
            print(f"PCA saved to {pca_path_save}")
        except Exception as e:
            logging.error(f"Error saving files: {e}")
            print(f"Error saving files: {e}")

if __name__ == '__main__':
    main()
