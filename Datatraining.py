import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process
import logging

# ================== Configuration Parameters ==================
# Set image brightness threshold (adjust as needed)
BRIGHTNESS_THRESHOLD = 30  # Range: 0-255

# Mediapipe Hands module configuration
STATIC_IMAGE_MODE = True
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# ==============================================================

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
    # Log subprocess initialization information
    logging.info(f"Subprocess {current_process().name} initialized successfully.")

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
            return False
    return True

def process_image(args):
    """
    Process a single image to extract hand landmarks.

    :param args: Tuple containing the image path and its category.
    :return: Tuple of (landmarks, category) if successful, otherwise None.
    """
    img_path, category = args
    image = cv2.imread(img_path)

    if image is None:
        logging.warning(f"Unable to read image: {img_path}")
        return None

    # Check image brightness
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
    except Exception as e:
        logging.error(f"Failed to convert image to grayscale: {img_path} - {e}")
        return None

    if avg_brightness < BRIGHTNESS_THRESHOLD:
        logging.info(f"Image too dark, skipping: {img_path} (Brightness={avg_brightness:.2f})")
        return None

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Failed to convert image to RGB: {img_path} - {e}")
        return None

    try:
        result = hands_instance.process(image_rgb)
    except Exception as e:
        logging.error(f"Mediapipe processing failed: {img_path} - {e}")
        return None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            wrist = hand_landmarks.landmark[0]
            wrist_x = wrist.x
            wrist_y = wrist.y

            # Extract relative coordinates (relative to wrist joint)
            for lm in hand_landmarks.landmark:
                x_rel = lm.x - wrist_x
                y_rel = lm.y - wrist_y
                landmarks.extend([x_rel, y_rel])

            if len(landmarks) == 42:  # 21 landmarks * 2 coordinates
                if is_valid_landmarks(landmarks):
                    return (landmarks, category)
                else:
                    logging.info(f"Invalid landmark coordinates, skipping: {img_path}")
                    return None
            else:
                logging.info(f"Mismatch in number of landmarks ({len(landmarks) // 2} landmarks), skipping: {img_path}")
                return None
    else:
        logging.info(f"No hand landmarks detected, skipping: {img_path}")
        return None

def main():
    """
    Main function to batch process ASL training images, extract hand landmarks, and save them to a CSV file.
    Utilizes multiprocessing to accelerate image processing.
    """
    # Dataset path
    dataset_path = r'C:\\Users\\Zhenyu Dai\\Desktop\\AMME4710\\Major\\archive\\asl_alphabet_train\\asl_alphabet_train'  
    categories = os.listdir(dataset_path)

    # Target save path
    save_path = os.path.join(save_dir, 'AMME4710asl_train_landmarks.csv')

    logging.info(f"Directory created: {save_dir}")

    # Prepare all image paths and corresponding categories
    image_category_pairs = []
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue
        img_names = os.listdir(category_path)
        for img_name in img_names:
            img_path = os.path.join(category_path, img_name)
            image_category_pairs.append((img_path, category))

    total_images = len(image_category_pairs)
    logging.info(f"Total number of images: {total_images}")

    # Use multiprocessing to process images
    data = []
    labels = []
    skipped_images = 0  # Counter for skipped images

    try:
        with Pool(processes=cpu_count(), initializer=initialize_hands) as pool:
            # Wrap imap_unordered with tqdm to display progress bar
            for result in tqdm(pool.imap_unordered(process_image, image_category_pairs), 
                              total=total_images, desc="Processing images"):
                if result is not None:
                    landmarks, category = result
                    data.append(landmarks)
                    labels.append(category)
                else:
                    skipped_images += 1
    except Exception as e:
        logging.error(f"Error during multiprocessing: {e}")
        return

    # Check if any data was extracted
    if not data:
        logging.error("No landmark data extracted. Please check the dataset and landmark extraction steps.")
    else:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        df['label'] = labels

        # Save to CSV file for future use
        try:
            df.to_csv(save_path, index=False)
            logging.info(f"Landmark data saved to {save_path}.")
            logging.info(f"Total images processed: {total_images}")
            logging.info(f"Successfully extracted landmarks from images: {len(data)}")
            logging.info(f"Skipped images: {skipped_images}")
            print(f"Landmark data saved to {save_path}.")
            print(f"Total images processed: {total_images}")
            print(f"Successfully extracted landmarks from images: {len(data)}")
            print(f"Skipped images: {skipped_images}")
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            print(f"Error saving file: {e}")

if __name__ == '__main__':
    main()
