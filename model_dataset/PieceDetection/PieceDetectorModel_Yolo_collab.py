import os
import subprocess
import logging
import torch
from ultralytics import YOLO
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Dataset and model paths
DATASET_DIR = "/content/chess_dataset"  # Colab working directory
MODEL_PATH = "/content/yolo_piece_classifier.pt"
IMG_SIZE = 96  # Updated to match test script's adjusted size
NUM_CLASSES = 13  # Number of classes

def download_dataset():
    """Download and extract the Roboflow dataset."""
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)
        subprocess.run(
            'curl -L "https://universe.roboflow.com/ds/lEIeDLYdtb?key=ytHQpJZNeT" > roboflow.zip; '
            'unzip roboflow.zip -d chess_dataset; rm roboflow.zip',
            shell=True, check=True, cwd="/content"
        )
        logging.info("Dataset downloaded and extracted to %s", DATASET_DIR)
    except subprocess.CalledProcessError as e:
        logging.error("Failed to download or extract dataset: %s", str(e))
        raise

def verify_dataset(dataset_dir=DATASET_DIR):
    """Verify the dataset structure and data.yaml, preserving original names."""
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(dataset_dir):
        logging.error("Dataset directory not found: %s", dataset_dir)
        raise FileNotFoundError("Dataset directory not found")
    if not os.path.exists(data_yaml):
        logging.error("data.yaml not found in %s", dataset_dir)
        raise FileNotFoundError("data.yaml not found")
    if not os.path.exists(os.path.join(dataset_dir, "train")) or not os.path.exists(os.path.join(dataset_dir, "valid")):
        logging.error("Train or valid folder missing in %s", dataset_dir)
        raise FileNotFoundError("Train or valid folder missing")

    # Verify data.yaml contents without overwriting names
    try:
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            if data_config['nc'] != NUM_CLASSES:
                logging.warning("Number of classes in data.yaml (%d) does not match expected (%d)",
                               data_config['nc'], NUM_CLASSES)
                # Preserve original names instead of overwriting
                logging.info("Using original class names from data.yaml: %s", data_config['names'])
            logging.info("Dataset verified: %s", data_config)
    except Exception as e:
        logging.error("Error reading data.yaml: %s", str(e))
        raise

def train_yolo_model(dataset_dir=DATASET_DIR):
    """Train a YOLOv8 model for chess piece detection."""
    verify_dataset(dataset_dir)

    # Load YOLOv8 nano model
    model = YOLO("yolov8n.pt")  # Pre-trained nano model

    # Train the model
    try:
        results = model.train(
            data=os.path.join(dataset_dir, "data.yaml"),
            epochs=50,
            imgsz=IMG_SIZE,  # Updated to 96
            batch=32,
            patience=10,  # Early stopping
            device="cuda" if torch.cuda.is_available() else "cpu",
            project="/content/runs/train",
            name="chess_piece_detector",
            verbose=True,
            augment=True,  # Enable augmentations
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4  # Color augmentations
        )

        # Evaluate on validation set
        metrics = model.val()
        logging.info("Validation mAP@0.5: %.2f%%", metrics.box.map50 * 100)
        logging.info("Validation mAP@0.5:0.95: %.2f%%", metrics.box.map * 100)

        # Save model
        model.save(MODEL_PATH)
        logging.info("Model saved to %s", MODEL_PATH)

        return results
    except Exception as e:
        logging.error("Error during training: %s", str(e))
        raise

def main(dataset_dir=DATASET_DIR):
    """Main function to download, verify, and train the YOLOv8 model."""
    try:
        subprocess.run("pip install ultralytics pyyaml", shell=True, check=True)

        if not os.path.exists(dataset_dir):
            download_dataset()
        else:
            logging.info("Dataset directory already exists, skipping download")

        results = train_yolo_model(dataset_dir)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error("Error in main: %s", str(e))
        raise

if __name__ == "__main__":
    main(dataset_dir=DATASET_DIR)
