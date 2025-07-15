import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Dataset and model paths
DATASET_DIR = "2d-chessboard-and-chess-pieces"  # Update to your dataset folder path
MODEL_PATH = "piece_classifier_model.h5"
IMG_SIZE = (80, 80)  # Match the size used in chess_movement_detector.py
NUM_CLASSES = 25  # 12 white pieces, 12 black pieces, 1 empty

def load_coco_dataset(dataset_dir=DATASET_DIR):
    """Load and preprocess the COCO dataset from the local directory."""
    classes = [
        'WhitePawn', 'WhiteRook', 'WhiteKnight', 'WhiteBishop', 'WhiteQueen', 'WhiteKing',
        'BlackPawn', 'BlackRook', 'BlackKnight', 'BlackBishop', 'BlackQueen', 'BlackKing',
        'WhitePawn_g', 'WhiteRook_g', 'WhiteKnight_g', 'WhiteBishop_g', 'WhiteQueen_g', 'WhiteKing_g',
        'BlackPawn_g', 'BlackRook_g', 'BlackKnight_g', 'BlackBishop_g', 'BlackQueen_g', 'BlackKing_g',
        'empty'
    ]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Mapping from dataset labels (e.g., 'P', 'R') to script classes
    dataset_to_script = {
        'P': ['WhitePawn', 'BlackPawn', 'WhitePawn_g', 'BlackPawn_g'],
        'R': ['WhiteRook', 'BlackRook', 'WhiteRook_g', 'BlackRook_g'],
        'N': ['WhiteKnight', 'BlackKnight', 'WhiteKnight_g', 'BlackKnight_g'],
        'B': ['WhiteBishop', 'BlackBishop', 'WhiteBishop_g', 'BlackBishop_g'],
        'Q': ['WhiteQueen', 'BlackQueen', 'WhiteQueen_g', 'BlackQueen_g'],
        'K': ['WhiteKing', 'BlackKing', 'WhiteKing_g', 'BlackKing_g'],
        'empty': ['empty']
    }
    
    images = []
    labels = []
    
    # Load COCO annotations for train and valid splits
    for split in ['train', 'valid']:
        annotation_file = os.path.join(dataset_dir, f"{split}/_annotations.coco.json")
        if not os.path.exists(annotation_file):
            logging.warning("Annotation file not found: %s", annotation_file)
            continue
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Map category IDs to dataset labels
        category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Load images and labels
        for img_info in coco_data['images']:
            img_path = os.path.join(dataset_dir, split, img_info['file_name'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.warning("Failed to load image: %s", img_path)
                continue
            
            # Get annotations for this image
            img_id = img_info['id']
            img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
            
            if not img_annotations:
                # Empty square
                label = 'empty'
                mapped_labels = dataset_to_script.get(label, ['empty'])
            else:
                # Use the first annotation's category
                category_id = img_annotations[0]['category_id']
                label = category_map.get(category_id, 'empty')
                mapped_labels = dataset_to_script.get(label, ['empty'])
            
            # Add image for each mapped label (e.g., WhiteRook, BlackRook, WhiteRook_g, BlackRook_g for 'R')
            img_resized = cv2.resize(img, IMG_SIZE) / 255.0
            for mapped_label in mapped_labels:
                images.append(img_resized)
                labels.append(class_to_idx[mapped_label])
    
    images = np.array(images)[..., np.newaxis]  # Add channel dimension
    labels = np.array(labels)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logging.info("Loaded %d training images and %d validation images", len(X_train), len(X_val))
    return X_train, X_val, y_train, y_val, classes

def build_cnn_model():
    """Build a CNN model for chess piece classification."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(dataset_dir=DATASET_DIR):
    """Train the CNN model and save it."""
    # Load dataset
    if not os.path.exists(dataset_dir):
        logging.error("Dataset directory not found: %s", dataset_dir)
        raise ValueError("Dataset directory not found")
    X_train, X_val, y_train, y_val, classes = load_coco_dataset(dataset_dir)
    
    # Build and train model
    model = build_cnn_model()
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    logging.info("Validation accuracy: %.2f%%", val_accuracy * 100)
    
    # Save model
    model.save(MODEL_PATH)
    logging.info("Model saved to %s", MODEL_PATH)
    
    return history

def main(dataset_dir=DATASET_DIR):
    """Main function to train and save the piece classifier model."""
    try:
        history = train_model(dataset_dir)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error("Error during training: %s", str(e))
        raise

if __name__ == "__main__":
    # Update DATASET_DIR to your actual dataset folder path if different
    main(dataset_dir="/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/dataset")
