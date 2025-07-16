import cv2
import numpy as np
import os
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Model path
MODEL_PATH = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/yolo_piece_classifier.pt"

# Default values
image_dir = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/templates"
threshold = 0.4  # Lowered to capture more predictions

def load_yolo_model(model_path=MODEL_PATH):
    """Load the pre-trained YOLOv8 model and verify class names."""
    piece_classes = [
        'WhitePawn', 'WhiteRook', 'WhiteKnight', 'WhiteBishop', 'WhiteQueen', 'WhiteKing',
        'BlackPawn', 'BlackRook', 'BlackKnight', 'BlackBishop', 'BlackQueen', 'BlackKing', 'empty'
    ]
    
    if not os.path.exists(model_path):
        logging.error(f"Pre-trained model not found at {model_path}")
        raise ValueError(f"Pre-trained model not found at {model_path}")
    
    logging.info(f"Loading pre-trained YOLOv8 model from {model_path}")
    try:
        model = YOLO(model_path)
        model_classes = list(model.names.values())
        if model_classes != piece_classes:
            logging.warning(f"Model classes {model_classes} differ from expected {piece_classes}")
            logging.info("Using model classes")
            piece_classes = model_classes
        class_indices = {piece: idx for idx, piece in enumerate(piece_classes)}
        return model, class_indices, piece_classes
    except Exception as e:
        logging.error(f"Failed to load YOLOv8 model: {e}")
        raise

def predict_piece(image_path, model, class_indices, threshold=0.4, debug=False):
    """Predict chess piece in a single image using the YOLOv8 model."""
    try:
        square_img = cv2.imread(image_path)
        if square_img is None or square_img.size == 0:
            logging.warning(f"Failed to load image: {image_path}")
            return None

        square_resized = cv2.resize(square_img, (96, 96))
        square_rgb = cv2.cvtColor(square_resized, cv2.COLOR_BGR2RGB)
        
        results = model.predict(square_rgb, imgsz=96, conf=threshold, verbose=False, device="mps")
        detections = results[0].boxes
        
        if len(detections) == 0:
            if debug:
                logging.debug(f"No detections in {image_path}")
            return "board"
        
        max_conf = 0
        predicted_piece = None
        idx_to_piece = {idx: piece for piece, idx in class_indices.items()}
        
        for box in detections:
            conf = float(box.conf)
            class_idx = int(box.cls)
            piece = idx_to_piece.get(class_idx, "board")
            if conf > max_conf and piece != "board":
                max_conf = conf
                predicted_piece = piece
        
        if debug:
            logging.info(f"{image_path}: Predicted {predicted_piece} (Confidence: {max_conf:.2f})")
            results[0].save(f"./test_detection_{os.path.basename(image_path)}")
        
        return predicted_piece if predicted_piece and max_conf >= threshold else "board"
    except Exception as e:
        logging.warning(f"Prediction failed for {image_path}: {e}")
        return None

def verify_model(image_dir, expected_labels, model, class_indices, threshold=0.4):
    """Verify model performance against a set of images with known labels."""
    correct = 0
    total = 0
    confusion_matrix = np.zeros((len(class_indices), len(class_indices)), dtype=int)
    class_names = list(class_indices.keys())
    
    for image_file, true_label in expected_labels.items():
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            logging.warning(f"Image not found: {image_path}")
            continue
        
        predicted_label = predict_piece(image_path, model, class_indices, threshold, debug=True)
        if predicted_label is None:
            continue
        
        true_idx = class_indices[true_label]
        pred_idx = class_indices.get(predicted_label, class_indices["board"])
        confusion_matrix[true_idx][pred_idx] += 1
        
        if predicted_label == true_label:
            correct += 1
        total += 1
        logging.info(f"{image_file}: True={true_label}, Predicted={predicted_label}")
    
    accuracy = correct / total if total > 0 else 0
    logging.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    logging.info("Confusion Matrix:")
    for i in range(len(class_names)):
        row = [f"{confusion_matrix[i][j]}" for j in range(len(class_names))]
        logging.info(f"{class_names[i]}: {' '.join(row)}")
    
    return accuracy, confusion_matrix

if __name__ == "__main__":
    image_dir = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/templates"
    threshold = 0.4
    
    expected_labels = {
        "B_g.png": "B",  # WhiteBishop
        "K_g.png": "K",  # WhiteKing
        "N_g.png": "N",  # WhiteKnight
        "P_g.png": "P",  # WhitePawn
        "Q_g.png": "Q",  # WhiteQueen
        "R_g.png": "R",  # WhiteRook
        "b_g.png": "b",  # BlackBishop
        "k_g.png": "k",  # BlackKing
        "n_g.png": "n",  # BlackKnight
        "p_g.png": "p",  # BlackPawn
        "q_g.png": "q",  # BlackQueen
        "r_g.png": "r",  # BlackRook
        "empty_1.png": "board",  # Assuming 'board' is empty
    }
    
    model, class_indices, piece_classes = load_yolo_model()
    accuracy, confusion_matrix = verify_model(image_dir, expected_labels, model, class_indices, threshold)
    print(f"Model verification complete. Accuracy: {accuracy:.2%}")
