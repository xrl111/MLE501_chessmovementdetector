import os
import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter, defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File mapping from training code
file_mapping = {
    'K.png': 'K', 'N.png': 'N', 'B_g.png': 'B', 'N_g.png': 'N', 'Q_g.png': 'Q',
    'B.png': 'B', 'R.png': 'R', 'R_g.png': 'R', 'P_g.png': 'P', 'K_g.png': 'K',
    'P.png': 'P', 'Q.png': 'Q', 'pb_g.png': 'p', 'rb.png': 'r', 'kb.png': 'k',
    'qb.png': 'q', 'qb_g.png': 'q', 'bb_g.png': 'b', 'nb_g.png': 'n', 'rb_g.png': 'r',
    'kb_g.png': 'k', 'pb.png': 'p', 'bb.png': 'b', 'nb.png': 'n'
}

# Define class names (excluding 'board' since model uses correct labels)
#class_names = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
class_names = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r'] # manual read from onx model

# Path to test images and model
TEST_IMAGE_PATH = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/templates"
MODEL_PATH = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/best.onnx"

def load_test_images(test_path):
    """Load test images and their expected labels."""
    images = []
    if not os.path.exists(test_path):
        logger.error(f"Test image path {test_path} does not exist")
        return images
    for file_name in os.listdir(test_path):
        if not file_name.lower().endswith(('.jpg', '.png')):
            logger.warning(f"Skipping non-image file {file_name}")
            continue
        # Infer class from filename if not in file_mapping
        base_name = file_name.split('.')[0]
        expected_class = file_mapping.get(file_name)
        if expected_class is None:
            if len(base_name) >= 1 and base_name[0].lower() in ['b', 'k', 'n', 'p', 'q', 'r']:
                expected_class = base_name[0].lower() if base_name[0].islower() else base_name[0].upper()
            else:
                logger.warning(f"File {file_name} not in file_mapping and class cannot be inferred, skipping")
                continue
        if expected_class not in class_names:
            logger.warning(f"Expected class {expected_class} for {file_name} not in class_names, skipping")
            continue
        img_path = os.path.join(test_path, file_name)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load image {img_path}")
            continue
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        images.append({"image": file_name, "img_data": img, "expected_class": expected_class})
    logger.info(f"Loaded {len(images)} test images from {test_path}")
    return images

def preprocess_image(img):
    """Preprocess image for ONNX model input."""
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def infer_expected_class_and_color(image):
    """Infer expected class and color from filename."""
    expected_class = file_mapping.get(image, 'unknown')
    if expected_class == 'unknown':
        base_name = image.split('.')[0]
        if len(base_name) >= 1 and base_name[0].lower() in ['b', 'k', 'n', 'p', 'q', 'r']:
            expected_class = base_name[0].lower() if base_name[0].islower() else base_name[0].upper()
    if expected_class == 'unknown':
        return expected_class, 'unknown'
    expected_color = 'white' if expected_class.isupper() else 'black'
    return expected_class, expected_color

def is_prediction_correct(pred, expected_class):
    """Check if prediction is correct (class and color)."""
    if expected_class == 'unknown':
        return False
    pred_color = 'white' if pred.isupper() else 'black'
    expected_color = 'white' if expected_class.isupper() else 'black'
    return pred == expected_class and pred_color == expected_color

def main():
    # Install onnxruntime if not already installed
    os.system("pip install -q onnxruntime")

    # Load the ONNX model
    try:
        session = ort.InferenceSession(MODEL_PATH)
        logger.info(f"ONNX model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load ONNX model from {MODEL_PATH}: {e}")
        return

    # Load test images
    test_images = load_test_images(TEST_IMAGE_PATH)
    if not test_images:
        logger.error("No valid test images found. Aborting.")
        return

    # Perform predictions and print detailed output
    predictions = []
    high_confidence_errors = []
    CONFIDENCE_THRESHOLD = 0.15
    HIGH_CONFIDENCE_ERROR_THRESHOLD = 0.9
    print("\nDetailed Classification Output:")
    print("=" * 50)
    for item in test_images:
        try:
            img = preprocess_image(item['img_data'])
            inputs = {session.get_inputs()[0].name: img}
            outputs = session.run(None, inputs)[0]
            probs = outputs[0]
            pred_idx = np.argmax(probs)
            conf = probs[pred_idx]
            pred_class = class_names[pred_idx]
            expected_class, expected_color = infer_expected_class_and_color(item['image'])
            is_correct = is_prediction_correct(pred_class, expected_class)

            # Get top-3 predictions
            top3_indices = np.argsort(probs)[-3:][::-1]
            top3_preds = [(class_names[i], probs[i]) for i in top3_indices]

            # Print detailed output
            print(f"Image: {item['image']}")
            print(f"Expected Class: {expected_class} ({expected_color})")
            print(f"Predicted Class: {pred_class} (Confidence: {conf:.2f})")
            print("Top-3 Predictions: " + ", ".join([f"{cls}: {prob:.2f}" for cls, prob in top3_preds]))
            print("Probabilities: " + ", ".join([f"{class_names[i]}: {probs[i]:.2f}" for i in range(len(class_names))]))
            print(f"Correct: {'Yes' if is_correct else 'No'}")
            if not is_correct and conf >= HIGH_CONFIDENCE_ERROR_THRESHOLD:
                high_confidence_errors.append({
                    'image': item['image'],
                    'pred': pred_class,
                    'conf': conf,
                    'expected_class': expected_class,
                    'expected_color': expected_color,
                    'pred_color': 'white' if pred_class.isupper() else 'black'
                })
            print("-" * 50)

            logger.info(f"{item['image']} → {pred_class} ({conf:.2f})")
            predictions.append({"image": item['image'], "pred": pred_class, "conf": conf})
        except Exception as e:
            logger.error(f"Error predicting {item['image']}: {e}")
            continue

    # Analyze predictions
    color_errors = []
    correct_preds = []
    low_confidence = []
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_confidences = defaultdict(list)
    color_confusion = defaultdict(lambda: defaultdict(int))

    for pred in predictions:
        expected_class, expected_color = infer_expected_class_and_color(pred['image'])
        if expected_class == 'unknown':
            color_errors.append({
                'image': pred['image'],
                'pred': pred['pred'],
                'conf': pred['conf'],
                'expected_class': expected_class,
                'expected_color': expected_color,
                'pred_color': 'white' if pred['pred'].isupper() else 'black',
                'error_type': 'unknown_class'
            })
            continue
        class_total[expected_class] += 1
        class_confidences[expected_class].append(pred['conf'])
        if pred['conf'] < CONFIDENCE_THRESHOLD:
            low_confidence.append(pred)
        if is_prediction_correct(pred['pred'], expected_class):
            correct_preds.append(pred)
            class_correct[expected_class] += 1
        else:
            pred_color = 'white' if pred['pred'].isupper() else 'black'
            error_type = 'color' if pred['pred'].lower() == expected_class.lower() else 'class'
            color_errors.append({
                'image': pred['image'],
                'pred': pred['pred'],
                'conf': pred['conf'],
                'expected_class': expected_class,
                'expected_color': expected_color,
                'pred_color': pred_color,
                'error_type': error_type
            })
            if error_type == 'color':
                color_confusion[expected_color][pred_color] += 1
        confusion_matrix[expected_class][pred['pred']] += 1

    # Summarize results
    print(f"Total predictions: {len(predictions)}")
    print(f"Correct predictions: {len(correct_preds)} ({len(correct_preds)/len(predictions)*100:.2f}%)")
    print(f"Total errors: {len(color_errors)} ({len(color_errors)/len(predictions)*100:.2f}%)")

    # Break down errors by type
    white_to_black = [e for e in color_errors if e['expected_color'] == 'white' and e['pred_color'] == 'black']
    black_to_white = [e for e in color_errors if e['expected_color'] == 'black' and e['pred_color'] == 'white']
    class_errors = [e for e in color_errors if e['error_type'] == 'class']

    print(f"\nWhite-to-Black errors: {len(white_to_black)}")
    print(f"Black-to-White errors: {len(black_to_white)}")
    print(f"Class errors (same color): {len(class_errors)}")
    print(f"Low confidence predictions (<{CONFIDENCE_THRESHOLD}): {len(low_confidence)}")

    # High-confidence error report
    if high_confidence_errors:
        print(f"\nHigh-Confidence Errors (>= {HIGH_CONFIDENCE_ERROR_THRESHOLD}):")
        for error in high_confidence_errors:
            print(f"Image: {error['image']}, Predicted: {error['pred']} ({error['pred_color']}), "
                  f"Expected: {error['expected_class']} ({error['expected_color']}), "
                  f"Confidence: {error['conf']:.2f}")

    # Color confusion summary
    print("\nColor Confusion Summary:")
    print(f"White predicted as Black: {color_confusion['white']['black']} (Classes: {[e['expected_class'] for e in white_to_black]})")
    print(f"Black predicted as White: {color_confusion['black']['white']} (Classes: {[e['expected_class'] for e in black_to_white]})")

    # Per-class accuracy and confidence
    print("\nPer-Class Accuracy and Confidence:")
    for cls in sorted(class_names):
        accuracy = (class_correct[cls] / class_total[cls] * 100) if class_total[cls] > 0 else 0
        avg_conf = sum(class_confidences[cls]) / len(class_confidences[cls]) if class_confidences[cls] else 0
        print(f"Class {cls}: Accuracy={accuracy:.2f}% ({class_correct[cls]}/{class_total[cls]}), Avg Confidence={avg_conf:.2f}")

    # Confusion matrix
    print("\nConfusion Matrix (True -> Predicted):")
    header = "True\\Pred | " + " | ".join(sorted(class_names))
    print(header)
    print("-" * len(header))
    for true_cls in sorted(class_names):
        row = [str(confusion_matrix[true_cls][pred_cls]) for pred_cls in sorted(class_names)]
        print(f"{true_cls:8} | {' | '.join(row)}")

    # Detailed error report
    print("\nDetailed Error Report:")
    for error in color_errors:
        print(f"Image: {error['image']}, Predicted: {error['pred']} ({error['pred_color']}), "
              f"Expected: {error['expected_class']} ({error['expected_color']}), "
              f"Confidence: {error['conf']}, Error Type: {error['error_type']}")

    # Low confidence predictions by class
    if low_confidence:
        print("\nLow Confidence Predictions (<0.15):")
        low_conf_by_class = defaultdict(list)
        for pred in low_confidence:
            expected_class, _ = infer_expected_class_and_color(pred['image'])
            low_conf_by_class[expected_class].append(pred)
        for cls in sorted(low_conf_by_class.keys()):
            print(f"\nClass {cls}:")
            for pred in low_conf_by_class[cls]:
                print(f"  Image: {pred['image']}, Predicted: {pred['pred']}, Confidence: {pred['conf']}")

    # Summarize errors by piece type
    piece_errors = Counter(error['expected_class'].lower() for error in color_errors if error['expected_class'] != 'unknown')
    print("\nErrors by Piece Type (ignoring case):")
    for piece, count in sorted(piece_errors.items()):
        print(f"{piece}: {count}")

    # Summarize prediction distribution
    pred_counts = Counter(pred['pred'] for pred in predictions)
    print("\nPrediction Distribution:")
    for piece, count in sorted(pred_counts.items()):
        print(f"{piece}: {count}")

    # Recommendations based on analysis
    print("\nRecommendations for Improvement:")
    print("- High error rate suggests model needs more training data or epochs. Increase epochs to 75-100 and TARGET_IMAGES_PER_CLASS to 1000 in training script.")
    if color_confusion['white']['black'] > 0:
        print(f"- White-to-Black errors ({color_confusion['white']['black']}) for {', '.join(set(e['expected_class'] for e in white_to_black))}. Add more white piece images (e.g., B, K, N, P, Q, R) with varied lighting to custom dataset.")
    if color_confusion['black']['white'] > 0:
        print(f"- Black-to-White errors ({color_confusion['black']['white']}) for {', '.join(set(e['expected_class'] for e in black_to_white))}. Add more black piece images (e.g., p, q, r, k) to custom dataset, especially black pawns (p) and rooks (r) to address errors like pb_g.png → P and rb.png → R.")
    if piece_errors.get('p', 0) > 0:
        print("- Black pawn (p) errors detected (e.g., pb_g.png → P). Add more black pawn images (e.g., pb_extra.png, p_new1.png) with varied lighting and verify pb.png, pb_g.png exist.")
    if piece_errors.get('r', 0) > 0:
        print("- Black rook (r) errors detected. Add more black rook images (e.g., r_new1.png, r_extra.png) to custom dataset and verify rb.png, rb_g.png exist.")
    if len(class_errors) > 0:
        print("- Class errors within same color suggest piece shape confusion. Ensure RandomBrightnessContrast is used and increase augmentations to 5 per image in training script.")
    if len(high_confidence_errors) > 0:
        print(f"- High-confidence errors (>= {HIGH_CONFIDENCE_ERROR_THRESHOLD}) detected (e.g., pb_g.png → P with 0.95). Increase RandomBrightnessContrast range and add more diverse black piece images.")
    if len(low_confidence) > len(predictions) // 2:
        print("- Many low-confidence predictions. Increase mixup to 0.7 and verify yolov8m-cls.pt was used in training.")
    if any(class_total[cls] == 0 for cls in class_names):
        missing_classes = [cls for cls in class_names if class_total[cls] == 0]
        print(f"- Missing classes in test set: {missing_classes}. Add images for these classes to custom dataset or increase oversampling in training script.")
    print("- Verify training with yolov8m-cls.pt completed successfully and exported to best.onnx. If errors persist, consider retraining with more diverse data or a larger model (e.g., yolov8l-cls.pt).")

if __name__ == "__main__":
    main()
