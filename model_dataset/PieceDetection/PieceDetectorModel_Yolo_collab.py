import os
import logging
import shutil
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_dataset(dataset_path):
    """Inspect class IDs from detection labels."""
    class_ids = defaultdict(int)
    sample_labels = []
    for split in ['train', 'valid']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(label_dir):
            logger.warning(f"Label directory not found: {label_dir}")
            continue
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue
            with open(os.path.join(label_dir, label_file), 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                try:
                    class_id = int(lines[0].strip().split()[0])
                    class_ids[class_id] += 1
                    if len(sample_labels) < 5:
                        sample_labels.append(f"{split}/{label_file}: {lines[0].strip()}")
                except (ValueError, IndexError):
                    logger.warning(f"Invalid label format in {label_file}")
    logger.info(f"Unique class IDs: {dict(class_ids)}")
    logger.info("Sample labels:\n" + "\n".join(sample_labels))
    return class_ids

def preprocess_dataset(dataset_path, output_path, class_names):
    """Convert detection dataset to classification dataset."""
    for split in ['train', 'valid']:
        for cls in class_names:
            os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)
    
    image_counts = {split: {cls: 0 for cls in class_names} for split in ['train', 'valid']}
    skipped_images = []
    for split in ['train', 'valid']:
        image_dir = os.path.join(dataset_path, split, 'images')
        label_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            logger.warning(f"Missing directory: {image_dir} or {label_dir}")
            continue
        for img_file in os.listdir(image_dir):
            if not img_file.endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(image_dir, img_file)
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                skipped_images.append(f"{split}/{img_file}")
                continue
            try:
                with open(label_path, 'r') as f:
                    line = f.readline()
                    if not line:
                        skipped_images.append(f"{split}/{img_file}")
                        continue
                    parts = line.strip().split()
                    if len(parts) < 5:
                        skipped_images.append(f"{split}/{img_file}")
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    class_id = int(class_id)
                    if class_id >= len(class_names):
                        skipped_images.append(f"{split}/{img_file}")
                        continue
                    target_cls = class_names[class_id]

                img = cv2.imread(img_path)
                if img is None:
                    skipped_images.append(f"{split}/{img_file}")
                    continue
                h, w = img.shape[:2]
                x_center, y_center = int(x_center * w), int(y_center * h)
                width, height = int(width * w), int(height * h)
                x1, y1 = max(0, x_center - width // 2), max(0, y_center - height // 2)
                x2, y2 = min(w, x_center + width // 2), min(h, y_center + height // 2)
                if x1 >= x2 or y1 >= y2:
                    skipped_images.append(f"{split}/{img_file}")
                    continue
                cropped_img = img[y1:y2, x1:x2]
                dst_path = os.path.join(output_path, split, target_cls, img_file)
                if not cv2.imwrite(dst_path, cropped_img):
                    skipped_images.append(f"{split}/{img_file}")
                    continue
                image_counts[split][target_cls] += 1
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
                skipped_images.append(f"{split}/{img_file}")
    
    for split in ['train', 'valid']:
        logger.info(f"Image counts for {split}: {image_counts[split]}")
        for cls in class_names:
            if image_counts[split][cls] == 0:
                logger.warning(f"No images for class '{cls}' in {split}")
    if skipped_images:
        logger.info(f"Skipped {len(skipped_images)} images. Example: {skipped_images[:5]}")
    
    return image_counts

def main():
    # Step 1: Install dependencies
    os.system("pip install -q ultralytics opencv-python")

    # Step 2: Download dataset
    try:
        os.system('curl -L "https://universe.roboflow.com/ds/lEIeDLYdtb?key=ytHQpJZNeT" -o roboflow.zip')
        os.system('unzip -q roboflow.zip')
        os.system('rm roboflow.zip')
        dataset_path = "/content"
        logger.info(f"Dataset downloaded to {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return

    # Step 3: Validate structure
    if not os.path.exists(os.path.join(dataset_path, 'train')) or not os.path.exists(os.path.join(dataset_path, 'valid')):
        logger.error("Missing 'train' or 'valid' folders")
        return

    # Step 4: Define class names (label index must match order)
    class_names = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R', 'board']  # 13 classes

    # Step 5: Inspect dataset
    class_ids = inspect_dataset(dataset_path)
    if max(class_ids.keys(), default=-1) >= len(class_names):
        logger.warning("Class IDs exceed available class_names list. Check alignment.")

    # Step 6: Preprocess to classification dataset
    output_path = "/content/chess_classification_dataset"
    image_counts = preprocess_dataset(dataset_path, output_path, class_names)

    # Step 7: Final checks
    train_dir = os.path.join(output_path, 'train')
    valid_dir = os.path.join(output_path, 'valid')
    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        logger.error("Preprocessed folders missing.")
        return
    if sum(image_counts['train'].values()) == 0 or sum(image_counts['valid'].values()) == 0:
        logger.error("No images after preprocessing. Aborting.")
        return

    # Step 8: Load classifier and train
    try:
        model = YOLO("yolov8n-cls.pt")
        logger.info("YOLOv8n-cls model loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    try:
        model.train(
            data=output_path,  # DO NOT use data.yaml
            epochs=50,
            imgsz=224,
            batch=16,
            name="chess_piece_classifier",
            patience=10,
            device=0,  # GPU
            optimizer="Adam",
            lr0=0.001
        )
        logger.info("Training completed.")

        # Step 9: Evaluate
        metrics = model.val()
        logger.info(f"Validation Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
        logger.info(f"Validation Top-5 Accuracy: {metrics.top5 * 100:.2f}%")

        # Step 10: Export
        model.export(format="onnx")
        logger.info("Model exported to ONNX.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main()
